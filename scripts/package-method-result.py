"""Validate/package an existing result. Does not run or train any model.

Metadata uses dts-method-result-v1 without result; result is actual JSON output.
Native source identity and single-profile shape are checked before handoff.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "services/web-api"))
from workbench.store import Store
from workbench.workspace_routes import ResultImport


def package(records, metadata, result):
    envelope = ResultImport.model_validate({**metadata, "result": result})
    # A temporary tile cache is unnecessary: Store only opens prepared mmap arrays.
    store = Store(records, BASE / ".runtime/result-adapter-cache")
    region = envelope.region
    native = store.get(region.record_id)
    if envelope.source_sha256 != native.manifest["source_sha256"]:
        raise ValueError("Result source SHA differs from the native record")
    if not (
        region.row_start < region.row_end <= native.raw_c.shape[0]
        and region.column_start < region.column_end <= native.raw_c.shape[1]
    ):
        raise ValueError("Result region lies outside native axes")
    if (region.row_end - region.row_start) * (
        region.column_end - region.column_start
    ) > 1_000_000:
        raise ValueError("Region exceeds 1,000,000 observations")
    if "denoised" in result:
        values = result["denoised"]
        if (
            region.row_end != region.row_start + 1
            or not isinstance(values, list)
            or len(values) != native.raw_c.shape[1]
        ):
            raise ValueError(
                "denoised requires one named row and a full native profile"
            )
        if any(
            v is not None
            and (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not np.isfinite(v)
            )
            for v in values
        ):
            raise ValueError("denoised must contain finite numbers or null")
    encoded = json.dumps(
        envelope.model_dump(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    if len(encoded) > 240 * 1024:
        raise ValueError(
            "Result exceeds the 240 KiB UI contract; use a dedicated binary adapter, never truncate"
        )
    return encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose another path")
    encoded = package(
        args.records,
        json.loads(args.metadata.read_text(encoding="utf-8-sig")),
        json.loads(args.result.read_text(encoding="utf-8-sig")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(
        f"Validated existing result: {len(encoded)} bytes. No model execution or training."
    )


if __name__ == "__main__":
    main()
