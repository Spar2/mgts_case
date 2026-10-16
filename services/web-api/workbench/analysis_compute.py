"""Pinned CPU calculation, running only in a supervised worker child."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .analysis_results import sha, validate
from .store import Store

ML_COMMIT = "237168f4d4a3e11260c9f692880cc477cd7af779"
METHOD = "median_x3+spatial_z_halo"
PIN = Path(__file__).parent / "vendor_ml"
PIN_MANIFEST_SHA = "d0ef9e817839086b40cd88d3eaf25f1ab3ec0aeff0f3a8ca1b09fbd803cc99e4"


def verify_pin():
    payload = (PIN / "source.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != PIN_MANIFEST_SHA:
        raise ValueError("Pinned source manifest differs")
    manifest = json.loads(payload)
    if manifest["commit"] != ML_COMMIT:
        raise ValueError("Pinned ML version differs")
    for name, checksum in manifest["files"].items():
        if hashlib.sha256((PIN / name).read_bytes()).hexdigest() != checksum:
            raise ValueError("Pinned ML code differs")


def package_result(native, region, result, output, commit, calculated_at):
    """Shared native package writer for operator and online calculations."""
    rs, re, cs, ce = (
        region[k] for k in ("row_start", "row_end", "column_start", "column_end")
    )
    x, t = native.x_m[cs:ce], native.elapsed_s[rs:re]
    good = native.invalid_reason[rs:re, cs:ce] == 0
    output = Path(output)
    output.mkdir(exist_ok=False)
    np.save(output / "x_m.npy", x)
    np.save(output / "time_s.npy", t)
    for name, source, availability in (
        ("processed_temperature_c", "clean", good),
        ("background_temperature_c", "background", good),
        ("score", "z", result["score_available"]),
    ):
        a = np.asarray(result[source])
        np.save(output / (name + ".npy"), a)
        np.save(output / (name + "_valid.npy"), good & availability & np.isfinite(a))
    zones = []
    for index, zone in enumerate(result["zones"]):

        def nearest(axis, value):
            return int(np.argmin(np.abs(axis - value)))

        zones.append(
            {
                **zone,
                "id": f"zone-{index + 1}",
                "row_start": rs + nearest(t, zone["t_start_s"]),
                "row_end": rs + nearest(t, zone["t_end_s"]) + 1,
                "column_start": cs + nearest(x, zone["start_m"]),
                "column_end": cs + nearest(x, zone["end_m"]) + 1,
                "boundary_origin": "nearest_native_centres_from_ml_rounded_bounds",
            }
        )
    (output / "zones.json").write_text(
        json.dumps(zones, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    manifest = {
        "schema_version": "dts-analysis-result-v2",
        "region": region,
        "source_sha256": native.manifest["source_sha256"],
        "method": METHOD,
        "method_version": commit,
        "parameters": result["manifest"],
        "calculated_at": calculated_at,
        "training_origin": "untrained",
        "research_status": "reference",
        "limitations": "CPU reference pipeline; shape hypotheses, no independent accident accuracy; rounded zone bounds mapped to nearest native centres",
        "score": {
            "name": "Spatial Z",
            "unit": "Z",
            "meaning": "Signed spatial deviation from local background",
            "kind": "signed",
            "threshold": 3.5,
            "comparison": "abs_ge",
        },
        "files": {p.name: sha(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def compute(request, output):
    verify_pin()
    sys.path.insert(0, str(PIN))
    from fiber.pipeline import analyze_field

    store = Store(request["records"], Path(output).parent / "tiles")
    region = request["region"]
    native = store.get(region["record_id"])
    if native.manifest["source_sha256"] != request["source_sha256"]:
        raise ValueError("Source SHA changed")
    rs, re, cs, ce = (
        region[k] for k in ("row_start", "row_end", "column_start", "column_end")
    )
    good = native.invalid_reason[rs:re, cs:ce] == 0
    raw = native.raw_c[rs:re, cs:ce]
    if np.any(good & (raw <= -200)):
        raise ValueError("ML convention would mask a valid measurement")
    result = analyze_field(
        raw,
        native.x_m[cs:ce],
        native.elapsed_s[rs:re],
        valid=good,
        time_iso=[p["timestamp"] for p in native.profiles[rs:re]],
    )
    package_result(
        native,
        region,
        result,
        output,
        ML_COMMIT,
        datetime.now(timezone.utc).isoformat(),
    )
    validate(output, store)


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    compute(request, Path(sys.argv[2]))
