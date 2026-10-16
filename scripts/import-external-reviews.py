"""Verify external review material against native patches; never creates reviews."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "web-api"))
from workbench.analysis_results import sha
from workbench.platform import Platform
from workbench.store import Store
from workbench.vendor.data import parse_time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reviews", type=Path, required=True)
parser.add_argument("--pack", type=Path, required=True)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--source-commit", required=True)
parser.add_argument("--project", required=True)
parser.add_argument("--validate-only", action="store_true")
args = parser.parse_args()
if args.project != "local":
    parser.error("Native archive belongs to local project")
runtime = Path(os.environ.get("DTS_RUNTIME", ".runtime"))
store = Store(args.records, runtime / "external-review-validation")
proposals = {
    v["id"]: v
    for line in (args.pack / "weak_labels.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
    for v in [json.loads(line)]
}
prepared = []
for line in args.reviews.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    material = json.loads(line)
    proposal = proposals[material["id"]]
    if (
        material["source_sha256"] != proposal["source_sha256"]
        or material["patch_sha256"] != proposal["patch_sha256"]
    ):
        raise ValueError("Review source or patch SHA differs")
    path = args.pack / "patches" / (material["id"] + ".npz")
    if (
        path.parent.resolve() != (args.pack / "patches").resolve()
        or sha(path) != material["patch_sha256"]
    ):
        raise ValueError("Patch identity differs")
    native = store.get(proposal["record"])
    if native.manifest["source_sha256"] != material["source_sha256"]:
        raise ValueError("Native SHA differs")
    rs, re = proposal["context_rows_half_open"]
    cs, ce = proposal["context_columns_half_open"]
    expected_time=native.elapsed_s[rs:re]
    if proposal.get("time_basis") == "utc_epoch":
        first=parse_time(native.profiles[0]["timestamp"])
        if first.tzinfo is None:raise ValueError("Cannot translate unknown timezone to epoch")
        expected_time=expected_time+first.timestamp()
    with np.load(path, allow_pickle=False) as patch:
        if (
            not np.array_equal(patch["x_m"], native.x_m[cs:ce])
            or not np.array_equal(patch["time_s"], expected_time)
            or not np.array_equal(
                patch["valid"], native.invalid_reason[rs:re, cs:ce] == 0
            )
            or not np.array_equal(
                patch["raw_c"],
                native.raw_c[rs:re, cs:ce].astype(np.float32),
                equal_nan=True,
            )
        ):
            raise ValueError("Patch does not match native cells")
    identity = hashlib.sha256(
        (
            args.project
            + args.source_commit
            + material["patch_sha256"]
            + sha(args.reviews)
        ).encode()
    ).hexdigest()
    prepared.append(
        {
            "id": identity,
            "project_id": args.project,
            "source_sha256": material["source_sha256"],
            "patch_sha256": material["patch_sha256"],
            "source_commit": args.source_commit,
            "source_file_sha256": sha(args.reviews),
            "review_id": material["id"],
            "declared_reviewer": material.get("reviewer"),
            "notes": material.get("notes", ""),
            "presence": material.get("presence"),
            "shape": material.get("shape"),
            "region": {
                "record_id": proposal["record"],
                "row_start": rs,
                "row_end": re,
                "column_start": cs,
                "column_end": ce,
            },
            "origin": "external_material",
            "authenticated_expert_answer": False,
        }
    )
if not args.validate_only:
    platform = Platform(os.environ["DTS_DATABASE_URL"], runtime)
    for value in prepared:
        platform.put("external_review", value["id"], value)
print(json.dumps({"verified_materials": len(prepared), "expert_reviews_created": 0}))
