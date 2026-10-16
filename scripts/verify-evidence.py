"""Verify ordinary research JSON offline against a prepared native record and run.

No server, account, database, model execution or write operation is required.
This checks consistency, not authorship, expert judgment or model accuracy.
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "services/web-api"))
from workbench.case_bundle import read_bundle
from workbench.evidence_check import verify_evidence
from workbench.vendor.data import load_record


def read_json(path):
    payload = path.read_bytes()
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError("Evidence input exceeds 16 MiB")
    return json.loads(payload.decode("utf-8-sig")), hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        type=Path,
        required=True,
        help="Prepared record directory containing manifest.json",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--report", type=Path)
    inputs.add_argument(
        "--bundle", type=Path, help="DTS case ZIP; files are never extracted"
    )
    parser.add_argument(
        "--run",
        type=Path,
        help="Saved GET /api/v1/runs/{id} JSON; not just import metadata",
    )
    args = parser.parse_args()
    if args.bundle and args.run:
        parser.error("--bundle already contains its selected run; do not pass --run")
    try:
        manifest = None
        if args.bundle:
            report, run, manifest = read_bundle(args.bundle)
            report_sha = manifest["files"]["evidence.json"]["sha256"]
            run_sha = manifest["files"].get("run.json", {}).get("sha256")
        else:
            report, report_sha = read_json(args.report)
            run, run_sha = read_json(args.run) if args.run else (None, None)
        result = verify_evidence(report, load_record(args.record, verify=True), run)
    except (ValueError, KeyError, TypeError, OSError, OverflowError, zipfile.BadZipFile):
        # Do not echo input contents, arbitrary paths or potentially sensitive run data.
        parser.exit(
            1, "Evidence verification failed: incompatible or malformed input.\n"
        )
    print(
        json.dumps(
            {
                "verified": True,
                "report_file_sha256": report_sha,
                "run_file_sha256": run_sha,
                "case_id": manifest["case_id"] if manifest else None,
                **result,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
