"""Verify every exported real patch against its native arrays and provenance."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiber.io import Record, file_sha256
from fiber.real_data import crop_partition, temporal_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=ROOT/"artifacts/real-data-v2")
    parser.add_argument("--experiment", type=Path, default=ROOT/"artifacts/real-reconstruction-v2")
    args = parser.parse_args()
    pack = args.pack
    manifest = json.loads((pack/"manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (pack/"weak_labels.jsonl").read_text(encoding="utf-8").splitlines()]
    templates = [json.loads(line) for line in (pack/"review_template.jsonl").read_text(encoding="utf-8").splitlines()]
    shortlist = json.loads((pack/"shortlist.json").read_text())["ids"]
    if len({r["id"] for r in rows}) != len(rows) or len(rows) != manifest["n_crops"]:
        raise ValueError("crop identity/count mismatch")
    verified = 0
    for name in manifest["sources"]:
        rec = Record.load(ROOT/"artifacts/records"/name)
        clean = np.load(ROOT/"artifacts/denoise"/name/"median_x3.npy", mmap_mode="r", allow_pickle=False)
        sources = manifest["sources"][name]
        assert sources["csv_sha256"] == file_sha256(Path(rec.source_csv))
        assert sources["raw_sha256"] == file_sha256(ROOT/"artifacts/records"/name/"raw_c.npy")
        assert sources["clean_sha256"] == file_sha256(ROOT/"artifacts/denoise"/name/"median_x3.npy")
        assert sources["quality_sha256"] == file_sha256(ROOT/"artifacts/records"/name/"quality_codes.npy")
        split, _ = temporal_partition(rec.time_s, manifest["rule_config"]["split_guard_s"])
        np.testing.assert_array_equal(split, np.load(pack/"splits"/f"{name}.npy", allow_pickle=False))
        for row in rows:
            if row["record"] != name:
                continue
            path = pack/"patches"/(row["id"]+".npz")
            assert file_sha256(path) == row["patch_sha256"]
            assert row["source_sha256"] == rec.source_sha256
            ca, cb = row["context_rows_half_open"]
            xa, xb = row["context_columns_half_open"]
            with np.load(path, allow_pickle=False) as patch:
                for field, expected in (("raw_c", rec.raw_c[ca:cb, xa:xb]), ("median_x3_c", clean[ca:cb, xa:xb]),
                    ("valid", rec.valid[ca:cb, xa:xb]), ("quality_codes", rec.quality_codes[ca:cb, xa:xb]),
                    ("x_m", rec.x_m[xa:xb]), ("time_s", rec.time_s[ca:cb]),
                    ("source_rows", np.arange(ca, cb)), ("source_columns", np.arange(xa, xb)), ("split_code", split[ca:cb])):
                    np.testing.assert_array_equal(patch[field], expected)
                assert row["split_code"] == crop_partition(split[ca:cb])
                a, b = row["rows_half_open"]
                np.testing.assert_array_equal(patch["core_rows_half_open"], [a-ca, b-ca])
            assert row["expert_label"] is None and row["expert_confirmed"] is False
            if manifest["plots_rendered"]:
                assert (pack/"cards"/(row["id"]+".png")).stat().st_size > 1000
            verified += 1
    assert len(templates) == len(rows)
    for row, template in zip(rows, templates, strict=True):
        assert template["id"] == row["id"] and template["patch_sha256"] == row["patch_sha256"]
        assert template["presence"] is None and template["shape"] is None and template["reviewer"] is None
    for key in shortlist:
        row = next(r for r in rows if r["id"] == key)
        assert row["split_code"] in (0, 1)
    assert len(shortlist) == len(set(shortlist))
    for relative, digest in manifest["code_sha256"].items():
        assert file_sha256(ROOT/relative) == digest
    report = json.loads((args.experiment/"report.json").read_text())
    assert report["model_sha256"] == file_sha256(args.experiment/"real_masked_ridge.json")
    assert report["script_sha256"] == file_sha256(ROOT/"scripts/benchmark_real_reconstruction.py")
    assert report["split_code_sha256"] == file_sha256(ROOT/"fiber/real_data.py")
    assert not report["synthetic_training_used"] and not report["operational_approved"]
    assert not report["holdout_used_for_fitting_or_selection"]
    result = {"schema": "real_data_verification_v1", "verified_crops": verified, "source_records": 6,
              "all_crop_arrays_equal_native": True, "all_patch_hashes_match": True,
              "shortlist_train_development_only": True, "expert_labels_created": 0,
              "synthetic_training_used": False, "reconstruction_model_sha256": report["model_sha256"],
              "pack_manifest_sha256": file_sha256(pack/"manifest.json")}
    (pack/"verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
