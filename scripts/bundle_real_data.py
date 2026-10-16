"""Bundle the verified real-data handoff, including small research weights."""
from pathlib import Path
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiber.io import file_sha256

pack = ROOT/"artifacts/real-data-v2"
verification = json.loads((pack/"verification.json").read_text())
assert verification["all_crop_arrays_equal_native"]
assert verification["pack_manifest_sha256"] == file_sha256(pack/"manifest.json")
archive = ROOT/"artifacts/zakhar-real-data-v2.zip"
if archive.exists():
    raise SystemExit("Archive already exists; preserve the completed handoff.")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as zipped:
    for path in sorted(pack.rglob("*")):
        if path.is_file():
            zipped.write(path, "real-data-v2/"+path.relative_to(pack).as_posix())
    for path in sorted((ROOT/"artifacts/real-reconstruction-v2").glob("*.json")):
        zipped.write(path, "real-reconstruction-v2/"+path.name)
    for filename in ("REAL_DATA_WORK.md", "REAL_DATA_GOAL.md"):
        zipped.write(ROOT/"docs"/filename, "docs/"+filename)
    for path in sorted((ROOT/"docs/images/real-data").glob("*.png")):
        zipped.write(path, "docs/images/real-data/"+path.name)
with zipfile.ZipFile(archive) as zipped:
    assert zipped.testzip() is None
receipt = {"archive": archive.name, "bytes": archive.stat().st_size,
           "sha256": file_sha256(archive), "verified_crops": verification["verified_crops"],
           "includes_real_only_reconstruction_weights": True,
           "expert_labels_created": 0, "synthetic_training_used": False}
(ROOT/"docs/evidence/real_data_archive.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt), flush=True)
