"""Package/validate/import a native-grid calculation; never runs a model."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "web-api"))
from workbench.analysis_results import import_package, sha, validate
from workbench.platform import Platform
from workbench.store import Store

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=["package", "validate", "import"])
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--metadata", type=Path)
parser.add_argument("--project")
parser.add_argument("--author", help="Existing account ID, not a declared display name")
args = parser.parse_args()
runtime = Path(os.environ.get("DTS_RUNTIME", ".runtime"))
store = Store(args.records, runtime / "analysis-validation")
if args.action == "package":
    if not args.metadata or (args.directory / "manifest.json").exists():
        parser.error("Provide metadata and a directory without an existing manifest")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    metadata["schema_version"] = "dts-analysis-result-v2"
    metadata["files"] = {
        p.name: sha(p)
        for p in args.directory.iterdir()
        if p.suffix in {".npy", ".json"}
    }
    output = args.directory / "manifest.json"
    output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    try:
        validate(args.directory, store)
    except Exception:
        output.unlink()
        raise
    print("Package validated")
elif args.action == "validate":
    manifest, layers, zones = validate(args.directory, store)
    print(
        json.dumps(
            {
                "verified": True,
                "source_sha256": manifest.source_sha256,
                "layers": layers,
                "zone_count": len(zones),
            }
        )
    )
else:
    if not args.project or not args.author:
        parser.error("Import requires explicit --project and --author")
    platform = Platform(os.environ["DTS_DATABASE_URL"], runtime)
    result = import_package(platform, store, args.directory, args.project, args.author)
    print(
        json.dumps({"run_id": result["id"], "url": "/analysis?run_id=" + result["id"]})
    )
