"""Operator import of a verified immutable queue, never annotations or CSV uploads."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services/web-api"))
from workbench.platform import Platform
from workbench.review_packages import publish_queue
from workbench.store import Store

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--project", required=True)
parser.add_argument("--author", required=True)
parser.add_argument("--title", required=True)
args = parser.parse_args()
if args.project != "local":
    parser.error("Native archive belongs to local project")
runtime = Path(os.environ.get("DTS_RUNTIME", ".runtime"))
platform = Platform(os.environ["DTS_DATABASE_URL"], runtime)
store = Store(args.records, runtime / "queue-import-cache")
with platform.connect() as db:
    if not db.execute(
        "SELECT 1 FROM memberships m JOIN users u ON u.id=m.user_id WHERE m.user_id=%s AND m.project_id=%s AND m.role IN ('analyst','admin') AND u.enabled",
        (args.author, args.project),
    ).fetchone():
        raise ValueError("Enabled project analyst/admin required")
    result = publish_queue(
        platform, store, args.directory, args.project, args.title, db
    )
    db.execute(
        "INSERT INTO audit(project_id,actor,action,resource,at) VALUES(%s,%s,'review_package.imported',%s,extract(epoch from now()))",
        (args.project, args.author, result["id"]),
    )
print(
    "Review package:",
    result["id"],
    "cards:",
    result["cards"],
    "expert labels created: 0",
)
