"""Encrypted daily control-plane snapshots; private decryption key stays off server."""

import argparse
import json
import os
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .maintenance import backup
from .platform import Platform


def snapshot():
    root = Path(os.environ.get("DTS_BACKUP_ROOT", "/backups"))
    root.mkdir(parents=True, exist_ok=True)
    recipient = os.environ["DTS_BACKUP_RECIPIENT"]
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    final = root / ("dts-" + day + ".tar.age")
    runtime = Path(os.environ.get("DTS_RUNTIME", "/runtime"))
    platform = Platform(os.environ["DTS_DATABASE_URL"], runtime)
    with platform.connect() as db:
        db.execute(
            "DELETE FROM access_requests WHERE closed<%s", (time.time() - 30 * 86400,)
        )
    if final.exists():
        return
    with tempfile.TemporaryDirectory(prefix="dts-backup-") as temporary:
        temporary = Path(temporary)
        backup(platform, temporary / "snapshot")
        with tarfile.open(temporary / "snapshot.tar", "w") as archive:
            archive.add(temporary / "snapshot", arcname="snapshot")
        pending = root / (final.name + ".pending")
        try:
            subprocess.run(
                [
                    "age",
                    "-r",
                    recipient,
                    "-o",
                    str(pending),
                    str(temporary / "snapshot.tar"),
                ],
                check=True,
                capture_output=True,
            )
            os.replace(pending, final)
        finally:
            pending.unlink(missing_ok=True)
    # Only prune our strictly named daily archives after a successful encrypted write.
    archives = sorted(p for p in root.glob("dts-????-??-??.tar.age") if p.is_file())
    for old in archives[:-7]:
        old.unlink()
    marker = runtime / "backup-latest.json"
    marker.write_text(
        json.dumps(
            {
                "completed_at": time.time(),
                "filename": final.name,
                "bytes": final.stat().st_size,
            }
        )
    )
    print(
        "Encrypted daily snapshot complete; retain seven copies and copy off-server.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            snapshot()
        except Exception:
            if not args.daily:
                raise
            print(
                "Daily snapshot failed; inspect storage and age configuration. No secrets logged.",
                flush=True,
            )
            time.sleep(300)
            continue
        if not args.daily:
            return
        time.sleep(3600)


if __name__ == "__main__":
    main()
