"""Read-only UI baseline check. No automatic update or permission override."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/web/design-system/frozen-ui.json"
TEXT_SUFFIXES = {".ts", ".tsx", ".scss", ".css", ".json", ".html", ".md", ".py", ".js", ".yml", ".yaml"}


def canonical_bytes(path: Path) -> bytes:
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError("Protected file resolves outside repository")
    value = path.read_bytes()
    return value.replace(b"\r\n", b"\n") if path.suffix.lower() in TEXT_SUFFIXES else value


def inventory(manifest: dict) -> set[str]:
    paths = set(manifest["protected_files"])
    for directory in manifest["protected_directories"]:
        for path in (ROOT / directory).rglob("*"):
            if path.is_file():
                paths.add(path.relative_to(ROOT).as_posix())
    return paths


def check() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "dts-frozen-ui-v1":
        raise ValueError("Unknown UI freeze schema")
    expected = manifest["sha256_lf"]
    current = inventory(manifest)
    problems = []
    for name in sorted(current | set(expected)):
        path = ROOT / name
        if name not in expected:
            problems.append(f"ADDED {name}")
        elif not path.is_file():
            problems.append(f"MISSING {name}")
        elif hashlib.sha256(canonical_bytes(path)).hexdigest() != expected[name]:
            problems.append(f"CHANGED {name}")
    if problems:
        print("Frozen web UI differs from the approved baseline:")
        print("\n".join(problems))
        print("Read docs/web/UI-FREEZE.md. Do not regenerate the baseline without explicit user authorization.")
        return 1
    print(f"UI freeze intact: {len(expected)} files ({manifest['baseline_id']}).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(check())
    except (OSError, ValueError, KeyError) as error:
        print(f"UI freeze check could not complete: {error}", file=sys.stderr)
        sys.exit(2)
