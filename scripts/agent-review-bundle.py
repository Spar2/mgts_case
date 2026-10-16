"""Build a bounded published-source packet for a tool-free independent reviewer."""

import argparse
import subprocess
from pathlib import Path

SECTIONS = [
    ("docs/web/RESULT-HANDOFF.md", 1, 47),
    ("services/web-api/workbench/workspace_routes.py", 25, 45),
    ("services/web-api/workbench/workspace_routes.py", 296, 430),
    ("services/web-api/workbench/demo_routes.py", 205, 315),
    ("services/web-api/workbench/platform.py", 80, 185),
    ("services/web-api/workbench/worker.py", 100, 143),
    ("apps/web/src/presentation.tsx", 180, 250),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkout", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if ".runtime" not in a.output.resolve().parts:
        p.error("Review bundle must stay in .runtime")
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=a.checkout, text=True
    ).strip()
    parts = [
        f"Independent Grok review for Sol. Source SHA {sha}. No tools; use only this published source bundle. Inspect ML -> prepared result -> browser -> standalone report scientific correctness. Up to 4 confirmed findings with exact line, trigger, minimum fix and test. Distinguish unknowns. Do not assume functions outside excerpts are missing. Never claim runtime tests. Respond in Russian."
    ]
    for name, start, end in SECTIONS:
        # Read committed bytes rather than accidentally including private edits.
        lines = (
            subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=a.checkout)
            .decode("utf-8")
            .splitlines()
        )
        parts.append(
            "\nFILE "
            + name
            + "\n"
            + "\n".join(
                f"{i + 1}: {lines[i]}" for i in range(start - 1, min(end, len(lines)))
            )
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("\n".join(parts), encoding="utf-8")
    print(f"Published-source bundle ready: {sha}; {len(SECTIONS)} sections")


if __name__ == "__main__":
    main()
