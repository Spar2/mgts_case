"""Persist CLI replay samples; keep reports outside the browser's ephemeral memory."""

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(".runtime/replay-final.json")
    )
    parser.add_argument(
        "--package",
        choices=["@playwright/cli", "@playwright/cli@0.1.22"],
        default="@playwright/cli@0.1.22",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.session):
        parser.error("Use a simple CLI session name")
    output = args.output.resolve()
    workspace = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(workspace / ".runtime"):
        parser.error("Output must stay inside repository .runtime")
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        parser.error("Install Node/npm first")
    deadline = time.monotonic() + 35 * 60
    command = [
        npx,
        "--yes",
        "--package",
        args.package,
        "playwright-cli",
        "--session",
        args.session,
        "eval",
        "window.__dtsSoak",
    ]
    while time.monotonic() < deadline:
        result = subprocess.run(
            command, capture_output=True, encoding="utf-8", timeout=45, check=True
        )
        match = re.search(
            r"### Result\s*\n(.*?)\n### Ran Playwright code", result.stdout, re.DOTALL
        )
        if not match:
            raise RuntimeError("No replay state returned; keep the CLI session open")
        report = json.loads(match.group(1))
        report.pop("frames", None)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
        last = report.get("samples", [{}])[-1]
        print(
            json.dumps(
                {
                    "done": report["done"],
                    "elapsed_s": last.get("elapsedS"),
                    "heap_mib": last.get("heapMiB"),
                }
            ),
            flush=True,
        )
        if report["done"]:
            return
        time.sleep(30)
    raise TimeoutError(
        "Replay did not finish within 35 minutes; partial report retained"
    )


if __name__ == "__main__":
    main()
