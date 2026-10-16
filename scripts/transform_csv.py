"""Transform MGTS CSV files into native numpy records."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.io import transform_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "MGTS_data")
    parser.add_argument("--out-root", type=Path, default=ROOT / "artifacts" / "records")
    args = parser.parse_args()
    started = time.perf_counter()
    reports = transform_all(args.raw_root, args.out_root)
    elapsed = time.perf_counter() - started
    print(json.dumps({"elapsed_s": round(elapsed, 2), "records": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
