"""Scan raw records with the depth CNN only if planted search chose it."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.classify.deep_scan import main_scan

if __name__ == "__main__":
    main_scan("depth")
