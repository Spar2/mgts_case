"""Train the texture logistic once and score the eight real witnesses.

The recipe is frozen in fiber.classify.texture. This script does not take
epoch or seed overrides.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.classify.texture_train import main


if __name__ == "__main__":
    main()
