"""Frozen split for the improved depth CNN, column graph, and transformer.

The 6-epoch run was scored on seeds 800_000–800_003. This recipe uses more
epochs and an equal count of each class. Its holdout starts at 900_000 and
does not reuse that block. Training draws `epoch * 1000 + index`.
"""

from __future__ import annotations

DEEP_EPOCHS = 48
DEEP_PER_EVENT = 64
DEEP_BATCH = 64
DEEP_EVAL_SEEDS = tuple(range(900_000, 900_004))
DEEP_PUBLISHED_EVAL = tuple(range(800_000, 800_004))
DEEP_MAX_CABLES = 40


def train_seed(epoch: int, index: int) -> int:
    return int(epoch) * 1000 + int(index)
