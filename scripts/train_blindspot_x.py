"""Fit the column blind-spot on MGTS records only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.io import Record
from fiber.models.blindspot_x import fit_blindspot_batch, neighbor_batch


def _batches(seed: int):
    rng = np.random.default_rng(seed)
    neighbors = []
    targets = []
    losses = []
    found = False
    for name in ("1", "2", "3", "4", "5", "6"):
        path = ROOT / "artifacts" / "records" / name
        if not (path / "raw_c.npy").exists():
            continue
        found = True
        record = Record.load(path)
        time_count = record.raw_c.shape[0]
        if time_count < 32:
            continue
        for _ in range(4):
            start = int(rng.integers(0, time_count - 32 + 1))
            distance = record.raw_c.shape[1]
            left = 0 if distance <= 128 else int(rng.integers(0, distance - 128 + 1))
            raw = record.raw_c[start:start + 32, left:left + 128]
            valid = record.valid[start:start + 32, left:left + 128]
            neigh, target, loss = neighbor_batch(raw, valid)
            neighbors.append(neigh)
            targets.append(target)
            losses.append(loss)
    if not found:
        raise SystemExit("MGTS records are missing; blindspot_x was not trained")
    return np.concatenate(neighbors), np.concatenate(targets), np.concatenate(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    neighbors, target, loss = _batches(args.seed)
    dest = ROOT / "models" / "weights" / "blindspot_x.pt"
    fit_blindspot_batch(neighbors, target, loss, dest, seed=args.seed, steps=args.steps, device=args.device)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
