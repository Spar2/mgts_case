"""Fit Neighbor2Neighbor-TSR. CUDA when the machine has it, unless --device cpu."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.io import Record
from fiber.models.n2n_tsr import train_n2n_tsr
from fiber.models.zsn2n import train_zsn2n


def _crop():
    path = ROOT / "artifacts" / "records" / "5"
    if not (path / "raw_c.npy").exists():
        raise SystemExit("MGTS record 5 is missing; n2n_tsr was not trained")
    record = Record.load(path)
    raw = record.raw_c[:64, 250:410]
    valid = record.valid[:64, 250:410]
    return raw, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    raw, valid = _crop()
    dest = ROOT / "models" / "weights" / "n2n_tsr.pt"
    train_n2n_tsr(raw, valid, dest, seed=args.seed, steps=args.steps, device=args.device)
    zdest = ROOT / "models" / "weights" / "zsn2n.pt"
    train_zsn2n(raw, valid, zdest, seed=args.seed, steps=args.steps, device=args.device)
    print(f"wrote {dest} and {zdest}", flush=True)


if __name__ == "__main__":
    main()
