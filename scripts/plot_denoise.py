"""Save before/after slices for the 5.csv witnesses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "artifacts" / "records" / "5")
    parser.add_argument("--denoise", type=Path, default=ROOT / "artifacts" / "denoise" / "5")
    parser.add_argument("--method", default="median_x3")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "reports")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    from fiber.io import Record

    rec = Record.load(args.record)
    denoised = np.load(args.denoise / f"{args.method}.npy")
    args.out.mkdir(parents=True, exist_ok=True)

    def _save(path: Path, title: str, sl, x_m, raw, clean):
        xs = x_m[sl]
        extent = [float(xs[0]), float(xs[-1]), 0, raw.shape[0]]
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        im0 = ax[0].imshow(raw[:, sl], aspect="auto", origin="lower", cmap="magma", extent=extent)
        im1 = ax[1].imshow(clean[:, sl], aspect="auto", origin="lower", cmap="magma", extent=extent)
        ax[0].set_title(f"raw · {title}")
        ax[1].set_title(f"{args.method} · {title}")
        ax[1].set_xlabel("distance along fibre, m")
        ax[0].set_ylabel("profile index")
        ax[1].set_ylabel("profile index")
        fig.colorbar(im0, ax=ax[0], fraction=0.02, label="°C")
        fig.colorbar(im1, ax=ax[1], fraction=0.02, label="°C")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print("wrote", path)

    x = rec.x_m
    heat = (x >= 290) & (x <= 310)
    _save(args.out / "5_heat_297_303.png", "5.csv 290-310 m", heat, x, rec.raw_c, denoised)
    tail = (x >= 1600) & (x <= 1800)
    raw_show = np.where(rec.valid, rec.raw_c, np.nan)
    _save(
        args.out / "5_tail_dead.png",
        "5.csv 1600-1800 m, sentinel masked",
        tail, x, raw_show, np.where(rec.valid, denoised, np.nan),
    )


if __name__ == "__main__":
    main()
