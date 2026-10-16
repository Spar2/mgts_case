"""Plot the spatial-Z witness on 5.csv."""

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
    parser.add_argument("--anomaly", type=Path, default=ROOT / "artifacts" / "anomaly" / "5")
    parser.add_argument("--denoise", type=Path, default=ROOT / "artifacts" / "denoise" / "5" / "median_x3.npy")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "images")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    from fiber.io import Record

    rec = Record.load(args.record)
    z = np.load(args.anomaly / "z_spatial.npy")
    clean = np.load(args.denoise)
    x = rec.x_m
    sl = (x >= 280.0) & (x <= 320.0)
    xs = x[sl]
    extent = [float(xs[0]), float(xs[-1]), 0, z.shape[0]]
    args.out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    im0 = ax[0].imshow(clean[:, sl], aspect="auto", origin="lower", cmap="magma", extent=extent)
    im1 = ax[1].imshow(z[:, sl], aspect="auto", origin="lower", cmap="coolwarm", vmin=-8, vmax=8, extent=extent)
    ax[0].set_title("median_x3 · 5.csv 280–320 m")
    ax[1].set_title("spatial Z · halo background, inner 8 m / outer 40 m")
    ax[1].set_xlabel("distance along fibre, m")
    ax[0].set_ylabel("profile index")
    ax[1].set_ylabel("profile index")
    fig.colorbar(im0, ax=ax[0], fraction=0.02, label="°C")
    fig.colorbar(im1, ax=ax[1], fraction=0.02, label="Z")
    fig.tight_layout()
    path = args.out / "5_anomaly_z_297.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
