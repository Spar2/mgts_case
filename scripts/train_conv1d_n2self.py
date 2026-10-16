"""Train the 1D residual CNN with Noise2Self on local GPU (4080 SUPER)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "artifacts" / "records")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "weights" / "conv1d_n2self.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop", type=int, default=1024)
    parser.add_argument("--mask-frac", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset

    from fiber.io import Record
    from fiber.models.conv1d_denoise import Conv1dDenoiser

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((args.records / "index.json").read_text(encoding="utf-8"))
    records = [Record.load(args.records / item["name"]) for item in index]

    class Crops(Dataset):
        def __init__(self):
            self.rows = []
            self.masks = []
            for rec in records:
                field = np.where(rec.valid, rec.raw_c, 0.0).astype(np.float32)
                mean = np.nanmean(np.where(rec.valid, rec.raw_c, np.nan))
                std = np.nanstd(np.where(rec.valid, rec.raw_c, np.nan))
                std = float(std) if np.isfinite(std) and std > 1e-3 else 1.0
                mean = float(mean) if np.isfinite(mean) else 0.0
                normed = (field - mean) / std
                width = rec.raw_c.shape[1]
                if width < args.crop:
                    continue
                for t in range(0, rec.raw_c.shape[0], 4):
                    if rec.valid[t].mean() < 0.5:
                        continue
                    start = 0
                    while start + args.crop <= width:
                        sl = slice(start, start + args.crop)
                        if rec.valid[t, sl].mean() < 0.7:
                            start += args.crop // 2
                            continue
                        self.rows.append(normed[t, sl].copy())
                        self.masks.append(rec.valid[t, sl].copy())
                        start += args.crop
            if not self.rows:
                raise RuntimeError("no training crops; transform CSV first")

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i], self.masks[i]

    data = Crops()
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True)
    model = Conv1dDenoiser().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"device={device} crops={len(data)} batches/epoch={len(loader)}")
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for row, valid in loader:
            row = row.to(device)
            valid = valid.to(device)
            hole = (torch.rand_like(row) < args.mask_frac) & valid
            if hole.sum() < 8:
                continue
            inp = row.clone()
            inp[hole] = 0.0
            pred = model(inp[:, None, :]).squeeze(1)
            loss = torch.mean((pred[hole] - row[hole]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "mse": mean_loss})
        print(f"epoch {epoch:02d} mse={mean_loss:.6f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": 32,
            "depth": 4,
            "kernel": 5,
            "history": history,
            "elapsed_s": time.perf_counter() - t0,
            "device": device,
            "n_crops": len(data),
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
