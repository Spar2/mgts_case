"""Train the 2D U-Net with Noise2Self on raw DTS crops (local GPU)."""

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
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "weights" / "unet2d_n2self.pt")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crops", type=int, default=4000)
    parser.add_argument("--mask-frac", type=float, default=0.15)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from fiber.io import Record
    from fiber.models.unet2d_denoise import UNet2DDenoiser

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((args.records / "index.json").read_text(encoding="utf-8"))
    t_crop, x_crop = 32, 128
    rng = np.random.default_rng(0)
    patches = []
    masks = []
    sample_vals = []
    per_file = max(1, args.crops // len(index))
    for item in index:
        record = Record.load(args.records / item["name"])
        field = record.raw_c
        t_count, x_count = field.shape
        if t_count < t_crop or x_count < x_crop:
            continue
        sample_vals.append(field[record.valid][::80])
        got = 0
        tries = 0
        while got < per_file and tries < per_file * 30:
            tries += 1
            t0 = int(rng.integers(0, t_count - t_crop + 1))
            x0 = int(rng.integers(0, x_count - x_crop + 1))
            sl = (slice(t0, t0 + t_crop), slice(x0, x0 + x_crop))
            ok = record.valid[sl] & np.isfinite(field[sl])
            if ok.mean() < 0.9:
                continue
            patches.append(np.where(ok, field[sl], 0.0).astype(np.float32))
            masks.append(ok)
            got += 1
        print(f"crops {item['name']}: {got}")
    values = np.concatenate(sample_vals)
    mean_c = float(values.mean())
    std_c = float(values.std()) if float(values.std()) > 1e-3 else 1.0
    x = (np.stack(patches) - mean_c) / std_c
    m = np.stack(masks)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(m)),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
    )
    model = UNet2DDenoiser().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"device={device} crops={len(patches)} mean={mean_c:.2f} std={std_c:.2f}")
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, mb in loader:
            xb = xb.to(device)[:, None]
            mb = mb.to(device)[:, None]
            hole = (torch.rand_like(xb) < args.mask_frac) & mb
            if int(hole.sum()) < 8:
                continue
            donor = torch.roll(xb, shifts=1, dims=-1)
            donor[:, :, :, 0] = xb[:, :, :, 0]
            inp = torch.where(hole, donor, xb)
            pred = model(inp)
            loss = ((pred - xb) ** 2 * hole).sum() / hole.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mse = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "mse": mse})
        print(f"epoch {epoch:02d} mse={mse:.6f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean_c": mean_c,
            "std_c": std_c,
            "time_crop": t_crop,
            "space_crop": x_crop,
            "history": history,
            "elapsed_s": time.perf_counter() - t0,
            "device": device,
            "n_crops": len(patches),
            "serve": "scorecard",
            "objective": "noise2void_absolute_checkerboard",
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
