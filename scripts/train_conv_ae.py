"""Train the background autoencoder on median_x3 (local GPU)."""

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
    parser.add_argument("--denoise", type=Path, default=ROOT / "artifacts" / "denoise")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "weights" / "conv_ae_bg.pt")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--crops", type=int, default=8000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from fiber.anomaly import spatial_background
    from fiber.io import Record
    from fiber.models.conv_ae import ConvBackgroundAE

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((args.records / "index.json").read_text(encoding="utf-8"))
    inputs = []
    targets = []
    valids = []
    rng = np.random.default_rng(0)
    t_crop, x_crop = 8, 64
    per_file = max(1, args.crops // len(index))
    sample_vals = []
    for item in index:
        record = Record.load(args.records / item["name"])
        clean = np.load(args.denoise / item["name"] / "median_x3.npy")
        bg = spatial_background(clean, record.valid, record.x_m)
        t_count, x_count = clean.shape
        if t_count < t_crop or x_count < x_crop:
            continue
        sample_vals.append(clean[record.valid][::50])
        got = 0
        tries = 0
        while got < per_file and tries < per_file * 20:
            tries += 1
            t0 = int(rng.integers(0, t_count - t_crop + 1))
            x0 = int(rng.integers(0, x_count - x_crop + 1))
            sl = (slice(t0, t0 + t_crop), slice(x0, x0 + x_crop))
            ok = record.valid[sl] & np.isfinite(clean[sl]) & np.isfinite(bg[sl])
            if ok.mean() < 0.85:
                continue
            inputs.append(np.where(ok, clean[sl], 0.0).astype(np.float32))
            targets.append(np.where(ok, bg[sl], 0.0).astype(np.float32))
            valids.append(ok)
            got += 1
        print(f"crops {item['name']}: {got}")
    if len(inputs) < args.batch:
        raise RuntimeError("not enough crops")
    values = np.concatenate(sample_vals)
    mean_c = float(values.mean())
    std_c = float(values.std())
    if std_c < 1e-3:
        std_c = 1.0
    x = (np.stack(inputs) - mean_c) / std_c
    y = (np.stack(targets) - mean_c) / std_c
    m = np.stack(valids).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
    )
    model = ConvBackgroundAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"device={device} crops={len(inputs)} mean={mean_c:.2f} std={std_c:.2f}")
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb, mb in loader:
            xb = xb.to(device)[:, None]
            yb = yb.to(device)[:, None]
            mb = mb.to(device)[:, None]
            pred = model(xb)
            denom = mb.sum().clamp_min(1.0)
            loss = ((pred - yb) ** 2 * mb).sum() / denom
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mse = float(np.mean(losses))
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
            "n_crops": len(inputs),
            "target": "halo_background_inner8_outer40_bin2",
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
