"""2D conv autoencoder: reconstruct the halo background of median_x3.

The error |median_x3 - reconstruction| is a learned local-anomaly score.
Invalid cells are not trained and stay at score 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:
    class ConvBackgroundAE(nn.Module):
        """8×64 patch. Bottleneck is 1×8 so a narrow hotspot cannot be copied."""

        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 8, 3, stride=2, padding=1),
                nn.GELU(),
            )
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(8, 32, 4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            )

        def forward(self, x):
            return self.dec(self.enc(x))
else:  # pragma: no cover
    ConvBackgroundAE = None  # type: ignore[misc, assignment]


def load_ae(path: Path, device: str | None = None):
    if torch is None:
        raise RuntimeError("torch is required to load the autoencoder")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(path, map_location=device, weights_only=True)
    model = ConvBackgroundAE()
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob, device


def reconstruction_z(
    clean: np.ndarray,
    valid: np.ndarray,
    weights: Path,
    *,
    time_crop: int = 8,
    space_crop: int = 64,
    batch: int = 64,
) -> np.ndarray:
    """Overlap-add |clean - AE(clean)| in °C, then divide by per-profile MAD."""
    if torch is None:
        raise RuntimeError("torch is required for the autoencoder")
    model, blob, device = load_ae(Path(weights))
    mean = float(blob["mean_c"])
    std = float(blob["std_c"])
    field = np.asarray(clean, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    t_count, x_count = field.shape
    err_sum = np.zeros(field.shape, dtype=np.float64)
    err_n = np.zeros(field.shape, dtype=np.float64)
    if t_count < time_crop or x_count < space_crop:
        return np.zeros(field.shape, dtype=np.float32)

    t_starts = list(range(0, t_count - time_crop + 1, time_crop // 2))
    x_starts = list(range(0, x_count - space_crop + 1, space_crop // 2))
    if t_starts[-1] != t_count - time_crop:
        t_starts.append(t_count - time_crop)
    if x_starts[-1] != x_count - space_crop:
        x_starts.append(x_count - space_crop)

    jobs = [(t0, x0) for t0 in t_starts for x0 in x_starts]
    with torch.no_grad():
        for start in range(0, len(jobs), batch):
            chunk = jobs[start:start + batch]
            batch_in = []
            for t0, x0 in chunk:
                patch = field[t0:t0 + time_crop, x0:x0 + space_crop].copy()
                patch[~np.isfinite(patch)] = mean
                batch_in.append((patch - mean) / std)
            batch_in = np.stack(batch_in)[:, None]
            pred = model(torch.from_numpy(batch_in).to(device)).cpu().numpy()[:, 0]
            pred = pred * std + mean
            for (t0, x0), rec in zip(chunk, pred, strict=True):
                sl = (slice(t0, t0 + time_crop), slice(x0, x0 + space_crop))
                err_sum[sl] += np.abs(field[sl] - rec)
                err_n[sl] += 1.0
    err = np.divide(err_sum, np.maximum(err_n, 1.0))
    err[~mask] = np.nan
    with np.errstate(all="ignore"):
        scale = 1.4826 * np.nanmedian(err, axis=1)
    scale = np.where(np.isfinite(scale), np.maximum(scale, 0.30), 0.30)
    z = np.zeros(field.shape, dtype=np.float32)
    ok = mask & np.isfinite(err)
    denom = np.broadcast_to(scale[:, None], field.shape)
    z[ok] = (err[ok] / denom[ok]).astype(np.float32)
    return z
