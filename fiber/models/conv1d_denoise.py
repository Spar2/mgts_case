"""Residual 1D CNN denoiser trained with Noise2Self (masked pixels)."""

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
    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, kernel: int = 5):
            super().__init__()
            pad = kernel // 2
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel, padding=pad),
                nn.GELU(),
                nn.Conv1d(channels, channels, kernel, padding=pad),
            )
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(x + self.net(x))


    class Conv1dDenoiser(nn.Module):
        def __init__(self, hidden: int = 32, depth: int = 4, kernel: int = 5):
            super().__init__()
            pad = kernel // 2
            self.stem = nn.Conv1d(1, hidden, kernel, padding=pad)
            self.blocks = nn.Sequential(*[ResidualBlock(hidden, kernel) for _ in range(depth)])
            self.head = nn.Conv1d(hidden, 1, kernel, padding=pad)

        def forward(self, x):
            y = self.head(self.blocks(self.stem(x)))
            return x + y
else:  # pragma: no cover
    Conv1dDenoiser = None  # type: ignore[misc, assignment]


def load_weights(path: Path, device: str | None = None):
    if torch is None:
        raise RuntimeError("torch is required to load conv1d weights")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(path, map_location=device, weights_only=True)
    model = Conv1dDenoiser(
        hidden=int(blob.get("hidden", 32)),
        depth=int(blob.get("depth", 4)),
        kernel=int(blob.get("kernel", 5)),
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, device


def apply_conv1d(
    raw_c: np.ndarray,
    valid: np.ndarray,
    weights: Path,
    batch: int = 32,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required for conv1d_n2self")
    model, device = load_weights(Path(weights))
    field = np.where(valid, raw_c, 0.0).astype(np.float32)
    mean = np.nanmean(np.where(valid, raw_c, np.nan))
    std = np.nanstd(np.where(valid, raw_c, np.nan))
    std = float(std) if np.isfinite(std) and std > 1e-3 else 1.0
    mean = float(mean) if np.isfinite(mean) else 0.0
    normed = (field - mean) / std
    out = np.full(raw_c.shape, np.nan, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, raw_c.shape[0], batch):
            sl = slice(start, start + batch)
            tensor = torch.from_numpy(normed[sl][:, None, :]).to(device)
            pred = model(tensor).squeeze(1).cpu().numpy() * std + mean
            out[sl] = pred
    out[~valid] = np.nan
    return out
