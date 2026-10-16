"""Small 2D U-Net denoiser trained as Noise2Void with an absolute head.

A residual head on zero-filled holes learned to add the signal back. On
5.csv that doubled the warm-core contrast (retention 2.12) and did not
reduce high-frequency MAD. The head now predicts temperature itself.
Inference hides each pixel behind its left neighbour on a 2x2 checkerboard,
so the centre value cannot be copied.

The crop is an offline patch: later rows sit in the same tensor as earlier
ones. The scorecard is the deliverable for machines without a GPU.
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
    class _Block(nn.Module):
        def __init__(self, c_in: int, c_out: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)


    class UNet2DDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = _Block(1, 16)
            self.enc2 = _Block(16, 32)
            self.pool = nn.MaxPool2d(2)
            self.bot = _Block(32, 64)
            self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec2 = _Block(64, 32)
            self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dec1 = _Block(32, 16)
            self.head = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            d2 = self.dec2(torch.cat([self.up2(self.bot(self.pool(e2))), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)
else:  # pragma: no cover
    UNet2DDenoiser = None  # type: ignore[misc, assignment]


def load_unet(path: Path, device: str | None = None):
    if torch is None:
        raise RuntimeError("torch is required to load the 2D U-Net")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(path, map_location=device, weights_only=True)
    model = UNet2DDenoiser()
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob, device


def _checkerboard_inputs(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hide one phase of a 2x2 grid behind the left neighbour.

    Returns inputs of shape (4, T, X) and a boolean keep-mask of the same shape.
    """
    inputs = np.empty((4, *patch.shape), dtype=np.float32)
    keep = np.zeros((4, *patch.shape), dtype=bool)
    donor = np.empty_like(patch)
    donor[:, 1:] = patch[:, :-1]
    donor[:, 0] = patch[:, 0]
    phase = 0
    for time_phase in (0, 1):
        for space_phase in (0, 1):
            hole = np.zeros(patch.shape, dtype=bool)
            hole[time_phase::2, space_phase::2] = True
            filled = patch.copy()
            filled[hole] = donor[hole]
            inputs[phase] = filled
            keep[phase] = hole
            phase += 1
    return inputs, keep


def apply_unet2d(
    raw_c: np.ndarray,
    valid: np.ndarray,
    weights: Path,
    *,
    time_crop: int = 32,
    space_crop: int = 128,
    batch: int = 16,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required for unet2d_n2self")
    model, blob, device = load_unet(Path(weights))
    mean = float(blob["mean_c"])
    std = float(blob["std_c"])
    field = np.asarray(raw_c, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    t_count, x_count = field.shape
    acc = np.zeros(field.shape, dtype=np.float64)
    count = np.zeros(field.shape, dtype=np.float64)
    if t_count < time_crop or x_count < space_crop:
        out = field.copy()
        out[~mask] = np.nan
        return out
    t_starts = list(range(0, t_count - time_crop + 1, time_crop // 2))
    x_starts = list(range(0, x_count - space_crop + 1, space_crop // 2))
    if t_starts[-1] != t_count - time_crop:
        t_starts.append(t_count - time_crop)
    if x_starts[-1] != x_count - space_crop:
        x_starts.append(x_count - space_crop)
    jobs = [(t0, x0) for t0 in t_starts for x0 in x_starts]
    with torch.inference_mode():
        for start in range(0, len(jobs), batch):
            chunk = jobs[start:start + batch]
            packed = []
            keeps = []
            for t0, x0 in chunk:
                patch = field[t0:t0 + time_crop, x0:x0 + space_crop].copy()
                local = mask[t0:t0 + time_crop, x0:x0 + space_crop]
                patch[~local | ~np.isfinite(patch)] = mean
                phases, keep = _checkerboard_inputs(((patch - mean) / std).astype(np.float32))
                packed.append(phases)
                keeps.append(keep)
            stacked = np.concatenate(packed, axis=0)
            pred = model(torch.from_numpy(stacked[:, None]).to(device))
            pred = pred.squeeze(1).detach().cpu().numpy() * std + mean
            pred = pred.reshape(len(chunk), 4, time_crop, space_crop)
            for index, (t0, x0) in enumerate(chunk):
                rec = np.zeros((time_crop, space_crop), dtype=np.float64)
                for phase in range(4):
                    hole = keeps[index][phase]
                    rec[hole] = pred[index, phase][hole]
                sl = (slice(t0, t0 + time_crop), slice(x0, x0 + space_crop))
                acc[sl] += rec
                count[sl] += 1.0
    out = np.divide(acc, np.maximum(count, 1.0)).astype(np.float32)
    out[~mask] = np.nan
    return out
