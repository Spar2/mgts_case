"""Zero-Shot Noise2Noise on one temperature field.

Mansour and Heckel, CVPR 2023, arXiv:2303.11253.
Two fixed 2x2 downsamplers supervise a tiny residual network.
Dead cells and the -300 sentinel stay out of the loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fiber.models.ssl_common import (
    field_scale,
    filled_field,
    live_mask,
    load_state,
    restore_dead,
    save_state,
    seed_torch,
    torch_device,
)

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


if nn is not None:
    class ZeroShotNet(nn.Module):
        def __init__(self, hidden: int = 16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, 3, padding=1),
            )

        def forward(self, image):
            return image + self.net(image)
else:  # pragma: no cover
    ZeroShotNet = None  # type: ignore[misc, assignment]


def _pair_downsampler(image):
    channels = image.shape[1]
    first = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], dtype=image.dtype, device=image.device)
    second = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], dtype=image.dtype, device=image.device)
    first = first.repeat(channels, 1, 1, 1)
    second = second.repeat(channels, 1, 1, 1)
    return (
        F.conv2d(image, first, stride=2, groups=channels),
        F.conv2d(image, second, stride=2, groups=channels),
    )


def _fit(filled: np.ndarray, mask: np.ndarray, steps: int, seed: int, device: str):
    seed_torch(seed)
    model = ZeroShotNet().to(device)
    if steps <= 0 or filled.shape[0] < 2 or filled.shape[1] < 2:
        model.eval()
        return model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    image = torch.from_numpy(filled[None, None]).to(device)
    live = torch.from_numpy(mask.astype(np.float32)[None, None]).to(device)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        down_a, down_b = _pair_downsampler(image)
        mask_a, mask_b = _pair_downsampler(live)
        pred_a = model(down_a)
        pred_b = model(down_b)
        rec = ((pred_a - down_b) ** 2 * (mask_b > 0).float()).sum() / (mask_b > 0).float().sum().clamp(min=1.0)
        rec = rec + ((pred_b - down_a) ** 2 * (mask_a > 0).float()).sum() / (mask_a > 0).float().sum().clamp(min=1.0)
        full = model(image)
        den_a, den_b = _pair_downsampler(full)
        cons = ((pred_a - den_a) ** 2 * (mask_a > 0).float()).sum() / (mask_a > 0).float().sum().clamp(min=1.0)
        cons = cons + ((pred_b - den_b) ** 2 * (mask_b > 0).float()).sum() / (mask_b > 0).float().sum().clamp(min=1.0)
        (rec + cons).backward()
        optimizer.step()
    model.eval()
    return model


def _predict(model, filled: np.ndarray, device: str) -> np.ndarray:
    image = torch.from_numpy(filled[None, None]).to(device)
    with torch.no_grad():
        pred = model(image)[0, 0].detach().cpu().numpy()
    return pred.astype(np.float32)


def denoise_zsn2n(
    raw_c: np.ndarray,
    valid: np.ndarray,
    *,
    seed: int = 0,
    steps: int = 8,
    device: str | None = None,
    weights: Path | None = None,
) -> np.ndarray:
    chosen = torch_device(device)
    filled, mask = filled_field(raw_c, valid)
    center, spread = field_scale(filled, mask)
    normal = ((filled - center) / spread).astype(np.float32)
    if weights is not None and Path(weights).exists():
        seed_torch(seed)
        blob = load_state(Path(weights), chosen)
        model = ZeroShotNet(hidden=int(blob.get("hidden", 16))).to(chosen)
        model.load_state_dict(blob["state_dict"])
        model.eval()
    else:
        model = _fit(normal, mask, steps=steps, seed=seed, device=chosen)
    restored = _predict(model, normal, chosen) * np.float32(spread) + np.float32(center)
    return restore_dead(restored, live_mask(raw_c, valid))


def train_zsn2n(raw_c: np.ndarray, valid: np.ndarray, dest: Path, *, seed: int = 0, steps: int = 40, device: str | None = None) -> Path:
    chosen = torch_device(device)
    filled, mask = filled_field(raw_c, valid)
    center, spread = field_scale(filled, mask)
    normal = ((filled - center) / spread).astype(np.float32)
    model = _fit(normal, mask, steps=steps, seed=seed, device=chosen)
    dest = Path(dest)
    save_state(dest, {"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "hidden": 16, "seed": seed, "steps": steps})
    return dest
