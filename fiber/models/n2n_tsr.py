"""Neighbor2Neighbor on a time-by-distance field, plus a smoothness term.

Neighbor pairs follow Huang et al., CVPR 2021
(https://github.com/TaoHuang2018/Neighbor2Neighbor).
The extra penalty on time and distance differences is the N2N-TSR idea.
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
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:
    class NeighborNet(nn.Module):
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
    NeighborNet = None  # type: ignore[misc, assignment]


_PAIRS = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (1, 0),
    (2, 0),
    (3, 1),
    (3, 2),
)


def _neighbor_pair(image, live, generator: torch.Generator):
    """Split each 2x2 block into a pixel and one edge-adjacent neighbor."""
    _n, _c, height, width = image.shape
    pad_h = height % 2
    pad_w = width % 2
    work = image
    work_live = live
    if pad_h or pad_w:
        work = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
        work_live = torch.nn.functional.pad(live, (0, pad_w, 0, pad_h), mode="replicate")
    _n, _c, height, width = work.shape
    flat = work.reshape(1, 1, height // 2, 2, width // 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(1, 1, height // 2, width // 2, 4)
    flat_live = work_live.reshape(1, 1, height // 2, 2, width // 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(1, 1, height // 2, width // 2, 4)
    pairs = torch.tensor(_PAIRS, dtype=torch.long)
    pick = torch.randint(0, len(_PAIRS), (height // 2, width // 2), generator=generator)
    chosen = pairs[pick].to(flat.device)
    left = chosen[..., 0].view(1, 1, height // 2, width // 2, 1)
    right = chosen[..., 1].view(1, 1, height // 2, width // 2, 1)
    first = torch.gather(flat, -1, left).squeeze(-1)
    second = torch.gather(flat, -1, right).squeeze(-1)
    first_live = torch.gather(flat_live, -1, left).squeeze(-1)
    second_live = torch.gather(flat_live, -1, right).squeeze(-1)
    return first, second, first_live, second_live


def _smooth(pred, live) -> torch.Tensor:
    time_ok = live[:, :, 1:, :] * live[:, :, :-1, :]
    space_ok = live[:, :, :, 1:] * live[:, :, :, :-1]
    time_jump = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs() * time_ok
    space_jump = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs() * space_ok
    time_loss = time_jump.sum() / time_ok.sum().clamp(min=1.0)
    space_loss = space_jump.sum() / space_ok.sum().clamp(min=1.0)
    return time_loss + space_loss


def _masked_mse(pred, target, live) -> torch.Tensor:
    weight = (live > 0.5).float()
    err = (pred - target) ** 2 * weight
    return err.sum() / weight.sum().clamp(min=1.0)


def _fit(filled: np.ndarray, mask: np.ndarray, steps: int, seed: int, device: str, smooth: float):
    seed_torch(seed)
    model = NeighborNet().to(device)
    if steps <= 0 or min(filled.shape) < 2:
        model.eval()
        return model
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    image = torch.from_numpy(filled[None, None]).to(device)
    live = torch.from_numpy(mask.astype(np.float32)[None, None]).to(device)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        first, second, first_live, second_live = _neighbor_pair(image, live, generator)
        loss = _masked_mse(model(first), second, second_live)
        loss = loss + _masked_mse(model(second), first, first_live)
        loss = loss + smooth * _smooth(model(image), live)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _predict(model, filled: np.ndarray, device: str) -> np.ndarray:
    image = torch.from_numpy(filled[None, None]).to(device)
    with torch.no_grad():
        return model(image)[0, 0].detach().cpu().numpy().astype(np.float32)


def denoise_n2n_tsr(
    raw_c: np.ndarray,
    valid: np.ndarray,
    *,
    seed: int = 0,
    steps: int = 8,
    device: str | None = None,
    weights: Path | None = None,
    smooth: float = 0.01,
) -> np.ndarray:
    chosen = torch_device(device)
    filled, mask = filled_field(raw_c, valid)
    center, spread = field_scale(filled, mask)
    normal = ((filled - center) / spread).astype(np.float32)
    if weights is not None and Path(weights).exists():
        seed_torch(seed)
        blob = load_state(Path(weights), chosen)
        model = NeighborNet(hidden=int(blob.get("hidden", 16))).to(chosen)
        model.load_state_dict(blob["state_dict"])
        model.eval()
    else:
        model = _fit(normal, mask, steps=steps, seed=seed, device=chosen, smooth=smooth)
    restored = _predict(model, normal, chosen) * np.float32(spread) + np.float32(center)
    return restore_dead(restored, live_mask(raw_c, valid))


def train_n2n_tsr(
    raw_c: np.ndarray,
    valid: np.ndarray,
    dest: Path,
    *,
    seed: int = 0,
    steps: int = 40,
    device: str | None = None,
    smooth: float = 0.01,
) -> Path:
    chosen = torch_device(device)
    filled, mask = filled_field(raw_c, valid)
    center, spread = field_scale(filled, mask)
    normal = ((filled - center) / spread).astype(np.float32)
    model = _fit(normal, mask, steps=steps, seed=seed, device=chosen, smooth=smooth)
    dest = Path(dest)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_state(dest, {"state_dict": state, "hidden": 16, "seed": seed, "steps": steps, "smooth": smooth})
    return dest
