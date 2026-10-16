"""Predict one distance column from the five columns on each side.

The blind spot is the jDAS idea (van den Ende et al.,
https://github.com/martijnende/jDAS): the center metre is hidden, and the
network sees only ±5 neighbours. Training samples come from MGTS records.
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

OFFSETS = np.asarray([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], dtype=np.int64)


if nn is not None:
    class BlindSpotNet(nn.Module):
        def __init__(self, hidden: int = 16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(len(OFFSETS), hidden, 5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, 5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden, 1, 1),
            )

        def forward(self, neighbors):
            return self.net(neighbors).squeeze(1)
else:  # pragma: no cover
    BlindSpotNet = None  # type: ignore[misc, assignment]


def neighbor_batch(field: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return neighbors (X, 10, T), target (X, T), and a loss mask (X, T)."""
    filled, mask = filled_field(field, valid)
    _time, distance = filled.shape
    centers = np.arange(distance)
    index = centers[None, :] + OFFSETS[:, None]
    inside = (index >= 0) & (index < distance)
    clipped = np.clip(index, 0, distance - 1)
    gathered = filled[:, clipped.reshape(-1)].reshape(_time, len(OFFSETS), distance)
    gathered[:, ~inside] = 0.0
    neighbor_live = mask[:, clipped.reshape(-1)].reshape(_time, len(OFFSETS), distance)
    neighbor_live[:, ~inside] = False
    loss = mask.T.copy()
    loss &= neighbor_live.all(axis=1).T
    neighbors = np.transpose(gathered, (2, 1, 0)).astype(np.float32)
    target = filled.T.astype(np.float32)
    return neighbors, target, loss


def _fit_arrays(neighbors: np.ndarray, target: np.ndarray, loss: np.ndarray, steps: int, seed: int, device: str):
    seed_torch(seed)
    model = BlindSpotNet().to(device)
    usable = loss.any(axis=1)
    if steps <= 0 or not np.any(usable):
        model.eval()
        return model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.from_numpy(neighbors[usable]).to(device)
    truth = torch.from_numpy(target[usable]).to(device)
    weight = torch.from_numpy(loss[usable].astype(np.float32)).to(device)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(inputs)
        err = (pred - truth) ** 2 * weight
        (err.sum() / weight.sum().clamp(min=1.0)).backward()
        optimizer.step()
    model.eval()
    return model


def _predict(model, neighbors: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        pred = model(torch.from_numpy(neighbors).to(device)).detach().cpu().numpy()
    return pred.astype(np.float32).T


def denoise_blindspot_x(
    raw_c: np.ndarray,
    valid: np.ndarray,
    *,
    seed: int = 0,
    steps: int = 0,
    device: str | None = None,
    weights: Path | None = None,
) -> np.ndarray:
    """Apply a blind-spot network. `steps` fits this field only when no weights are given.

    The training script is what learns from MGTS. A missing weight file uses a
    seeded forward pass (steps=0) so a cable without MGTS still returns a field.
    """
    chosen = torch_device(device)
    neighbors, target, loss = neighbor_batch(raw_c, valid)
    center, spread = field_scale(target, loss)
    neighbors = ((neighbors - center) / spread).astype(np.float32)
    target = ((target - center) / spread).astype(np.float32)
    if weights is not None and Path(weights).exists():
        seed_torch(seed)
        blob = load_state(Path(weights), chosen)
        model = BlindSpotNet(hidden=int(blob.get("hidden", 16))).to(chosen)
        model.load_state_dict(blob["state_dict"])
        model.eval()
    else:
        model = _fit_arrays(neighbors, target, loss, steps=steps, seed=seed, device=chosen)
    pred = _predict(model, neighbors, chosen) * np.float32(spread) + np.float32(center)
    return restore_dead(pred, live_mask(raw_c, valid))


def fit_blindspot_batch(
    neighbors: np.ndarray,
    target: np.ndarray,
    loss: np.ndarray,
    dest: Path,
    *,
    seed: int = 0,
    steps: int = 40,
    device: str | None = None,
) -> Path:
    chosen = torch_device(device)
    center, spread = field_scale(target, loss)
    neighbors = ((neighbors - np.float32(center)) / np.float32(spread)).astype(np.float32)
    target = ((target - np.float32(center)) / np.float32(spread)).astype(np.float32)
    model = _fit_arrays(neighbors, target, loss, steps=steps, seed=seed, device=chosen)
    dest = Path(dest)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_state(dest, {"state_dict": state, "hidden": 16, "seed": seed, "steps": steps, "source": "MGTS_data"})
    return dest
