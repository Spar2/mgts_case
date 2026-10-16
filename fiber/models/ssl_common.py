"""Shared pieces for the research self-supervised denoisers.

Torch stays optional. Invalid cells and the -300 sentinel never enter the loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fiber.quality import SENTINEL_MAX

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def torch_device(requested: str | None) -> str:
    if torch is None:
        raise RuntimeError("torch is required for this research denoiser")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda was requested but is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def live_mask(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cells that are allowed to supervise a denoiser."""
    values = np.asarray(field)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    mask &= values > SENTINEL_MAX
    return mask


def field_scale(filled: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Per-field location and scale. A constant cable must not become a division by zero."""
    if not np.any(mask):
        return 0.0, 1.0
    values = np.asarray(filled, dtype=np.float64)[mask]
    center = float(np.mean(values))
    spread = float(np.std(values))
    if not np.isfinite(spread) or spread < 1e-3:
        spread = 1.0
    if not np.isfinite(center):
        center = 0.0
    return center, spread


def filled_field(field: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace dead cells with the valid median so a convolution never sees -300."""
    mask = live_mask(field, valid)
    values = np.asarray(field, dtype=np.float32)
    if not np.any(mask):
        return np.zeros(values.shape, dtype=np.float32), mask
    fill = float(np.median(values[mask]))
    out = np.array(values, dtype=np.float32, copy=True)
    out[~mask] = fill
    return out, mask


def restore_dead(pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(pred, dtype=np.float32).copy()
    out[~np.asarray(valid, dtype=bool)] = np.nan
    return out


def seed_torch(seed: int) -> None:
    if torch is None:
        raise RuntimeError("torch is required for this research denoiser")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_state(path: Path, device: str) -> dict:
    if torch is None:
        raise RuntimeError("torch is required to load weights")
    return torch.load(Path(path), map_location=device, weights_only=True)


def save_state(path: Path, payload: dict) -> None:
    if torch is None:
        raise RuntimeError("torch is required to save weights")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
