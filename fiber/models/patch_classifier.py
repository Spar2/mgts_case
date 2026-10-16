"""Small 2D CNN that names a planted crop. Offline patch, scorecard on CPU hosts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


TIME_CROP = 28
SPACE_CROP = 40

if nn is not None:
    class PatchClassifier(nn.Module):
        def __init__(self, n_class: int = 6):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.GELU(),
            )
            self.head = nn.Linear(64, n_class)

        def forward(self, x):
            hidden = self.net(x)
            pooled = torch.cat([hidden.mean(dim=(2, 3)), hidden.amax(dim=(2, 3))], dim=1)
            return self.head(pooled)
else:  # pragma: no cover
    PatchClassifier = None  # type: ignore[misc, assignment]


def crop_deviation(field: np.ndarray, valid: np.ndarray, t0: int, x0: int) -> np.ndarray:
    """Crop minus its own valid median, in degrees. Invalid cells are 0."""
    patch = np.asarray(field[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP], dtype=np.float64)
    mask = np.asarray(valid[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP], dtype=bool)
    center = float(np.median(patch[mask])) if mask.any() else 0.0
    out = np.zeros_like(patch, dtype=np.float32)
    out[mask] = (patch[mask] - center).astype(np.float32)
    return out


def load_patch_classifier(path: Path, device: str | None = None):
    if torch is None:
        raise RuntimeError("torch is required to load the patch classifier")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(path, map_location=device, weights_only=True)
    model = PatchClassifier(n_class=int(blob.get("n_class", 6)))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob, device
