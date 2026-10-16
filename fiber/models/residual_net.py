"""Small CNN on a 28 by 64 metre residual. Scorecard only."""

from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:
    class ResidualNet(nn.Module):
        def __init__(self, n_class: int = 6):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1),
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
    ResidualNet = None  # type: ignore[misc, assignment]


def load_residual_net(path: Path, device: str | None = None):
    if torch is None:
        raise RuntimeError("torch is required to load the residual net")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(path, map_location=device, weights_only=True)
    model = ResidualNet(n_class=int(blob.get("n_class", 6)))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob, device
