"""Residual CNN for a context crop. Scorecard only.

The published patch net is three convolutions and a 64-wide head. This stack
is a stem plus two residual blocks, so the weights are not that checkpoint.
"""

from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:
    class _ResidualBlock(nn.Module):
        def __init__(self, cin: int, cout: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
            if stride != 1 or cin != cout:
                self.skip = nn.Conv2d(cin, cout, 1, stride=stride)
            else:
                self.skip = nn.Identity()

        def forward(self, x):
            hidden = nn.functional.gelu(self.conv1(x))
            hidden = self.conv2(hidden)
            return nn.functional.gelu(hidden + self.skip(x))

    class DepthCNN(nn.Module):
        def __init__(self, n_class: int = 6):
            super().__init__()
            self.stem = nn.Conv2d(1, 16, 3, padding=1)
            self.block1 = _ResidualBlock(16, 16)
            self.block2 = _ResidualBlock(16, 32, stride=2)
            self.head = nn.Linear(32, n_class)

        def forward(self, x):
            hidden = nn.functional.gelu(self.stem(x))
            hidden = self.block1(hidden)
            hidden = self.block2(hidden)
            return self.head(hidden.mean(dim=(2, 3)))
else:  # pragma: no cover
    DepthCNN = None  # type: ignore[misc, assignment]


def load_depth_cnn(path: Path, device: str | None = None):
    if torch is None:
        raise RuntimeError("torch is required to load the depth CNN")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(path, map_location=device, weights_only=True)
    model = DepthCNN(n_class=int(blob.get("n_class", 6)))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob, device
