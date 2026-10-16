"""Two self-attention layers over cable columns.

Column order is a fixed sinusoid, not a learned table. Dropout is off so a
saved checkpoint rescores the same windows.
"""

from __future__ import annotations

import math
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _sinusoid(n_columns: int, width: int, device, dtype):
    position = torch.arange(n_columns, device=device, dtype=dtype)[:, None]
    div = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10000.0) / width))
    encoding = torch.zeros(n_columns, width, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * div)
    encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].shape[1]])
    return encoding


if nn is not None:
    class ColumnTransformer(nn.Module):
        def __init__(self, n_class: int = 6, time_steps: int = 28, d_model: int = 32, nhead: int = 4, n_layers: int = 2):
            super().__init__()
            if n_layers < 2:
                raise ValueError("the column transformer needs at least two attention layers")
            self.proj = nn.Linear(time_steps, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=64,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
            self.head = nn.Linear(d_model, n_class)

        def forward(self, x):
            if x.dim() == 4:
                x = x[:, 0]
            tokens = self.proj(x.transpose(1, 2))
            tokens = tokens + _sinusoid(tokens.shape[1], tokens.shape[2], tokens.device, tokens.dtype)
            hidden = self.encoder(tokens)
            return self.head(hidden.mean(dim=1))
else:  # pragma: no cover
    ColumnTransformer = None  # type: ignore[misc, assignment]


def load_column_transformer(path: Path, device: str | None = None):
    if torch is None:
        raise RuntimeError("torch is required to load the column transformer")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(path, map_location=device, weights_only=True)
    model = ColumnTransformer(n_class=int(blob.get("n_class", 6)))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob, device
