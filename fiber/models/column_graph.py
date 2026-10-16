"""Three rounds of messages between neighbouring cable columns.

Each column of the crop is a node. A node reads itself and the mean of the
columns on its left and right. The head sees the mean and the max of the
column states, so a short event is not averaged away. PyTorch Geometric is
not required.
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
    class ColumnGraph(nn.Module):
        def __init__(self, n_class: int = 6, time_steps: int = 28, hidden: int = 32):
            super().__init__()
            self.lift = nn.Linear(time_steps, hidden)
            self.msg1 = nn.Linear(hidden * 2, hidden)
            self.msg2 = nn.Linear(hidden * 2, hidden)
            self.msg3 = nn.Linear(hidden * 2, hidden)
            self.head = nn.Linear(hidden * 2, n_class)

        def _messages(self, hidden, layer):
            left = torch.zeros_like(hidden)
            right = torch.zeros_like(hidden)
            left[:, 1:] = hidden[:, :-1]
            right[:, :-1] = hidden[:, 1:]
            neighbour = 0.5 * (left + right)
            return nn.functional.gelu(layer(torch.cat((hidden, neighbour), dim=-1)))

        def forward(self, x):
            if x.dim() == 4:
                x = x[:, 0]
            hidden = nn.functional.gelu(self.lift(x.transpose(1, 2)))
            hidden = hidden + self._messages(hidden, self.msg1)
            hidden = hidden + self._messages(hidden, self.msg2)
            hidden = hidden + self._messages(hidden, self.msg3)
            pooled = torch.cat((hidden.mean(dim=1), hidden.amax(dim=1)), dim=-1)
            return self.head(pooled)
else:  # pragma: no cover
    ColumnGraph = None  # type: ignore[misc, assignment]


def load_column_graph(path: Path, device: str | None = None):
    if torch is None:
        raise RuntimeError("torch is required to load the column graph")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(path, map_location=device, weights_only=True)
    model = ColumnGraph(n_class=int(blob.get("n_class", 6)))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob, device
