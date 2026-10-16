"""Small nets for the dts_synth_v1 summary pipeline.

These are experimental shapes. They are not the operational cleaner.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class DilatedDenoiser(nn.Module):
        """1D stack with dilations 1, 2, 4 and 8. Output is the sum of the inputs plus a correction."""

        def __init__(self, in_channels: int = 1, hidden: int = 32):
            super().__init__()
            self.stem = nn.Conv1d(in_channels, hidden, 3, padding=1)
            self.blocks = nn.ModuleList(
                [nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation) for dilation in (1, 2, 4, 8)]
            )
            self.act = nn.GELU()
            self.head = nn.Conv1d(hidden, 1, 1)

        def forward(self, x):
            hidden = self.act(self.stem(x))
            for block in self.blocks:
                hidden = self.act(hidden + block(hidden))
            return x.sum(dim=1, keepdim=True) + self.head(hidden)


    class _Stream(nn.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.GELU(),
            )

        def forward(self, x):
            hidden = self.net(x)
            return torch.cat([hidden.mean(dim=(2, 3)), hidden.amax(dim=(2, 3))], dim=1)


    class GradientFusion(nn.Module):
        """One stream on the column residual, one stream on the time and space derivatives."""

        def __init__(self, n_class: int = 6):
            super().__init__()
            self.raw = _Stream(1)
            self.grad = _Stream(2)
            self.head = nn.Linear(128, n_class)

        def forward(self, x):
            pooled = torch.cat([self.raw(x[:, :1]), self.grad(x[:, 1:])], dim=1)
            return self.head(pooled)


    class _Block(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)


    class SegUNet(nn.Module):
        """32 by 128 pixel labels. Channel attention sits on the bottleneck only."""

        def __init__(self, n_class: int = 4):
            super().__init__()
            self.enc1 = _Block(1, 16)
            self.enc2 = _Block(16, 32)
            self.pool = nn.MaxPool2d(2)
            self.bot = _Block(32, 64)
            self.attend = nn.Sequential(nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 64), nn.Sigmoid())
            self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec2 = _Block(64, 32)
            self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dec1 = _Block(32, 16)
            self.head = nn.Conv2d(16, n_class, 1)

        def forward(self, x):
            skip1 = self.enc1(x)
            skip2 = self.enc2(self.pool(skip1))
            hidden = self.bot(self.pool(skip2))
            weight = self.attend(hidden.mean(dim=(2, 3)))
            hidden = hidden * weight[:, :, None, None]
            high = self.dec2(torch.cat([self.up2(hidden), skip2], dim=1))
            low = self.dec1(torch.cat([self.up1(high), skip1], dim=1))
            return self.head(low)

else:  # pragma: no cover
    DilatedDenoiser = None  # type: ignore[misc, assignment]
    GradientFusion = None  # type: ignore[misc, assignment]
    SegUNet = None  # type: ignore[misc, assignment]
