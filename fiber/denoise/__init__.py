"""Denoise zoo: standard DSP, modern filters, research CNN."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from fiber.denoise import dsp

DenoiseFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

# family: standard | modern | research
METHODS: dict[str, dict] = {
    "identity": {
        "family": "standard",
        "fn": dsp.identity,
        "doc": "No smoothing. Residual is zero. Control channel.",
    },
    "median_x3": {
        "family": "standard",
        "fn": lambda raw, valid: dsp.median_x(raw, valid, k=3),
        "doc": "Spatial median, 3 samples. Keeps narrow peaks better than larger kernels.",
    },
    "median_x5": {
        "family": "standard",
        "fn": lambda raw, valid: dsp.median_x(raw, valid, k=5),
        "doc": "Spatial median, 5 samples. Stronger spike kill, more peak loss.",
    },
    "median_xt": {
        "family": "standard",
        "fn": dsp.median_xt,
        "doc": "3x3 nan-median on the time-distance waterfall.",
    },
    "gaussian_x": {
        "family": "standard",
        "fn": dsp.gaussian_x,
        "doc": "Gaussian smoothing along the fibre on valid runs.",
    },
    "savgol_x": {
        "family": "standard",
        "fn": dsp.savgol_x,
        "doc": "Savitzky-Golay polynomial smoother on valid runs.",
    },
    "temporal_median": {
        "family": "standard",
        "fn": dsp.temporal_median,
        "doc": "Median along time at each metre. Good for shot noise, keeps spatial edges.",
    },
    "bilateral_x": {
        "family": "modern",
        "fn": dsp.bilateral_x,
        "doc": "Edge-preserving bilateral filter (Tomasi/Manduchi) in 1D.",
    },
    "wavelet_x": {
        "family": "modern",
        "fn": dsp.wavelet_x,
        "doc": "VisuShrink wavelet threshold, Donoho-Johnstone 1994.",
    },
    "wavelet_bayes": {
        "family": "modern",
        "fn": dsp.wavelet_bayes,
        "doc": "BayesShrink, per-level threshold, Chang/Yu/Vetterli 2000.",
    },
    "nlm_x": {
        "family": "modern",
        "fn": dsp.nlm_x,
        "doc": "1D non-local means, Buades/Coll/Morel. Short search, local patches.",
    },
    "nlm_wide": {
        "family": "modern",
        "fn": dsp.nlm_wide,
        "doc": "Non-local means with a 21-sample search and h scaled by 1.5.",
    },
    "wiener_x": {
        "family": "standard",
        "fn": dsp.wiener_x,
        "doc": "Local Wiener filter along the fibre.",
    },
    "tv_x": {
        "family": "modern",
        "fn": dsp.tv_x,
        "doc": "1D total-variation (ROF) smoothing. Piecewise-constant bias.",
    },
}


def available_methods() -> list[str]:
    return list(METHODS)


def apply_method(
    name: str,
    raw_c: np.ndarray,
    valid: np.ndarray,
    *,
    weights: Path | None = None,
    device: str | None = None,
) -> np.ndarray:
    if name == "conv1d_n2self":
        from fiber.models.conv1d_denoise import apply_conv1d

        path = weights or Path("models/weights/conv1d_n2self.pt")
        return apply_conv1d(raw_c, valid, path)
    if name == "unet2d_n2self":
        from fiber.models.unet2d_denoise import apply_unet2d

        path = weights or Path("models/weights/unet2d_n2self.pt")
        return apply_unet2d(raw_c, valid, path)
    if name == "n2n_tsr":
        from fiber.models.n2n_tsr import denoise_n2n_tsr

        path = weights or Path("models/weights/n2n_tsr.pt")
        saved = path if Path(path).exists() else None
        steps = 0 if saved else (4 if device == "cpu" else 12)
        return denoise_n2n_tsr(raw_c, valid, seed=0, steps=steps, device=device, weights=saved)
    if name == "blindspot_x":
        from fiber.models.blindspot_x import denoise_blindspot_x

        path = weights or Path("models/weights/blindspot_x.pt")
        saved = path if Path(path).exists() else None
        return denoise_blindspot_x(raw_c, valid, seed=0, steps=0, device=device, weights=saved)
    if name == "zsn2n":
        from fiber.models.zsn2n import denoise_zsn2n

        path = weights or Path("models/weights/zsn2n.pt")
        saved = path if Path(path).exists() else None
        steps = 0 if saved else (4 if device == "cpu" else 12)
        return denoise_zsn2n(raw_c, valid, seed=0, steps=steps, device=device, weights=saved)
    if name not in METHODS:
        known = ", ".join(available_methods())
        raise KeyError(f"unknown denoise method {name!r}; known: {known}")
    return METHODS[name]["fn"](raw_c, valid)


def residual(raw_c: np.ndarray, denoised: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(raw_c, dtype=np.float32) - np.asarray(denoised, dtype=np.float32)
    out[~np.asarray(valid, dtype=bool)] = np.nan
    return out
