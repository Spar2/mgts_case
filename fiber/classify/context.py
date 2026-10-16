"""Spatial context around a search window.

A 40-column crop sees a narrow full-time streak and the edge of a wide step
as the same jump. The profile here extends 40 columns past each side of the
crop. Width is counted at half the crop's own peak, so a taller event does
not automatically look wider. The fraction, the side width, and the feature
list were locked on planted cables before the held-out search seeds.
"""

from __future__ import annotations

import numpy as np

from fiber.classify.plant import SoftmaxSceneModel
from fiber.classify.rules import LABELS
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP

CONTEXT_SIDE = 40
CONTEXT_PEAK_FRACTION = 0.5
SPIKE_MAD_FLOOR = 0.05
FEATURE_NAMES = ("support", "peak", "n_half", "spike", "runs")


def _runs(mask: np.ndarray) -> int:
    if mask.size == 0 or not bool(mask.any()):
        return 0
    padded = np.concatenate((np.zeros(1, dtype=bool), mask, np.zeros(1, dtype=bool)))
    return int(np.count_nonzero(padded[1:] & ~padded[:-1]))


def _column_medians(slab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    count = np.asarray(mask).sum(axis=0)
    med = np.full(int(count.shape[0]), np.nan, dtype=np.float64)
    live = count > 0
    if not live.any():
        return med
    kept = np.where(np.asarray(mask)[:, live], np.asarray(slab, dtype=np.float64)[:, live], np.nan)
    med[live] = np.nanmedian(kept, axis=0)
    return med


def _features(prof: np.ndarray, cell_dev: np.ndarray) -> np.ndarray:
    """Five numbers. `cell_dev` is the crop after its own median is removed."""
    center = prof[CONTEXT_SIDE:CONTEXT_SIDE + SPACE_CROP]
    finite = np.isfinite(center)
    if not finite.any():
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    peak = float(center[int(np.nanargmax(np.abs(center)))])
    if abs(peak) < 1e-8:
        support = 0
        runs = 0
    else:
        same = (
            np.isfinite(prof)
            & (np.sign(prof) == np.sign(peak))
            & (np.abs(prof) >= CONTEXT_PEAK_FRACTION * abs(peak))
        )
        support = int(same.sum())
        runs = _runs(same)
    if cell_dev.size == 0:
        return np.array([support, peak, 0.0, 0.0, runs], dtype=np.float64)
    peak_abs = float(np.max(np.abs(cell_dev)))
    if peak_abs < 1e-8:
        n_half = 0
        spike = 0.0
    else:
        n_half = int(np.count_nonzero(np.abs(cell_dev) >= CONTEXT_PEAK_FRACTION * peak_abs))
        mad = float(np.median(np.abs(cell_dev - np.median(cell_dev))))
        spike = peak_abs / max(mad, SPIKE_MAD_FLOOR)
    return np.array([support, peak, n_half, spike, runs], dtype=np.float64)


def _window_arrays(field: np.ndarray, valid: np.ndarray, t0: int, x0: int, col_med: np.ndarray):
    t1 = t0 + TIME_CROP
    x1 = x0 + SPACE_CROP
    slab = np.asarray(field[t0:t1], dtype=np.float64)
    mask = np.asarray(valid[t0:t1], dtype=bool)
    block = np.where(mask[:, x0:x1], slab[:, x0:x1], np.nan)
    finite = np.isfinite(block)
    if not finite.any():
        empty = np.full(CONTEXT_SIDE + SPACE_CROP + CONTEXT_SIDE, np.nan)
        return empty, np.zeros(0, dtype=np.float64)
    center = float(np.median(block[finite]))
    width = CONTEXT_SIDE + SPACE_CROP + CONTEXT_SIDE
    prof = np.full(width, np.nan)
    x_lo = x0 - CONTEXT_SIDE
    x_hi = x0 + SPACE_CROP + CONTEXT_SIDE
    left = max(x_lo, 0)
    right = min(x_hi, col_med.shape[0])
    dest = left - x_lo
    prof[dest:dest + (right - left)] = col_med[left:right] - center
    return prof, block[finite] - center


WIDE_COLUMNS = CONTEXT_SIDE + SPACE_CROP + CONTEXT_SIDE


def wide_context_crop(field: np.ndarray, valid: np.ndarray, t0: int, x0: int) -> np.ndarray:
    """28×120 field. The center 40 columns use the crop median; wings keep that same zero.

    Cells outside the cable and invalid cells stay 0. The label of the window
    still comes from the center crop alone.
    """
    t0 = int(t0)
    x0 = int(x0)
    t1 = t0 + TIME_CROP
    slab = np.asarray(field[t0:t1], dtype=np.float64)
    mask = np.asarray(valid[t0:t1], dtype=bool)
    center_block = slab[:, x0:x0 + SPACE_CROP]
    center_mask = mask[:, x0:x0 + SPACE_CROP]
    center = float(np.median(center_block[center_mask])) if center_mask.any() else 0.0
    out = np.zeros((TIME_CROP, WIDE_COLUMNS), dtype=np.float32)
    x_lo = x0 - CONTEXT_SIDE
    left = max(x_lo, 0)
    right = min(x_lo + WIDE_COLUMNS, slab.shape[1])
    if right <= left:
        return out
    piece_mask = mask[:, left:right]
    piece = np.zeros((TIME_CROP, right - left), dtype=np.float32)
    if piece_mask.any():
        values = slab[:, left:right]
        piece[piece_mask] = (values[piece_mask] - center).astype(np.float32)
    dest = left - x_lo
    out[:, dest:dest + (right - left)] = piece
    return out


def wide_context_crops(field: np.ndarray, valid: np.ndarray, origins: np.ndarray) -> np.ndarray:
    origins = np.asarray(origins, dtype=np.int64)
    out = np.zeros((len(origins), TIME_CROP, WIDE_COLUMNS), dtype=np.float32)
    for index, (t0, x0) in enumerate(origins):
        out[index] = wide_context_crop(field, valid, int(t0), int(x0))
    return out


def window_context(field: np.ndarray, valid: np.ndarray, t0: int, x0: int) -> np.ndarray:
    """Context features for one crop. Invalid and out-of-range columns stay out."""
    t1 = int(t0) + TIME_CROP
    col_med = _column_medians(np.asarray(field[t0:t1], dtype=np.float64), np.asarray(valid[t0:t1], dtype=bool))
    prof, cell_dev = _window_arrays(field, valid, int(t0), int(x0), col_med)
    return _features(prof, cell_dev)


def context_table(field: np.ndarray, valid: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """One row of `FEATURE_NAMES` per origin. An empty origin list returns shape (0, 5)."""
    origins = np.asarray(origins, dtype=np.int64)
    out = np.zeros((len(origins), len(FEATURE_NAMES)), dtype=np.float64)
    if len(origins) == 0:
        return out
    field64 = np.asarray(field, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    order = np.argsort(origins[:, 0], kind="mergesort")
    cached_t = None
    col_med = None
    for index in order:
        t0 = int(origins[index, 0])
        x0 = int(origins[index, 1])
        if cached_t != t0:
            col_med = _column_medians(field64[t0:t0 + TIME_CROP], mask[t0:t0 + TIME_CROP])
            cached_t = t0
        prof, cell_dev = _window_arrays(field64, mask, t0, x0, col_med)
        out[index] = _features(prof, cell_dev)
    return out


def dump_context_model(model: SoftmaxSceneModel, extra: dict | None = None) -> dict:
    if model.weight is None or model.mu is None or model.sig is None or model.bias is None:
        raise RuntimeError("model is not fit")
    payload = {
        "family": "softmax_context",
        "features": list(FEATURE_NAMES),
        "context_side": CONTEXT_SIDE,
        "peak_fraction": CONTEXT_PEAK_FRACTION,
        "spike_mad_floor": SPIKE_MAD_FLOOR,
        "labels": list(LABELS),
        "mu": model.mu.tolist(),
        "sig": model.sig.tolist(),
        "weight": model.weight.tolist(),
        "bias": model.bias.tolist(),
    }
    if extra:
        payload.update(extra)
    return payload


def load_context_model(payload: dict) -> SoftmaxSceneModel:
    if list(payload["features"]) != list(FEATURE_NAMES):
        raise ValueError("context feature list does not match this code")
    if int(payload["context_side"]) != CONTEXT_SIDE or float(payload["peak_fraction"]) != CONTEXT_PEAK_FRACTION:
        raise ValueError("context constants do not match this code")
    model = SoftmaxSceneModel()
    model.mu = np.asarray(payload["mu"], dtype=np.float64)
    model.sig = np.asarray(payload["sig"], dtype=np.float64)
    model.weight = np.asarray(payload["weight"], dtype=np.float64)
    model.bias = np.asarray(payload["bias"], dtype=np.float64)
    return model
