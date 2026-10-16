"""Event scores on a sampled time x distance grid.

Matching is one-to-one by intersection-over-union. A single hit inside a
long event does not count as a full detection. Predictions on invalid cells
are counted and removed from the score.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from fiber.grid import sample_widths, segments


def _weights(time_s: np.ndarray, x_m: np.ndarray, max_gap_s: float = 300.0) -> np.ndarray:
    cell_dx = sample_widths(x_m)
    dt = sample_widths(time_s, max_gap=max_gap_s)
    return dt[:, None] * cell_dx[None, :]


def _components(mask: np.ndarray, time_s: np.ndarray, max_gap_s: float) -> tuple[np.ndarray, int]:
    labeled = np.zeros(mask.shape, dtype=np.int32)
    count = 0
    for part in segments(time_s, max_gap_s):
        chunk, found = ndimage.label(mask[part])
        chunk[chunk > 0] += count
        labeled[part] = chunk
        count += found
    return labeled, count


def evaluate_events(
    truth: np.ndarray,
    prediction: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    min_iou: float = 0.2,
    max_gap_s: float = 300.0,
) -> dict:
    """Binary event precision/recall. Unknown cells must be False in both masks."""
    true = np.asarray(truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    if true.shape != pred.shape or true.ndim != 2:
        raise ValueError("truth and prediction must share a [time, distance] shape")
    if valid is None:
        valid = np.ones(true.shape, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
    if valid.shape != true.shape or true.shape != (len(time_s), len(x_m)):
        raise ValueError("valid and physical axes must match the field")
    if not 0 < min_iou <= 1:
        raise ValueError("min_iou must lie in (0, 1]")
    invalid_pred = int(np.count_nonzero(pred & ~valid))
    true = true & valid
    pred = pred & valid
    weights = _weights(time_s, x_m, max_gap_s)
    true = true & (weights > 0)
    pred = pred & (weights > 0)
    tl, n_true = _components(true, time_s, max_gap_s)
    pl, n_pred = _components(pred, time_s, max_gap_s)
    true_area = np.bincount(tl.ravel(), weights=weights.ravel(), minlength=n_true + 1)[1:]
    pred_area = np.bincount(pl.ravel(), weights=weights.ravel(), minlength=n_pred + 1)[1:]
    inter = np.zeros((n_true, n_pred), dtype=np.float64)
    overlap = (tl > 0) & (pl > 0)
    if n_true and n_pred and overlap.any():
        np.add.at(inter, (tl[overlap] - 1, pl[overlap] - 1), weights[overlap])
    union = true_area[:, None] + pred_area[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    eligible = iou >= min_iou
    matched = 0
    if n_true and n_pred:
        reward = np.where(eligible, 1.0 + iou, 0.0)
        rows, cols = linear_sum_assignment(reward, maximize=True)
        matched = int(sum(bool(eligible[r, c]) for r, c in zip(rows, cols, strict=True)))
    precision = matched / n_pred if n_pred else None
    recall = matched / n_true if n_true else None
    f1 = (2 * matched / (n_true + n_pred)) if (n_true + n_pred) else None
    inter_sum = float(inter.sum())
    true_sum = float(true_area.sum())
    pred_sum = float(pred_area.sum())
    return {
        "truth_events": n_true,
        "predicted_events": n_pred,
        "matched_events": matched,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "support_iou": (inter_sum / (true_sum + pred_sum - inter_sum)) if (true_sum + pred_sum - inter_sum) else None,
        "predicted_invalid_cells": invalid_pred,
        "min_iou": min_iou,
        "max_gap_s": max_gap_s,
        "support_convention": "midpoint_cells_no_gap_time",
    }


def alarm_fraction(prediction: np.ndarray, valid: np.ndarray) -> float:
    mask = np.asarray(valid, dtype=bool)
    if not mask.any():
        return 0.0
    return float(np.mean(np.asarray(prediction, dtype=bool)[mask]))
