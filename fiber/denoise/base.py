"""Shared helpers for masked 1D/2D denoisers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def nan_field(raw_c: np.ndarray, valid: np.ndarray) -> np.ndarray:
    field = np.asarray(raw_c, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if field.shape != mask.shape:
        raise ValueError("raw_c and valid must share a shape")
    out = field.copy()
    out[~mask] = np.nan
    return out


def restore_invalid(denoised: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(denoised, dtype=np.float32)
    out[~np.asarray(valid, dtype=bool)] = np.nan
    return out


def contiguous_runs(valid_row: np.ndarray) -> list[tuple[int, int]]:
    v = np.asarray(valid_row, dtype=bool)
    padded = np.concatenate([[False], v, [False]])
    delta = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(delta == 1)
    ends = np.flatnonzero(delta == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def map_segments(
    values: np.ndarray,
    valid: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
    min_len: int = 5,
) -> np.ndarray:
    """Apply `fn` independently on each contiguous valid run of a 1D row."""
    src = np.asarray(values, dtype=np.float64)
    out = np.full(src.shape, np.nan, dtype=np.float64)
    for a, b in contiguous_runs(valid):
        if b - a < min_len:
            out[a:b] = src[a:b]
            continue
        piece = np.asarray(fn(src[a:b]), dtype=np.float64)
        if piece.shape != (b - a,):
            piece = piece[: b - a]
        out[a:b] = piece
    return out


def apply_rows(
    raw_c: np.ndarray,
    valid: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
    min_len: int = 5,
) -> np.ndarray:
    field = nan_field(raw_c, valid)
    mask = np.asarray(valid, dtype=bool)
    out = np.full(field.shape, np.nan, dtype=np.float64)
    for i in range(field.shape[0]):
        out[i] = map_segments(field[i], mask[i], fn, min_len=min_len)
    return restore_invalid(out, mask)
