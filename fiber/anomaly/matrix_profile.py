"""Discord scores from a matrix profile along time, one distance bin at a time.

STUMPY `stump` is required. `mstump` is used only when `multivariate=True`.
The value passed to extract_zones is signed by the column residual so a hot
discord and a cold discord stay on opposite sides of zero.
"""

from __future__ import annotations

import numpy as np

from fiber.models.ssl_common import live_mask


def discord_score(
    field: np.ndarray,
    valid: np.ndarray,
    *,
    window: int = 8,
    multivariate: bool = False,
) -> np.ndarray:
    import stumpy

    values = np.asarray(field, dtype=np.float64)
    mask = live_mask(field, valid)
    score = np.zeros(values.shape, dtype=np.float32)
    if window < 3 or values.shape[0] <= window:
        return score
    profiles = np.zeros(values.shape[0], dtype=np.float64)
    counts = np.zeros(values.shape[0], dtype=np.float64)
    for column in range(values.shape[1]):
        series = values[:, column].copy()
        ok = mask[:, column]
        if int(ok.sum()) <= window:
            continue
        fill = float(np.median(series[ok]))
        series[~ok] = fill
        if multivariate and column + 1 < values.shape[1]:
            neighbour = values[:, column + 1].copy()
            neighbour_ok = mask[:, column + 1]
            if int(neighbour_ok.sum()) <= window:
                profile = stumpy.stump(series, m=window)[:, 0]
            else:
                neighbour[~neighbour_ok] = float(np.median(neighbour[neighbour_ok]))
                profile = stumpy.mstump(np.vstack([series, neighbour]), m=window)[0][0]
        else:
            profile = stumpy.stump(series, m=window)[:, 0]
        profile = np.asarray(profile, dtype=np.float64)
        start = window // 2
        stop = start + profile.size
        profiles[start:stop] += np.nan_to_num(profile, nan=0.0)
        counts[start:stop] += 1.0
        residual = series - fill
        sign = np.sign(residual).astype(np.float32)
        sign[sign == 0] = 1.0
        column_score = np.zeros(values.shape[0], dtype=np.float32)
        column_score[start:stop] = np.nan_to_num(profile, nan=0.0).astype(np.float32)
        score[:, column] = column_score * sign
    finite = score[mask]
    if finite.size:
        center = float(np.median(np.abs(finite)))
        mad = 1.4826 * float(np.median(np.abs(np.abs(finite) - center)))
        mad = mad if mad > 1e-6 else 1e-6
        score /= np.float32(mad)
    score[~mask] = 0.0
    return score
