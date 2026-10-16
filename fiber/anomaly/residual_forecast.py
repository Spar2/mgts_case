"""Chronos-Bolt residual forecast on each distance column.

The requested interval is q05-q95. Chronos-Bolt's trained grid is q10-q90, and
predict_quantiles substitutes those endpoints (the numeric values match). The
score uses the levels the pipeline actually emits, one interval per time, from
the history that ends at that time. A missing package or a failed weight
download raises ImportError / RuntimeError. Callers must not invent a number.
"""

from __future__ import annotations

import numpy as np

from fiber.models.ssl_common import live_mask, torch_device

REQUESTED_QUANTILES = (0.05, 0.95)
MIN_HISTORY = 4
_BATCH = 32


def _emitted_quantile_levels(pipeline, requested: tuple[float, float] = REQUESTED_QUANTILES) -> tuple[float, float]:
    """Levels Chronos will actually emit for a q05/q95 request.

    Bolt warns and clamps anything outside its trained grid to the nearest
    trained endpoint. On the current tiny checkpoint that is q10 and q90,
    and the clamped tensor is identical to requesting those endpoints.
    """
    trained = [float(level) for level in pipeline.quantiles]
    if len(trained) < 2:
        raise RuntimeError("Chronos pipeline did not expose trained quantile levels")
    low_req, high_req = requested
    eligible_low = [level for level in trained if level >= low_req - 1e-9]
    eligible_high = [level for level in trained if level <= high_req + 1e-9]
    low = min(eligible_low) if eligible_low else min(trained)
    high = max(eligible_high) if eligible_high else max(trained)
    if not low < high:
        raise RuntimeError(f"Chronos emitted quantile levels are not ordered: {low}, {high}")
    return low, high


def forecast_score(
    field: np.ndarray,
    valid: np.ndarray,
    *,
    model_name: str = "amazon/chronos-bolt-tiny",
    context: int = 32,
    device: str | None = None,
    column_stride: int = 1,
) -> np.ndarray:
    try:
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        raise ImportError("chronos-forecasting is not installed") from exc
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for Chronos-Bolt") from exc

    values = np.asarray(field, dtype=np.float32)
    mask = live_mask(field, valid)
    score = np.zeros(values.shape, dtype=np.float32)
    if values.shape[0] <= MIN_HISTORY:
        return score
    chosen = torch_device(device)
    try:
        pipeline = BaseChronosPipeline.from_pretrained(model_name, device_map=chosen)
    except Exception as exc:
        raise RuntimeError(f"Chronos-Bolt weights could not be loaded: {exc}") from exc
    emitted = _emitted_quantile_levels(pipeline)
    forecast_score.emitted_quantile_levels = emitted
    grouped: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    for column in range(0, values.shape[1], max(1, column_stride)):
        for time_index in range(MIN_HISTORY, values.shape[0]):
            start = max(0, time_index - context)
            hist_mask = mask[start:time_index, column]
            length = time_index - start
            if int(hist_mask.sum()) < max(2, length // 2):
                continue
            series = values[start:time_index, column].astype(np.float32, copy=True)
            if not bool(hist_mask.all()):
                series[~hist_mask] = np.float32(np.median(series[hist_mask]))
            grouped.setdefault(length, []).append((time_index, column, series))

    for jobs in grouped.values():
        for offset in range(0, len(jobs), _BATCH):
            chunk = jobs[offset:offset + _BATCH]
            context_tensor = torch.tensor(np.stack([job[2] for job in chunk]), dtype=torch.float32)
            try:
                quantiles, _mean = pipeline.predict_quantiles(
                    inputs=context_tensor,
                    prediction_length=1,
                    quantile_levels=[emitted[0], emitted[1]],
                )
            except Exception as exc:
                raise RuntimeError(f"Chronos-Bolt predict_quantiles failed: {exc}") from exc
            predicted = quantiles.detach().cpu().numpy()
            if predicted.ndim != 3 or predicted.shape[0] != len(chunk) or predicted.shape[1] < 1 or predicted.shape[2] < 2:
                raise RuntimeError(
                    f"Chronos-Bolt quantiles have shape {predicted.shape}, expected (batch, 1, 2)"
                )
            # Horizon 0 is this time only. A later time gets its own call.
            low = predicted[:, 0, 0]
            high = predicted[:, 0, 1]
            for (time_index, column, _series), q_low, q_high in zip(chunk, low, high, strict=True):
                if not mask[time_index, column]:
                    continue
                observed = float(values[time_index, column])
                span = max(float(q_high) - float(q_low), 1e-3)
                if observed > q_high:
                    score[time_index, column] = np.float32((observed - float(q_high)) / span)
                elif observed < q_low:
                    score[time_index, column] = np.float32((observed - float(q_low)) / span)
    score[~mask] = 0.0
    return score
