"""Versioned offline shape features in metres and seconds.

These describe a candidate region over its supplied observation history.
They are not a cause classifier, a streaming prediction or an expert label.
Unlike context-v1, a permutation of time rows changes episode features.
"""
from __future__ import annotations

import numpy as np
import json
from pathlib import Path

from fiber.grid import sample_widths, segments, validate_axis

SCHEMA = 'physical_shape_v1'
FEATURE_NAMES = (
    'positive_peak_c', 'negative_peak_c', 'rms_c', 'spatial_width_m',
    'positive_duty', 'negative_duty', 'positive_episodes', 'negative_episodes',
    'longest_episode_s', 'period_s', 'period_cv', 'step_c', 'coverage',
)


def shape_features(field, valid, x_m, time_s, *, max_gap_s=300.0) -> np.ndarray:
    x = validate_axis(x_m, 'x_m')
    times = validate_axis(time_s, 'time_s')
    values = np.asarray(field, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.shape != mask.shape or values.shape != (len(times), len(x)):
        raise ValueError('physical feature field and axes must agree')
    dx = sample_widths(x)
    dt = sample_widths(times, max_gap=max_gap_s)
    # A row's common temperature is removed; temporal HVAC drift is not a
    # local event. All statistics omit missing observations rather than fill.
    row_live = mask.any(axis=1)
    centered = np.full_like(values, np.nan)
    for row in np.flatnonzero(row_live):
        centered[row, mask[row]] = values[row, mask[row]] - np.median(values[row, mask[row]])
    live_cols = mask.any(axis=0)
    if not live_cols.any():
        raise ValueError('physical shape has no observations')
    energy = np.zeros(len(x))
    profile = np.full(len(x), np.nan)
    for col in np.flatnonzero(live_cols):
        kept = mask[:, col]
        energy[col] = np.average(np.abs(centered[kept, col]), weights=dt[kept])
        profile[col] = np.median(centered[kept, col])
    anchor = int(np.argmax(energy))
    active_cols = live_cols & (energy >= 0.5 * max(energy[anchor], 1e-12))
    # Only the connected candidate around the strongest column, not disjoint
    # events elsewhere in the supplied region.
    lo = anchor
    hi = anchor + 1
    while lo > 0 and active_cols[lo-1]: lo -= 1
    while hi < len(x) and active_cols[hi]: hi += 1
    core = np.arange(lo, hi)
    signal = np.full(len(times), np.nan)
    for row in np.flatnonzero(row_live):
        kept = core[mask[row, core]]
        if len(kept):
            signal[row] = np.average(centered[row, kept], weights=dx[kept])
    finite = np.isfinite(signal)
    peak_pos = max(0.0, float(np.max(signal[finite])))
    peak_neg = max(0.0, float(-np.min(signal[finite])))
    threshold = max(1.0, 0.35 * max(peak_pos, peak_neg))
    hot = finite & (signal >= threshold)
    cold = finite & (signal <= -threshold)
    counts = []
    durations = []
    repeat_intervals = []
    for support in (hot, cold):
        count = 0
        for part in segments(times, max_gap_s):
            # Missing rows are barriers, not inferred quiet or active states.
            starts = []
            begin = None
            for row in range(part.start, part.stop + 1):
                active = row < part.stop and support[row]
                if active and begin is None:
                    begin = row
                    starts.append(row)
                if not active and begin is not None:
                    durations.append(float(dt[begin:row].sum()))
                    count += 1
                    begin = None
            # Period evidence needs at least 3 repeats in one observed segment.
            if len(starts) >= 3 and finite[part].all():
                repeat_intervals.extend(np.diff(times[starts]).tolist())
        counts.append(count)
    observed_time = float(dt[finite].sum())
    intervals = np.asarray(repeat_intervals)
    period = float(np.median(intervals)) if len(intervals) >= 2 else 0.0
    cv = float(np.std(intervals) / np.mean(intervals)) if len(intervals) >= 2 else 1.0
    live_profile = profile[live_cols]
    side = max(1, len(live_profile) // 4)
    step = float(np.median(live_profile[:side]) - np.median(live_profile[-side:]))
    return np.array([
        peak_pos, peak_neg, np.sqrt(np.average(signal[finite]**2, weights=dt[finite])),
        float(dx[core].sum()), float(dt[hot].sum()/observed_time), float(dt[cold].sum()/observed_time),
        *counts, max(durations, default=0.0), period, cv, step,
        float(np.sum(mask*dt[:,None]*dx[None,:]) / (dt.sum()*dx.sum())),
    ], dtype=np.float64)


def selection_under_budget(challenger: dict, baseline: dict | None, *, max_false_alarm_rate: float) -> dict:
    """Explicit opt-in deployment gate; never silently replace the v1 research policy."""
    if not 0 <= max_false_alarm_rate <= 1:
        raise ValueError('false alarm budget must lie in [0, 1]')
    eligible = []
    rejected = {}
    for name, score in [('challenger', challenger), ('baseline', baseline)]:
        if score is None:
            rejected[name] = 'missing_score'
            continue
        recall, far = score.get('event_recall'), score.get('false_alarm_rate')
        if recall is None or far is None or not np.isfinite(recall) or not np.isfinite(far):
            rejected[name] = 'missing_or_nonfinite_metric'
        elif not (0 <= recall <= 1 and 0 <= far <= 1):
            rejected[name] = 'invalid_metric'
        elif far > max_false_alarm_rate:
            rejected[name] = 'false_alarm_budget_exceeded'
        else:
            eligible.append((float(recall), -float(far), name))
    best = max(eligible, default=None)
    return {'selected': best[2] if best else None, 'max_false_alarm_rate': max_false_alarm_rate, 'rejected': rejected}


def load_physical_model(path: str | Path) -> dict:
    """Load research weights only when schema, label order and dimensions agree."""
    from fiber.classify.rules import LABELS
    blob = json.loads(Path(path).read_text(encoding='utf-8'))
    if blob.get('schema') != SCHEMA or blob.get('features') != list(FEATURE_NAMES) or blob.get('labels') != list(LABELS):
        raise ValueError('physical model feature/label schema mismatch')
    if blob.get('serve') != 'research_offline' or blob.get('expert_confirmed') is not False:
        raise ValueError('physical model must retain its research provenance')
    shapes = {'mu':(len(FEATURE_NAMES),),'sig':(len(FEATURE_NAMES),),
              'weight':(len(FEATURE_NAMES),len(LABELS)),'bias':(len(LABELS),)}
    for key,shape in shapes.items():
        arr = np.asarray(blob[key], dtype=np.float64)
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f'invalid physical model {key}')
        blob[key] = arr
    if np.any(blob['sig'] <= 0) or blob.get('threshold') is None or not 0 <= blob['threshold'] <= 1:
        raise ValueError('invalid or unselected physical model gate')
    return blob


def classify_physical_shape(model: dict, field, valid, x_m, time_s) -> dict:
    """Offline research inference; scores are softmax outputs, not calibrated risk."""
    from fiber.classify.rules import LABELS
    features = shape_features(field,valid,x_m,time_s)
    logits = (features-model['mu'])/model['sig'] @ model['weight'] + model['bias']
    score = np.exp(logits-logits.max())
    score /= score.sum()
    index = int(np.argmax(score)) if float(score.max()) >= model['threshold'] else 0
    return {'shape_hypothesis':LABELS[index], 'feature_schema':SCHEMA,
            'scores':dict(zip(LABELS,score.tolist(),strict=True)),
            'score_semantics':'uncalibrated_softmax','expert_confirmed':False,
            'serve':'research_offline','features':dict(zip(FEATURE_NAMES,features.tolist(),strict=True))}
