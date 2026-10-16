"""Real-measurement diagnostics and abstaining weak labels.

No synthetic scenes, fitted classifier, expert truth or hardware diagnosis.
Thresholds are explicit development assumptions, not calibrated alarm gates.
"""
from __future__ import annotations

import warnings
import numpy as np

from fiber.grid import segments, sample_widths, validate_axis

RULE_CONFIG = {
    "contrast_c": 1.0, "min_coverage": 0.8, "min_reference_columns": 3,
    "reference_inner_m": 8.0, "reference_outer_m": 40.0,
    "min_episode_rows": 3, "min_episode_s": 180.0, "max_gap_s": 300.0,
    "period_cv_max": 0.3, "min_repeats": 3,
    "split_train_fraction": 0.6, "split_dev_fraction": 0.8,
    "split_guard_s": 1800.0,
}


def median(values, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def quantiles(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return None
    return dict(zip(("q01", "q10", "q50", "q90", "q99"),
                    map(float, np.quantile(finite, [.01, .1, .5, .9, .99])), strict=True))


def correlation(a, b, eligible=None, limit=200_000):
    mask = np.isfinite(a) & np.isfinite(b)
    if eligible is not None:
        mask &= eligible
    ids = np.flatnonzero(mask)
    count = len(ids)
    ids = ids[::max(1, int(np.ceil(count / limit)))]
    x, y = np.asarray(a).ravel()[ids], np.asarray(b).ravel()[ids]
    value = None
    if len(ids) >= 3 and np.std(x) > 0 and np.std(y) > 0:
        value = float(np.corrcoef(x, y)[0, 1])
    return {"pearson": value, "eligible_pairs": count, "sampled_pairs": len(ids)}


def temporal_partition(time_s, guard_s=1800.0):
    """Chronological development split with an excluded interval at boundaries.

    Codes: -1 excluded guard, 0 train, 1 development, 2 internal holdout.
    A crop touching different codes must never be assigned to one split.
    """
    t = validate_axis(time_s, "time_s")
    if not np.isfinite(guard_s) or guard_s < 0:
        raise ValueError("guard_s must be finite and nonnegative")
    c1, c2 = t[min(len(t)-1, int(len(t)*.6))], t[min(len(t)-1, int(len(t)*.8))]
    labels = np.full(len(t), -1, np.int8)
    labels[t < c1-guard_s/2] = 0
    labels[(t >= c1+guard_s/2) & (t < c2-guard_s/2)] = 1
    labels[t >= c2+guard_s/2] = 2
    return labels, {"cut_time_s": [float(c1), float(c2)], "guard_s": guard_s,
                    "counts": {str(k): int((labels == k).sum()) for k in (-1, 0, 1, 2)},
                    "codes": {"-1": "excluded_guard", "0": "train", "1": "development", "2": "internal_holdout"},
                    "unseen_measurement_test": False}


def crop_partition(codes):
    unique = np.unique(codes)
    return int(unique[0]) if len(unique) == 1 and unique[0] >= 0 else -1


def diagnose_record(raw, clean, valid, x_m, time_s):
    """Observed variation and filter modification; neither is true denoising MSE."""
    x, t = validate_axis(x_m, "x_m"), validate_axis(time_s, "time_s")
    raw = np.where(valid & np.isfinite(raw), raw, np.nan).astype(float)
    clean = np.where(valid & np.isfinite(clean), clean, np.nan).astype(float)
    residual = raw-clean
    adjacency = np.diff(x) <= 1.5*np.median(np.diff(x)) if len(x) > 1 else np.zeros(0, bool)
    continuous = np.diff(t) <= RULE_CONFIG["max_gap_s"]
    changes = np.diff(raw, axis=0)
    changes[~continuous] = np.nan
    common_change = median(changes, axis=1)
    differential = changes-common_change[:, None]
    spatial = np.diff(raw, axis=1)
    spatial[:, ~adjacency] = np.nan
    clean_spatial = np.diff(clean, axis=1)
    clean_spatial[:, ~adjacency] = np.nan
    profile_level = median(raw, axis=1)
    live_residual = residual[np.isfinite(residual)]
    modification = {
        "abs_change_c": quantiles(np.abs(residual)),
        "changed_valid_fraction": float(np.count_nonzero(np.abs(live_residual) > 1e-6)/len(live_residual)) if len(live_residual) else None,
        "raw_spatial_increment_abs_c": quantiles(np.abs(spatial)),
        "clean_spatial_increment_abs_c": quantiles(np.abs(clean_spatial)),
        "interpretation": "filter modification and roughness, not error versus clean truth",
    }
    return {
        "shape": list(raw.shape), "valid_cells": int(np.isfinite(raw).sum()),
        "coverage": float(np.isfinite(raw).mean()),
        "dx_m": quantiles(np.diff(x)), "cadence_s": quantiles(np.diff(t)),
        "segments": [[s.start, s.stop] for s in segments(t)],
        "temperature_c": quantiles(raw), "profile_median_c": quantiles(profile_level),
        "temporal_increment_abs_c": quantiles(np.abs(changes)),
        "differential_increment_abs_c": quantiles(np.abs(differential)),
        "increment_variability_proxy_c": float(median(np.abs(differential))/.67448975/np.sqrt(2)) if np.isfinite(differential).any() else None,
        "residual_adjacent_space": correlation(residual[:, :-1], residual[:, 1:], adjacency[None, :]),
        "residual_adjacent_time": correlation(residual[:-1], residual[1:], continuous[:, None]),
        "differential_increment_adjacent_space": correlation(differential[:, :-1], differential[:, 1:], adjacency[None, :]),
        "filter": modification,
        "independent_noise_assumption_verified": False,
        "interpretation": "real signal and noise are mixed; correlations of median residuals may be introduced by the filter itself",
    }


def episodes(active, observed, time_s):
    """Runs cannot cross missing observations or a temporal gap."""
    active, observed = np.asarray(active, bool), np.asarray(observed, bool)
    t = validate_axis(time_s, "time_s")
    if active.shape != t.shape or observed.shape != t.shape:
        raise ValueError("episode arrays must match time axis")
    result = []
    for s in segments(t, RULE_CONFIG["max_gap_s"]):
        live = active[s] & observed[s]
        edges = np.diff(np.r_[False, live, False].astype(np.int8))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            result.append((int(a+s.start), int(b+s.start)))
    return result


def weak_evidence(raw, valid, x_m, time_s, core_bounds):
    """Rule votes use raw contrast to flanking measurements, never detector Z."""
    raw, valid = np.asarray(raw), np.asarray(valid, bool)
    x, t = validate_axis(x_m, "x_m"), validate_axis(time_s, "time_s")
    if raw.shape != valid.shape or raw.shape != (len(t), len(x)):
        raise ValueError("field dimensions disagree")
    left, right = map(float, core_bounds)
    if not np.isfinite([left, right]).all() or right < left:
        raise ValueError("invalid core bounds")
    core = (x >= left) & (x <= right)
    distance = np.maximum(left-x, x-right)
    reference = (distance >= RULE_CONFIG["reference_inner_m"]) & (distance <= RULE_CONFIG["reference_outer_m"])
    finite = valid & np.isfinite(raw)
    live = np.where(finite, raw, np.nan)
    core_c = median(live[:, core], axis=1) if core.any() else np.full(len(t), np.nan)
    reference_c = median(live[:, reference], axis=1) if reference.any() else np.full(len(t), np.nan)
    core_fraction = finite[:, core].mean(axis=1) if core.any() else np.zeros(len(t))
    ref_count = finite[:, reference].sum(axis=1)
    observed = (core_fraction >= RULE_CONFIG["min_coverage"]) & (ref_count >= RULE_CONFIG["min_reference_columns"])
    contrast = np.where(observed, core_c-reference_c, np.nan)
    dt = sample_widths(t, max_gap=RULE_CONFIG["max_gap_s"])
    observed_fraction = float(np.sum(dt[observed])/np.sum(dt))
    hot = episodes(contrast >= RULE_CONFIG["contrast_c"], observed, t)
    cold = episodes(contrast <= -RULE_CONFIG["contrast_c"], observed, t)
    sustained_hot = [run for run in hot if run[1]-run[0] >= RULE_CONFIG["min_episode_rows"] and dt[run[0]:run[1]].sum() >= RULE_CONFIG["min_episode_s"]]
    sustained_cold = [run for run in cold if run[1]-run[0] >= RULE_CONFIG["min_episode_rows"] and dt[run[0]:run[1]].sum() >= RULE_CONFIG["min_episode_s"]]
    votes = {"sustained_hot": None, "sustained_cold": None, "repeated_contrast": None,
             "isolated_contrast": None, "low_local_contrast": None}
    shape_votes = []
    period, cv = None, None
    if observed_fraction >= RULE_CONFIG["min_coverage"]:
        if sustained_hot:
            votes["sustained_hot"] = 1
            shape_votes.append("local_heating")
        if sustained_cold:
            votes["sustained_cold"] = 1
            shape_votes.append("local_cooling")
        # Require repetitions inside ONE continuous, fully observed segment.
        for runs in (sustained_hot, sustained_cold):
            for segment in segments(t):
                contained = [run for run in runs if run[0] >= segment.start and run[1] <= segment.stop]
                if len(contained) >= RULE_CONFIG["min_repeats"] and observed[contained[0][0]:contained[-1][1]].all():
                    intervals = np.diff([t[a] for a, _ in contained])
                    local_cv = float(np.std(intervals)/np.mean(intervals))
                    if local_cv <= RULE_CONFIG["period_cv_max"]:
                        votes["repeated_contrast"] = 1
                        period, cv = float(np.median(intervals)), local_cv
                        shape_votes.append("repeated_local_change")
                        break
        if any(b-a <= 2 for a, b in hot+cold):
            votes["isolated_contrast"] = 1
            shape_votes.append("brief_local_change")
        if np.isfinite(contrast).any() and float(np.nanmax(np.abs(contrast))) < RULE_CONFIG["contrast_c"]:
            votes["low_local_contrast"] = 0
    active_votes = {v for v in votes.values() if v is not None}
    conflict = len(active_votes) > 1
    weak_label = next(iter(active_votes)) if len(active_votes) == 1 else None
    reasons = []
    if observed_fraction < RULE_CONFIG["min_coverage"]:
        reasons.append("insufficient_core_or_reference_coverage")
    if not active_votes:
        reasons.append("all_rules_abstain")
    if conflict:
        reasons.append("presence_rules_disagree")
    if sustained_hot and sustained_cold:
        reasons.append("both_heating_and_cooling")
    # Multiple shapes are permitted (e.g. periodic heating), not a forced class.
    evidence = {
        "rule_votes": votes, "weak_presence_label": weak_label,
        "label_scope": "observable local contrast to flanks; not all anomalies or safety",
        "shape_hypotheses": sorted(set(shape_votes)), "review_reasons": reasons,
        "rule_conflict": conflict, "observed_fraction": observed_fraction,
        "core_columns": int(core.sum()), "reference_columns": int(reference.sum()),
        "contrast_c": quantiles(contrast), "hot_runs_half_open": hot, "cold_runs_half_open": cold,
        "period_s": period, "period_cv": cv, "expert_label": None,
        "expert_confirmed": False, "physical_cause": None,
        "source": "raw_measurement_rules_v1", "probability_calibrated": False,
    }
    return evidence, {"core_c": core_c, "reference_c": reference_c, "contrast_c": contrast,
                      "observed": observed}


def diversity_key(row):
    """Fixed physical bins, NOT a fitted cluster or independently proven class."""
    e = row["weak_evidence"]
    q = e["contrast_c"] or {"q10": 0., "q90": 0.}
    amplitude = max(abs(q["q10"]), abs(q["q90"]))
    return (row["record"], row["origin"], tuple(e["shape_hypotheses"]),
            int(np.searchsorted([1., 3., 10.], amplitude)),
            int(np.searchsorted([3., 10., 40.], row["x_m_inclusive"][1]-row["x_m_inclusive"][0])),
            e["observed_fraction"] < .8)


def select_review(rows, per_record=4):
    """Balanced development shortlist; random controls retain their own slot."""
    selected = []
    for record in sorted({r["record"] for r in rows}):
        candidates = [r for r in rows if r["record"] == record and r["split_code"] in (0, 1)]
        # Disagreements are development questions, not additional expert truth.
        ranked = sorted(candidates, key=lambda r: (-len(r["weak_evidence"]["review_reasons"]), r["id"]))
        chosen, seen = [], set()
        controls = [r for r in ranked if r["origin"] == "random_review"]
        if controls:
            chosen.append(controls[0]); seen.add(diversity_key(controls[0]))
        for row in ranked:
            if len(chosen) >= per_record:
                break
            key = diversity_key(row)
            if row not in chosen and key not in seen:
                chosen.append(row); seen.add(key)
        for row in ranked:
            if len(chosen) >= per_record:
                break
            if row not in chosen:
                chosen.append(row)
        selected.extend(r["id"] for r in chosen)
    return selected
