"""Planted thermal scenes. The label is the generator, not a rule output."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from fiber.anomaly import extract_zones, spatial_z
from fiber.classify.rules import LABELS, classify_zones, profile_shift

KIND_TO_LABEL = {
    "background": "Background",
    "local_heating": "Local_Heating",
    "local_cooling": "Local_Cooling",
    "periodic": "Periodic_Thermal",
    "broad_shift": "Broad_Shift",
    "sensor_fault": "Sensor_Fault",
}
LABEL_INDEX = {name: index for index, name in enumerate(LABELS)}


def plant_scene(kind: str, seed: int = 0) -> dict:
    """One class on a quiet cable, plus a dead tail.

    Positions and amplitudes move with `seed`. Invalid cells are not labeled.
    """
    if kind not in KIND_TO_LABEL:
        raise KeyError(f"unknown planted kind {kind!r}")
    rng = np.random.default_rng(seed)
    t_count, x_count = 36, 80
    x_m = np.arange(x_count, dtype=np.float64)
    time_s = np.arange(t_count, dtype=np.float64) * 60.0
    noise = float(rng.uniform(0.12, 0.22))
    field = np.full((t_count, x_count), 22.0, dtype=np.float32)
    field += rng.normal(0.0, noise, size=field.shape).astype(np.float32)
    labels = np.zeros(field.shape, dtype=np.int8)
    valid = np.ones(field.shape, dtype=bool)
    valid[:, 72:] = False
    field[:, 72:] = -300.0
    class_id = LABEL_INDEX[KIND_TO_LABEL[kind]]

    if kind == "local_heating":
        _paint(field, labels, rng, class_id, sign=1.0, low=5.0, high=7.0)
    elif kind == "local_cooling":
        _paint(field, labels, rng, class_id, sign=-1.0, low=4.5, high=6.5)
    elif kind == "periodic":
        column = int(rng.integers(8, 40))
        amplitude = float(rng.uniform(4.5, 6.5))
        for row in (2, 14, 26):
            field[row:row + 4, column:column + 6] += amplitude
            labels[row:row + 4, column:column + 6] = class_id
    elif kind == "broad_shift":
        cut = int(rng.integers(28, 42))
        amplitude = float(rng.uniform(3.2, 4.5))
        field[:, :cut] += amplitude
        labels[:, :cut] = class_id
        labels[:, 72:] = 0
    elif kind == "sensor_fault":
        row = int(rng.integers(2, t_count - 2))
        column = int(rng.integers(4, 60))
        field[row, column] += float(rng.uniform(14.0, 20.0))
        labels[row, column] = class_id

    labels[~valid] = 0
    return {
        "kind": kind,
        "label": KIND_TO_LABEL[kind],
        "field": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "labels": labels,
        "noise_c": noise,
    }


def _paint(field, labels, rng, class_id: int, *, sign: float, low: float, high: float) -> None:
    column = int(rng.integers(6, 46))
    row = int(rng.integers(4, 16))
    width = int(rng.integers(5, 8))
    height = int(rng.integers(8, 12))
    field[row:row + height, column:column + width] += sign * float(rng.uniform(low, high))
    labels[row:row + height, column:column + width] = class_id


LONG_T = 96
LONG_X = 360
LONG_DEAD_FROM = 330
# Room-scale baseline, chosen before any witness scan of this generator.
# Peak is 1–3 °C. Periods sit on the scale of one crop (40) and a few crops.
PROFILE_AMP_C = (1.0, 3.0)
PROFILE_PERIODS = (40, 80, 160)
PROFILE_DRIFT_C = 0.3
# A long hot or cold streak, narrower than a 40-column crop.
# Width 5..16 keeps a gap of at least 40 columns to the next event.
PERSISTENT_WIDTH = (5, 17)
# Rightmost cell of one event and leftmost cell of the next stay at least 40
# columns apart, so a 40-wide crop cannot hold two events.
# Short events, max last columns: heat 30, cool 96, periodic 160, fault 213.
# Persistent heat/cool, max last columns: heat 41, cool 107.
# Min first columns of the next event: 84, 148, 206, 258.


def plant_long(seed: int = 0, *, profile: bool = False, persistent: bool = False) -> dict:
    """Quiet cable with five separated events and a dead tail.

    Jitter stays inside those bands. The broad segment is wider than a crop,
    so a window in its interior does not see the cut. `profile=True` adds a
    smooth room-scale baseline under the same events. Event positions do not
    move: the baseline uses its own seed stream. `persistent=True` paints
    heating and cooling on every time row, still only a few columns wide.
    """
    rng = np.random.default_rng(seed)
    x_m = np.arange(LONG_X, dtype=np.float64)
    time_s = np.arange(LONG_T, dtype=np.float64) * 60.0
    noise = float(rng.uniform(0.12, 0.22))
    field = np.full((LONG_T, LONG_X), 22.0, dtype=np.float32)
    field += rng.normal(0.0, noise, size=field.shape).astype(np.float32)
    labels = np.zeros(field.shape, dtype=np.int8)
    event_id = np.zeros(field.shape, dtype=np.int16)
    valid = np.ones(field.shape, dtype=bool)
    valid[:, LONG_DEAD_FROM:] = False
    field[:, LONG_DEAD_FROM:] = -300.0
    if profile:
        baseline = _room_profile(np.random.default_rng(seed + 10_000_019))
        field[:, :LONG_DEAD_FROM] += baseline[:, :LONG_DEAD_FROM]

    heat_x = 18 + int(rng.integers(0, 8))
    heat_t = 16 + int(rng.integers(0, 12))
    heat_amp = float(rng.uniform(5.0, 7.0))
    heat_id = LABEL_INDEX["Local_Heating"]
    if persistent:
        heat_w = int(rng.integers(PERSISTENT_WIDTH[0], PERSISTENT_WIDTH[1]))
        _stamp(field, labels, event_id, 0, LONG_T, heat_x, heat_x + heat_w, heat_amp, heat_id, 1)
    else:
        _stamp(field, labels, event_id, heat_t, heat_t + 12, heat_x, heat_x + 6, heat_amp, heat_id, 1)

    cool_x = 84 + int(rng.integers(0, 8))
    cool_t = 44 + int(rng.integers(0, 12))
    cool_amp = float(rng.uniform(4.5, 6.5))
    cool_id = LABEL_INDEX["Local_Cooling"]
    if persistent:
        cool_w = int(rng.integers(PERSISTENT_WIDTH[0], PERSISTENT_WIDTH[1]))
        _stamp(field, labels, event_id, 0, LONG_T, cool_x, cool_x + cool_w, -cool_amp, cool_id, 2)
    else:
        _stamp(field, labels, event_id, cool_t, cool_t + 12, cool_x, cool_x + 6, -cool_amp, cool_id, 2)

    period_x = 148 + int(rng.integers(0, 8))
    period_id = LABEL_INDEX["Periodic_Thermal"]
    period_amp = float(rng.uniform(4.5, 6.5))
    for row in (6, 36, 66):
        _stamp(field, labels, event_id, row, row + 4, period_x, period_x + 6,
               period_amp, period_id, 3)

    fault_x = 206 + int(rng.integers(0, 8))
    fault_t = 40 + int(rng.integers(0, 20))
    fault_id = LABEL_INDEX["Sensor_Fault"]
    field[fault_t, fault_x] += float(rng.uniform(14.0, 20.0))
    labels[fault_t, fault_x] = fault_id
    event_id[fault_t, fault_x] = 4

    broad_id = LABEL_INDEX["Broad_Shift"]
    field[:, 258:308] += float(rng.uniform(3.2, 4.5))
    labels[:, 258:308] = broad_id
    event_id[:, 258:308] = 5

    labels[~valid] = 0
    event_id[~valid] = 0
    events = (
        {"id": 1, "label": "Local_Heating"},
        {"id": 2, "label": "Local_Cooling"},
        {"id": 3, "label": "Periodic_Thermal"},
        {"id": 4, "label": "Sensor_Fault"},
        {"id": 5, "label": "Broad_Shift"},
    )
    return {
        "kind": "persistent" if persistent else "profile" if profile else "long",
        "field": field,
        "valid": valid,
        "x_m": x_m,
        "time_s": time_s,
        "labels": labels,
        "event_id": event_id,
        "events": events,
        "noise_c": noise,
    }


def _room_profile(rng: np.random.Generator) -> np.ndarray:
    """Smooth spatial baseline plus a small ramp in time. Shape (T, X)."""
    x = np.arange(LONG_X, dtype=np.float64)
    wave = np.zeros(LONG_X, dtype=np.float64)
    for period in PROFILE_PERIODS:
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        weight = float(rng.normal())
        wave += weight * np.sin(2.0 * np.pi * x / period + phase)
    wave -= wave.mean()
    peak = float(np.max(np.abs(wave)))
    amp = float(rng.uniform(PROFILE_AMP_C[0], PROFILE_AMP_C[1]))
    if peak > 0.0:
        wave *= amp / peak
    drift = float(rng.uniform(-PROFILE_DRIFT_C, PROFILE_DRIFT_C))
    ramp = np.linspace(-0.5, 0.5, LONG_T, dtype=np.float64) * drift
    return (wave[None, :] + ramp[:, None]).astype(np.float32)


def _stamp(field, labels, event_id, t0, t1, x0, x1, value, class_id, eid) -> None:
    field[t0:t1, x0:x1] += value
    labels[t0:t1, x0:x1] = class_id
    event_id[t0:t1, x0:x1] = eid


def _time_iso(time_s: np.ndarray) -> list[str]:
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [(origin + timedelta(seconds=float(second))).isoformat() for second in time_s]


def scene_features(scene: dict) -> np.ndarray:
    """Measurements of one scene. The planted name is not an input."""
    field = scene["field"]
    valid = scene["valid"]
    x_m = scene["x_m"]
    values = np.where(valid, field, np.nan).astype(np.float64)
    center = float(np.nanmedian(values))
    deviation = values - center
    finite = deviation[np.isfinite(deviation)]
    max_dev = float(np.max(finite)) if finite.size else 0.0
    min_dev = float(np.min(finite)) if finite.size else 0.0
    shift = profile_shift(field, valid, x_m)
    z, background, _scale = spatial_z(field, valid, x_m)
    contrast = np.where(valid, np.asarray(field, dtype=np.float64) - background, np.nan)
    zones = extract_zones(
        z,
        valid,
        x_m,
        _time_iso(scene["time_s"]),
        scene["time_s"],
        contrast_c=contrast,
    )
    hot = [zone for zone in zones if zone["polarity"] == "hot"]
    cold = [zone for zone in zones if zone["polarity"] == "cold"]
    peaks = [abs(float(zone.get("peak_contrast_c") or 0.0)) for zone in zones]
    widths = [float(zone["width_m"]) for zone in zones]
    return np.array(
        [
            max_dev,
            min_dev,
            float(shift["delta_c"]),
            float(shift["raw_delta_c"]),
            float(len(hot)),
            float(len(cold)),
            max(widths) if widths else 0.0,
            max(peaks) if peaks else 0.0,
            _period_cv(hot),
            _period_cv(cold),
        ],
        dtype=np.float64,
    )


def _period_cv(zones: list[dict]) -> float:
    if len(zones) < 3:
        return 1.0
    order = sorted(range(len(zones)), key=lambda index: 0.5 * (zones[index]["start_m"] + zones[index]["end_m"]))
    clusters: list[list[int]] = []
    for index in order:
        center = 0.5 * (zones[index]["start_m"] + zones[index]["end_m"])
        placed = False
        for cluster in clusters:
            anchor = float(np.median([0.5 * (zones[j]["start_m"] + zones[j]["end_m"]) for j in cluster]))
            if abs(center - anchor) <= 8.0:
                cluster.append(index)
                placed = True
                break
        if not placed:
            clusters.append([index])
    best = 1.0
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        times = np.array([_epoch(zones[index]) for index in cluster], dtype=np.float64)
        gaps = np.diff(np.sort(times))
        if gaps.size < 2 or np.any(gaps <= 0):
            continue
        best = min(best, float(np.std(gaps) / np.mean(gaps)))
    return best


def _epoch(zone: dict) -> float:
    return datetime.fromisoformat(zone["t_start"]).timestamp()


def rule_scene_label(scene: dict) -> str:
    """What the frozen shape rules say about a planted scene.

    This is a baseline. It is not the training target.
    """
    shift = profile_shift(scene["field"], scene["valid"], scene["x_m"])
    if shift["label"] == "Broad_Shift":
        return "Broad_Shift"
    z, background, _scale = spatial_z(scene["field"], scene["valid"], scene["x_m"])
    contrast = np.where(scene["valid"], np.asarray(scene["field"], dtype=np.float64) - background, np.nan)
    zones = extract_zones(
        z,
        scene["valid"],
        scene["x_m"],
        _time_iso(scene["time_s"]),
        scene["time_s"],
        contrast_c=contrast,
    )
    cable = float(scene["x_m"][scene["valid"].any(axis=0)][-1] - scene["x_m"][scene["valid"].any(axis=0)][0])
    labels = classify_zones(zones, cable)
    for name in ("Sensor_Fault", "Periodic_Thermal", "Broad_Shift", "Local_Heating", "Local_Cooling"):
        if name in labels:
            return name
    return "Background"


class SoftmaxSceneModel:
    """Multinomial logistic regression on `scene_features`."""

    def __init__(self) -> None:
        self.mu: np.ndarray | None = None
        self.sig: np.ndarray | None = None
        self.weight: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def fit(self, features: np.ndarray, class_index: np.ndarray, *, steps: int = 500, lr: float = 0.35) -> None:
        matrix = np.asarray(features, dtype=np.float64)
        target = np.asarray(class_index, dtype=np.int64)
        n_class = len(LABELS)
        self.mu = matrix.mean(axis=0)
        self.sig = np.maximum(matrix.std(axis=0), 1e-6)
        scaled = (matrix - self.mu) / self.sig
        weight = np.zeros((scaled.shape[1], n_class), dtype=np.float64)
        bias = np.zeros(n_class, dtype=np.float64)
        eye = np.eye(n_class, dtype=np.float64)[target]
        count = float(matrix.shape[0])
        for _ in range(steps):
            logits = scaled @ weight + bias
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            error = prob - eye
            weight -= lr * (scaled.T @ error / count + 1e-2 * weight)
            bias -= lr * error.mean(axis=0)
        self.weight = weight
        self.bias = bias

    def predict(self, features: np.ndarray) -> list[str]:
        if self.weight is None or self.mu is None or self.sig is None or self.bias is None:
            raise RuntimeError("model is not fit")
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        scaled = (matrix - self.mu) / self.sig
        logits = scaled @ self.weight + self.bias
        index = np.argmax(logits, axis=1)
        return [LABELS[int(item)] for item in index]
