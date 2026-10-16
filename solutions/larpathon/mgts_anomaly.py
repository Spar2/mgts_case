"""MGTS fiber thermal anomaly ensemble: CNN autoencoder + statistical residual."""

from __future__ import annotations

import ast
import json
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter, label, median_filter
from torch.utils.data import DataLoader, TensorDataset

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

LOCAL_HEATING = "local_heating (power cable / equipment)"
LOCAL_COOLING = "local_cooling (AC / ventilation)"
ZONE_COOLING = "zone_cooling (AC / occupancy)"
TRANSIENT = "transient"
EXTENDED_MILD = "extended_mild (possible residual cycle)"
OTHER = "other"
BROAD_HEATING = "broad_heating"
POSSIBLE_FIRE = "possible_fire"
BROAD_MILD = "broad_mild"
TYPE_HYPOTHESES = (
    LOCAL_HEATING,
    LOCAL_COOLING,
    ZONE_COOLING,
    TRANSIENT,
    EXTENDED_MILD,
    OTHER,
)
BROAD_HYPOTHESES = (BROAD_HEATING, POSSIBLE_FIRE, BROAD_MILD)
ALL_TYPE_LABELS = TYPE_HYPOTHESES + BROAD_HYPOTHESES

# Regional median window in grid steps. At the loader's 0.25 m target this is
# about 60 m: cable-scale and zone-scale departures stay, a smooth
# building-scale profile does not.
REGIONAL_MEDIAN_STEPS = 241
LOCAL_MEDIAN_STEPS = 15

# Slow per-column background, in time steps, not a fraction of the record.
# At the ~2 min step of 4.csv this is about 20 h. Half a window is 300 steps,
# so a 125-step (~250 min) heat and a 268-step heat stay a minority and are
# not subtracted away. A 30–60 min window would make those heats the median.
# The window is clamped to real samples: repeating the edge value would copy
# an early heat into the padding and erase it.
TEMPORAL_MEDIAN_STEPS = 601

# Score percentile is computed on each time segment of this many rows,
# not as one percentile of the whole file.
SEGMENT_ROWS = 120

# Default detector percentile. A calm ensemble stays here. A wide score
# bulk raises it, up to the cap, in proportion to the median absolute
# deviation from the file median. On the full ensemble, 4.csv stays at or
# below 0.03 and 1.csv is about 0.033, so the calm ceiling is 0.04.
# 6.csv and 5.csv reach the 0.05 cap. OpenCV does not read these values.
DETECTOR_PERCENTILE = 99.2
DETECTOR_CALM_MAD = 0.04
DETECTOR_NOISY_MAD = 0.05
DETECTOR_PERCENTILE_CAP = 99.8

# Local junk filter after connected components. The morphology gate still
# uses min_duration=3 so a 4–5 step flash is counted, then dropped here.
# Wide events are typed instead, so these two cuts do not apply to them.
MIN_DURATION_STEPS = 6
MIN_LOCAL_AMPLITUDE_C = 0.6

# Amplitude background: a few time steps by the regional spatial window.
# 241 steps (~60 m) still sees cable beside an 8–20 m section. A shorter
# spatial window sits inside that section and reports no contrast.
LOCAL_BG_TIME = 3
LOCAL_BG_SPACE = REGIONAL_MEDIAN_STEPS

# Below this share of spatial samples the event stays on the local rules.
# At or above it the event is broad and is typed, not discarded.
LOCAL_BROAD_FRACTION = 0.16

# local_heating stays under 15 m. The broad split already catches a spot
# wider than LOCAL_BROAD_FRACTION; this cap remains for a direct call that
# passes a fraction and still expects the local rules.
LOCAL_HEATING_MAX_FRACTION = 0.25

# local_cooling is a narrow cold spot. The cut is the neighbor-ring
# local_amplitude, not the 241-step amplitude used for local_heating.
# Strict: colder than -1.0 °C and longer than 15 min.
LOCAL_COOLING_MAX_AMPLITUDE_C = -1.0
LOCAL_COOLING_MIN_DURATION_MIN = 15.0

# zone_cooling is the wide cold spot. Not local: cable share at least the
# 0.16 cutoff, or wider than 20 m. Neighbor-ring amplitude colder than
# -0.8 °C and longer than 25 min (inside the 20–30 min band). A fire or a
# broad heat still wins.
ZONE_COOLING_MAX_AMPLITUDE_C = -0.8
ZONE_COOLING_MIN_EXTENT_M = 20.0
ZONE_COOLING_MIN_DURATION_MIN = 25.0

# periodic is a flag on local_heating and local_cooling, not a new type.
# The period has to sit between 15 min and 6 h, and at least two cycles
# have to fit in the series. The peak must be a line above the rest of
# the spectrum, not the top of a smooth slope.
PERIODIC_MIN_PERIOD_MIN = 15.0
PERIODIC_MAX_PERIOD_MIN = 360.0
PERIODIC_MIN_CYCLES = 2.0
PERIODIC_PEAK_TO_MEDIAN = 4.0
PERIODIC_PEAK_TO_NEIGHBOR = 2.0

# Broad rules. possible_fire needs the hold and either a fast rise or a hot peak.
# broad_heating is a high local amplitude whose rise is not faster than the
# fire rate. Anything else broad is broad_mild (weak drift, or a fast rise
# that did not hold).
FIRE_MAX_DT_PER_MIN = 0.4
FIRE_PEAK_TEMPERATURE_C = 48.0
FIRE_HOLD_MINUTES = 6.0
BROAD_HIGH_AMPLITUDE_C = 1.2

# Neighbor ring outside the event bbox, in grid steps. 16 is inside the
# requested ±10–20 points (~4 m at 0.25 m). Plus a few rows before the event.
NEIGHBOR_HALF_POINTS = 16
NEIGHBOR_PRE_STEPS = 5

# Rank score in [0, 1]. Contrast saturates around the 1.2 °C local-heating band.
# Weights sum to 1. Without an autoencoder map the first two are renormalized.
CONFIDENCE_CONTRAST_SCALE_C = 1.2
CONFIDENCE_WEIGHT_CONTRAST = 0.50
CONFIDENCE_WEIGHT_STABILITY = 0.30
CONFIDENCE_WEIGHT_AGREEMENT = 0.20

# Light merge: gap is strict less-than. 8 samples is about 2 m, so the
# 94 m and 106 m heats on 4.csv (about 6 m apart) stay separate. 12 steps
# is about 24 min at the 2 min sample of that file.
MERGE_MAX_GAP_POINTS = 8
MERGE_MAX_GAP_STEPS = 12


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_points(value) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    return list(parsed)


def _resample_row(points, target_dists: np.ndarray) -> np.ndarray:
    parsed = np.asarray(points, dtype=np.float64)
    if parsed.ndim != 2 or parsed.shape[1] < 2 or parsed.shape[0] == 0:
        return np.full(target_dists.shape, np.nan, dtype=np.float32)
    distance = parsed[:, 0]
    temperature = parsed[:, 1]
    finite = np.isfinite(distance) & np.isfinite(temperature)
    distance = distance[finite]
    temperature = temperature[finite]
    if distance.size == 0:
        return np.full(target_dists.shape, np.nan, dtype=np.float32)
    order = np.argsort(distance, kind="mergesort")
    distance = distance[order]
    temperature = temperature[order]
    unique_d, inverse = np.unique(distance, return_inverse=True)
    if unique_d.size != distance.size:
        sums = np.zeros(unique_d.shape, dtype=np.float64)
        counts = np.zeros(unique_d.shape, dtype=np.float64)
        np.add.at(sums, inverse, temperature)
        np.add.at(counts, inverse, 1.0)
        distance = unique_d
        temperature = sums / counts
    if distance.size == 1:
        return np.full(target_dists.shape, float(temperature[0]), dtype=np.float32)
    interpolator = interp1d(
        distance,
        temperature,
        kind="linear",
        bounds_error=False,
        fill_value=(float(temperature[0]), float(temperature[-1])),
    )
    return interpolator(target_dists).astype(np.float32)


def _fill_non_finite(matrix: np.ndarray) -> np.ndarray:
    if np.isfinite(matrix).all():
        return matrix
    filled = matrix.copy()
    col_median = np.nanmedian(filled, axis=0)
    bad = ~np.isfinite(filled)
    if np.isfinite(col_median).any():
        col_median = np.where(np.isfinite(col_median), col_median, np.nanmedian(col_median))
        filled[bad] = np.take(col_median, np.where(bad)[1])
    if not np.isfinite(filled).all():
        fallback = np.nanmedian(filled)
        if not np.isfinite(fallback):
            fallback = 0.0
        filled[~np.isfinite(filled)] = fallback
    return filled


def load_dts_to_matrix(
    csv_path: str,
    target_step: float = 0.25,
    max_spatial_points: int = 2500,
    max_rows: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Load one DTS CSV into a finite time x distance temperature matrix.

    `points_size` is optional and ignored. `points` holds [distance_m, temperature_C] pairs.
    Rows are returned in timestamp order on one linear spatial grid.
    """
    frame = pd.read_csv(csv_path, nrows=max_rows)
    if "timestamp" not in frame.columns or "points" not in frame.columns:
        missing = {"timestamp", "points"} - set(frame.columns)
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    parsed_time = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = frame.assign(_ts=parsed_time).sort_values(
        ["_ts", "timestamp"], kind="mergesort", na_position="last"
    )
    frame = frame.reset_index(drop=True)
    parsed_rows = [parse_points(value) for value in frame["points"]]
    max_dist = 0.0
    for points in parsed_rows:
        if not points:
            continue
        distances = np.asarray(points, dtype=np.float64)
        if distances.ndim != 2 or distances.shape[1] < 1:
            continue
        finite = distances[:, 0]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            max_dist = max(max_dist, float(finite.max()))
    if max_dist <= 0.0:
        max_dist = float(target_step)
    n_points = int(max_dist / target_step) + 1
    n_points = min(max(n_points, 2), int(max_spatial_points))
    target_dists = np.linspace(0.0, max_dist, n_points, dtype=np.float64)
    matrix = np.stack([_resample_row(points, target_dists) for points in parsed_rows]).astype(np.float32)
    matrix = _fill_non_finite(matrix)
    timestamps = [str(value) for value in frame["timestamp"].tolist()]
    return matrix, target_dists, timestamps


def normalize_matrix(matrix: np.ndarray, method: str = "robust") -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if method == "robust":
        center = np.median(values)
        mad = np.median(np.abs(values - center)) + 1e-6
        scaled = (values - center) / (1.4826 * mad)
    else:
        scaled = (values - values.mean()) / (values.std() + 1e-6)
    return scaled.astype(np.float32)


def denoise_matrix(
    matrix: np.ndarray,
    sigma_time: float = 1.5,
    sigma_space: float = 2.5,
    detrend: bool = True,
) -> np.ndarray:
    """Gaussian noise clean, then remove slow fiber-wide drift.

    The drift term is a low-pass of the per-time spatial median. Subtracting that
    common mode keeps a local hot cable and a wide cool zone. The per-column
    temporal median that follows is a separate, much longer window.
    """
    denoised = gaussian_filter(np.asarray(matrix, dtype=np.float64), sigma=(sigma_time, sigma_space))
    if detrend and denoised.shape[0] > 1:
        profile = np.median(denoised, axis=1)
        sigma = max(5.0, denoised.shape[0] / 20.0)
        drift = gaussian_filter(profile, sigma=sigma)
        denoised = denoised - drift[:, None] + np.median(denoised)
    return denoised.astype(np.float32)


def temporal_median_window(n_times: int, window_steps: int = TEMPORAL_MEDIAN_STEPS) -> int:
    """Odd per-column window of a fixed step count, capped by the record."""
    if n_times <= 1:
        return 1
    window = min(max(int(window_steps), 1), int(n_times))
    if window % 2 == 0:
        window = window - 1 if window > 1 else 1
    return max(window, 1)


def _column_median(block: np.ndarray) -> np.ndarray:
    """Median down time. The window length is odd, so this is the middle value."""
    mid = block.shape[0] // 2
    return np.partition(block, mid, axis=0)[mid]


def slow_temporal_background(
    matrix: np.ndarray,
    window_steps: int = TEMPORAL_MEDIAN_STEPS,
) -> np.ndarray:
    """Per-column rolling median with the window shifted to stay inside the record.

    Edge rows reuse the first or last full window. They do not repeat the edge
    sample, which would turn a heat at the start of the file into the median.
    """
    values = np.asarray(matrix, dtype=np.float64)
    n_times, n_space = values.shape
    window = temporal_median_window(n_times, window_steps)
    if window <= 1:
        return np.zeros((n_times, n_space), dtype=np.float64)
    if window >= n_times:
        return np.repeat(_column_median(values)[None, :], n_times, axis=0)
    n_windows = n_times - window + 1
    medians = np.empty((n_windows, n_space), dtype=np.float64)
    for start in range(n_windows):
        medians[start] = _column_median(values[start : start + window])
    half = window // 2
    index = np.clip(np.arange(n_times) - half, 0, n_windows - 1)
    return medians[index]


def subtract_slow_temporal_median(
    matrix: np.ndarray,
    window_steps: int = TEMPORAL_MEDIAN_STEPS,
) -> np.ndarray:
    """Temperature minus the slow per-column median, in the same units."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        return values.copy()
    return values - slow_temporal_background(values, window_steps=window_steps)


def statistical_anomaly_score(
    matrix: np.ndarray,
    residual_w: float = 0.45,
    spatial_w: float = 0.35,
    gradient_w: float = 0.20,
    background: bool = True,
    window_steps: int = TEMPORAL_MEDIAN_STEPS,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if background:
        values = subtract_slow_temporal_median(values, window_steps=window_steps)
    residual = np.abs(values - gaussian_filter(values, sigma=(2.0, 3.0)))
    local = np.abs(values - median_filter(values, size=(1, _odd_limit(LOCAL_MEDIAN_STEPS, values.shape[1]))))
    regional_steps = _odd_limit(REGIONAL_MEDIAN_STEPS, values.shape[1])
    regional = np.abs(values - median_filter(values, size=(1, regional_steps)))
    spatial_dev = np.maximum(local, regional)
    temporal_gradient = np.abs(ndimage.sobel(values, axis=0))
    score = residual_w * residual + spatial_w * spatial_dev + gradient_w * temporal_gradient
    return score.astype(np.float32)


def _odd_limit(requested: int, limit: int) -> int:
    window = min(int(requested), int(limit))
    if window < 1:
        return 1
    if window % 2 == 0:
        window -= 1
    return max(window, 1)


def _norm_branch(score: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(score, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.percentile(values, 99) - np.percentile(values, 1) + 1e-6
    normalized = (values - np.median(values)) / scale
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def ensemble_score(
    matrix: np.ndarray,
    ae_score: Optional[np.ndarray] = None,
    stat_w: float = 0.55,
    ae_w: float = 0.45,
    background: bool = True,
    window_steps: int = TEMPORAL_MEDIAN_STEPS,
    stat_score: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weighted sum of normalized statistical and autoencoder branches.

    `stat_score`, when given, is used as-is and the background is not subtracted again.
    """
    if stat_score is None:
        stat_values = statistical_anomaly_score(
            matrix, background=background, window_steps=window_steps
        )
    else:
        stat_values = stat_score
    combined = stat_w * _norm_branch(stat_values)
    if ae_score is not None:
        combined = combined + ae_w * _norm_branch(ae_score)
    return np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class ConvAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        recon = self.decoder(self.encoder(batch))
        if recon.shape[-2] < batch.shape[-2] or recon.shape[-1] < batch.shape[-1]:
            recon = nn.functional.pad(
                recon,
                (0, batch.shape[-1] - recon.shape[-1], 0, batch.shape[-2] - recon.shape[-2]),
            )
        return recon[..., : batch.shape[-2], : batch.shape[-1]]


def _axis_starts(length: int, patch: int, stride: int) -> list[int]:
    if length < patch:
        return []
    starts = list(range(0, length - patch + 1, stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def extract_patches(
    matrix: np.ndarray,
    patch_t: int = 32,
    patch_s: int = 64,
    stride_t: int = 16,
    stride_s: int = 32,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    time_starts = _axis_starts(values.shape[0], patch_t, stride_t)
    space_starts = _axis_starts(values.shape[1], patch_s, stride_s)
    patches = [
        values[t : t + patch_t, s : s + patch_s]
        for t in time_starts
        for s in space_starts
    ]
    if not patches:
        return np.zeros((0, 1, patch_t, patch_s), dtype=np.float32)
    return np.stack(patches)[:, None, :, :].astype(np.float32)


def train_autoencoder(
    patches: np.ndarray,
    epochs: int = 6,
    batch_size: int = 32,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> ConvAutoencoder:
    set_seed(seed)
    if device is None:
        device = torch.device("cpu")
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(patches, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model.train()
    for epoch in range(epochs):
        total = 0.0
        batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = loss_fn(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        mean_loss = total / max(batches, 1)
        print(f"autoencoder epoch {epoch + 1}/{epochs} loss={mean_loss:.6f}", flush=True)
    model.eval()
    return model


def ae_error_map(
    matrix: np.ndarray,
    model: ConvAutoencoder,
    patch_t: int = 32,
    patch_s: int = 64,
    stride_t: int = 8,
    stride_s: int = 16,
    batch_size: int = 64,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    time_len, space_len = values.shape
    coords = [
        (t, s)
        for t in _axis_starts(time_len, patch_t, stride_t)
        for s in _axis_starts(space_len, patch_s, stride_s)
    ]
    error_sum = np.zeros((time_len, space_len), dtype=np.float64)
    count = np.zeros((time_len, space_len), dtype=np.float64)
    if not coords:
        return error_sum.astype(np.float32)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            batch_coords = coords[start : start + batch_size]
            patches = np.stack(
                [values[t : t + patch_t, s : s + patch_s] for t, s in batch_coords]
            )[:, None]
            recon = model(torch.tensor(patches, dtype=torch.float32, device=device)).cpu().numpy()[:, 0]
            error = np.abs(patches[:, 0] - recon)
            for (t, s), patch_error in zip(batch_coords, error):
                error_sum[t : t + patch_t, s : s + patch_s] += patch_error
                count[t : t + patch_t, s : s + patch_s] += 1.0
    return (error_sum / np.maximum(count, 1.0)).astype(np.float32)


def compute_score_parts(
    matrix: np.ndarray,
    use_ae: bool = True,
    seed: int = 0,
    epochs: int = 6,
    device: str = "cpu",
    window_steps: int = TEMPORAL_MEDIAN_STEPS,
) -> dict:
    """Ensemble score plus the statistical and autoencoder maps that built it.

    The autoencoder map is None when that branch is off or the field is too
    small for patches. Both maps, when present, match the matrix shape.
    """
    prepared = np.asarray(
        subtract_slow_temporal_median(matrix, window_steps=window_steps),
        dtype=np.float32,
    )
    stat_score = statistical_anomaly_score(prepared, background=False, window_steps=window_steps)
    ae_score = None
    if use_ae and prepared.shape[0] >= 40 and prepared.shape[1] >= 64:
        normalized = normalize_matrix(prepared)
        patches = extract_patches(normalized)
        if patches.shape[0] > 10:
            print(f"training autoencoder patches={patches.shape[0]}", flush=True)
            torch_device = torch.device(device)
            model = train_autoencoder(
                patches, epochs=epochs, seed=seed, device=torch_device
            )
            print("building autoencoder error map", flush=True)
            ae_score = ae_error_map(normalized, model)
    score = ensemble_score(
        prepared,
        ae_score,
        background=False,
        window_steps=window_steps,
        stat_score=stat_score,
    )
    return {"ensemble": score, "statistical": stat_score, "autoencoder": ae_score}


def compute_ensemble(
    matrix: np.ndarray,
    use_ae: bool = True,
    seed: int = 0,
    epochs: int = 6,
    device: str = "cpu",
    window_steps: int = TEMPORAL_MEDIAN_STEPS,
) -> np.ndarray:
    """Score one matrix after the slow temporal median.

    The same background-removed field is the residual input and the autoencoder
    input. The autoencoder trains when that field can host patches.
    `window_steps` is an absolute length. It does not grow with the record.
    """
    return compute_score_parts(
        matrix,
        use_ae=use_ae,
        seed=seed,
        epochs=epochs,
        device=device,
        window_steps=window_steps,
    )["ensemble"]


def median_timestamp_step_minutes(timestamps: Sequence[str]) -> Optional[float]:
    parsed = pd.to_datetime(pd.Series(list(timestamps)), errors="coerce", utc=True).dropna()
    if len(parsed) < 2:
        return None
    parsed = parsed.sort_values()
    deltas = parsed.diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return None
    step = float(np.median(deltas.to_numpy()) / 60.0)
    if not np.isfinite(step) or step <= 0:
        return None
    return step


def _event_shape_measures(
    temperatures: np.ndarray,
    mask: np.ndarray,
    t0: int,
    t1: int,
    s0: int,
    s1: int,
    time_step_min: float,
) -> dict:
    """Rise, peak, and whether the mask width changes, from the original field.

    The rise is the largest step that lands inside the mask. The step into the
    event counts. A later re-entry after a hole does not, so a short gap
    between two pieces of one heat is not read as a sudden fire. Width is the
    per-row sample span of the mask.
    """
    start = t0 - 1 if t0 > 0 else t0
    box = np.asarray(temperatures[start : t1 + 1, s0 : s1 + 1], dtype=np.float64)
    step = float(time_step_min)
    if box.shape[0] >= 2 and np.isfinite(step) and step > 0:
        rises = np.diff(box, axis=0) / step
        mask_box = np.zeros(box.shape, dtype=bool)
        offset = t0 - start
        mask_box[offset:, :] = np.asarray(mask[t0 : t1 + 1, s0 : s1 + 1], dtype=bool)
        later_in = mask_box[1:]
        earlier_in = mask_box[:-1]
        earlier_time = np.arange(start, t1)[:, None]
        count = later_in & (earlier_in | (earlier_time < t0))
        chosen = rises[count]
        finite = chosen[np.isfinite(chosen)]
        max_dt = float(finite.max()) if finite.size else 0.0
    else:
        max_dt = 0.0
    on_event = np.asarray(temperatures, dtype=np.float64)[mask]
    finite_temps = on_event[np.isfinite(on_event)]
    peak = float(finite_temps.max()) if finite_temps.size else 0.0
    widths: list[int] = []
    for row in range(t0, t1 + 1):
        cols = np.flatnonzero(mask[row])
        if cols.size:
            widths.append(int(cols[-1] - cols[0] + 1))
    width_changes = len(widths) >= 2 and (max(widths) - min(widths)) > 1
    return {
        "max_dt_per_min": max_dt,
        "peak_temperature_c": peak,
        "width_changes": bool(width_changes),
    }


def _meets_zone_cooling(event: dict) -> bool:
    """Wide, long, and colder than the neighbor ring. Not a type by itself."""
    local_amplitude = event.get("local_amplitude")
    if local_amplitude is None:
        return False
    amplitude = float(local_amplitude)
    if not np.isfinite(amplitude) or amplitude >= ZONE_COOLING_MAX_AMPLITUDE_C:
        return False
    duration_min = float(event.get("duration_min") or 0.0)
    if duration_min <= ZONE_COOLING_MIN_DURATION_MIN:
        return False
    fraction = event.get("spatial_fraction")
    wide_share = fraction is not None and float(fraction) >= LOCAL_BROAD_FRACTION
    wide_meters = float(event.get("spatial_extent_m") or 0.0) > ZONE_COOLING_MIN_EXTENT_M
    return wide_share or wide_meters


def classify_broad_event(
    event: dict,
    fire_rate: float = FIRE_MAX_DT_PER_MIN,
    fire_peak: float = FIRE_PEAK_TEMPERATURE_C,
    fire_hold_min: float = FIRE_HOLD_MINUTES,
    high_amplitude: float = BROAD_HIGH_AMPLITUDE_C,
) -> str:
    """Fire, broad heat, zone cooling, or a mild wide drift."""
    rate = float(event.get("max_dt_per_min") or 0.0)
    peak = float(event.get("peak_temperature_c") or 0.0)
    amplitude = float(event["amplitude"])
    duration_min = float(event["duration_min"])
    if duration_min >= fire_hold_min and (rate > fire_rate or peak > fire_peak):
        return POSSIBLE_FIRE
    if amplitude >= high_amplitude and rate <= fire_rate:
        return BROAD_HEATING
    if _meets_zone_cooling(event):
        return ZONE_COOLING
    return BROAD_MILD


def classify_event_type(event: dict) -> str:
    fraction = event.get("spatial_fraction")
    if fraction is not None and float(fraction) >= LOCAL_BROAD_FRACTION:
        return classify_broad_event(event)
    spatial_m = float(event["spatial_extent_m"])
    duration_min = float(event["duration_min"])
    amplitude = float(event["amplitude"])
    local_span = fraction is None or float(fraction) <= LOCAL_HEATING_MAX_FRACTION
    if amplitude > 1.2 and spatial_m < 15 and duration_min > 20 and local_span:
        return LOCAL_HEATING
    if _meets_zone_cooling(event):
        return ZONE_COOLING
    local_amplitude = event.get("local_amplitude")
    if (
        local_amplitude is not None
        and float(local_amplitude) < LOCAL_COOLING_MAX_AMPLITUDE_C
        and duration_min > LOCAL_COOLING_MIN_DURATION_MIN
        and spatial_m <= ZONE_COOLING_MIN_EXTENT_M
    ):
        return LOCAL_COOLING
    if abs(amplitude) > 0.8 and duration_min < 15:
        return TRANSIENT
    if spatial_m > 30 and abs(amplitude) < 1.0:
        return EXTENDED_MILD
    return OTHER


def _series_is_periodic(series: np.ndarray, time_step_min: float) -> bool:
    """True when one FFT line in the period window stands above the spectrum."""
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    step = float(time_step_min)
    if n < 8 or not np.isfinite(step) or step <= 0:
        return False
    time_index = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(time_index, values, 1)
    centered = values - (slope * time_index + intercept)
    if float(np.std(centered)) < 1e-6:
        return False
    window = np.hanning(n)
    spectrum = np.fft.rfft(centered * window)
    power = np.abs(spectrum) ** 2
    if power.size < 3:
        return False
    freqs = np.fft.rfftfreq(n, d=step)
    duration_min = n * step
    best_power = 0.0
    best_k = 0
    for k in range(1, power.size):
        freq = float(freqs[k])
        if freq <= 0:
            continue
        period = 1.0 / freq
        if period < PERIODIC_MIN_PERIOD_MIN or period > PERIODIC_MAX_PERIOD_MIN:
            continue
        if period * PERIODIC_MIN_CYCLES > duration_min + 1e-6:
            continue
        left = float(power[k - 1])
        right = float(power[k + 1]) if k + 1 < power.size else 0.0
        height = float(power[k])
        neighbor = max(left, right)
        if height <= left or height <= right or neighbor <= 0:
            continue
        if height < PERIODIC_PEAK_TO_NEIGHBOR * neighbor or height <= best_power:
            continue
        best_power = height
        best_k = k
    if best_k == 0:
        return False
    rest = [float(power[k]) for k in range(1, power.size) if k != best_k]
    background = float(np.median(rest)) if rest else 0.0
    if background <= 1e-18:
        total = float(np.sum(power[1:]))
        return total > 0 and best_power >= 0.5 * total
    return best_power >= PERIODIC_PEAK_TO_MEDIAN * background


def event_is_periodic(
    temperatures: np.ndarray,
    mask: np.ndarray,
    t0: int,
    t1: int,
    s0: int,
    s1: int,
    time_step_min: float,
) -> bool:
    """FFT of the in-event mean temperature, or of the row-presence mask."""
    slab = np.asarray(temperatures[t0 : t1 + 1, s0 : s1 + 1], dtype=np.float64)
    finite = np.isfinite(slab)
    filled = np.where(finite, slab, np.nan)
    with np.errstate(all="ignore"):
        means = np.nanmean(filled, axis=1)
    good = np.isfinite(means)
    if not good.any():
        means = np.zeros(filled.shape[0], dtype=np.float64)
    else:
        means = np.where(good, means, float(np.median(means[good])))
    present = np.asarray(mask[t0 : t1 + 1, s0 : s1 + 1], dtype=bool).any(axis=1)
    present = present.astype(np.float64)
    return _series_is_periodic(means, time_step_min) or _series_is_periodic(present, time_step_min)


def local_spacetime_background(temperatures: np.ndarray) -> np.ndarray:
    """Short-time, wide-space median. A long heat is compared with its neighbors."""
    values = np.asarray(temperatures, dtype=np.float64)
    time_window = _odd_limit(LOCAL_BG_TIME, values.shape[0])
    space_window = _odd_limit(LOCAL_BG_SPACE, values.shape[1])
    return median_filter(values, size=(time_window, space_window), mode="nearest")


def neighbor_local_amplitude(
    temperatures: np.ndarray,
    mask: np.ndarray,
    t0: int,
    t1: int,
    s0: int,
    s1: int,
    half_points: int = NEIGHBOR_HALF_POINTS,
    pre_steps: int = NEIGHBOR_PRE_STEPS,
) -> float:
    """Mean on the event minus the median just outside it.

    The ring is `half_points` samples to each side of the bbox, on the event's
    rows, plus up to `pre_steps` rows before the event on the event's columns.
    A hot mass far along the cable does not enter this median.
    """
    values = np.asarray(temperatures, dtype=np.float64)
    _n_time, n_space = values.shape
    inside = values[mask]
    inside = inside[np.isfinite(inside)]
    if inside.size == 0:
        return 0.0
    samples: list[np.ndarray] = []
    left0 = max(0, int(s0) - int(half_points))
    if left0 < s0:
        block = values[t0 : t1 + 1, left0:s0]
        block = block[np.isfinite(block)]
        if block.size:
            samples.append(block.ravel())
    right1 = min(n_space, int(s1) + 1 + int(half_points))
    if s1 + 1 < right1:
        block = values[t0 : t1 + 1, s1 + 1 : right1]
        block = block[np.isfinite(block)]
        if block.size:
            samples.append(block.ravel())
    pre0 = max(0, int(t0) - int(pre_steps))
    if pre0 < t0:
        block = values[pre0:t0, s0 : s1 + 1]
        block = block[np.isfinite(block)]
        if block.size:
            samples.append(block.ravel())
    if not samples:
        return 0.0
    reference = float(np.median(np.concatenate(samples)))
    return float(inside.mean() - reference)


def _score_agreement(
    stat_score: Optional[np.ndarray],
    ae_score: Optional[np.ndarray],
    mask: np.ndarray,
) -> Optional[float]:
    """Map correlation of the two score branches on the mask into 0..1."""
    if stat_score is None or ae_score is None:
        return None
    stat = np.asarray(stat_score, dtype=np.float64)
    ae = np.asarray(ae_score, dtype=np.float64)
    if stat.shape != mask.shape or ae.shape != mask.shape:
        return None
    both = mask & np.isfinite(stat) & np.isfinite(ae)
    if int(both.sum()) < 2:
        return None
    left = stat[both]
    right = ae[both]
    if float(np.std(left)) < 1e-8 or float(np.std(right)) < 1e-8:
        return 0.5
    corr = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(corr):
        return 0.5
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def event_confidence(
    local_amplitude: float,
    mask: np.ndarray,
    t0: int,
    t1: int,
    stat_score: Optional[np.ndarray] = None,
    ae_score: Optional[np.ndarray] = None,
    contrast_scale: float = CONFIDENCE_CONTRAST_SCALE_C,
) -> float:
    """Rank score in [0, 1]: neighbor contrast, filled time, branch agreement.

    Agreement is omitted, and the other two weights renormalized, when either
    score map is missing. This is not a probability from labels.
    """
    amp = abs(float(local_amplitude))
    scale = float(contrast_scale) if contrast_scale and contrast_scale > 0 else 1.0
    contrast = amp / (amp + scale)
    rows = np.asarray(mask[t0 : t1 + 1]).any(axis=1)
    stability = float(rows.mean()) if rows.size else 0.0
    agreement = _score_agreement(stat_score, ae_score, mask)
    contrast_w = CONFIDENCE_WEIGHT_CONTRAST
    stability_w = CONFIDENCE_WEIGHT_STABILITY
    agreement_w = CONFIDENCE_WEIGHT_AGREEMENT
    if agreement is None:
        total = contrast_w + stability_w
        value = (contrast_w * contrast + stability_w * stability) / total if total else 0.0
    else:
        value = contrast_w * contrast + stability_w * stability + agreement_w * agreement
    return float(np.clip(value, 0.0, 1.0))


def outside_cable_amplitude(
    temperatures: np.ndarray,
    mask: np.ndarray,
    t0: int,
    t1: int,
) -> Optional[float]:
    """Mean on the event minus the mean of the same rows outside the mask.

    A broad block on a full cable is wider than the 241-step local median, so
    that median sits inside the heat and reports no contrast. The cable beside
    the event is the reference. The local window is not widened.
    """
    rows = np.asarray(temperatures[t0 : t1 + 1], dtype=np.float64)
    row_mask = np.asarray(mask[t0 : t1 + 1], dtype=bool)
    inside = rows[row_mask]
    outside = rows[~row_mask]
    inside = inside[np.isfinite(inside)]
    outside = outside[np.isfinite(outside)]
    if inside.size == 0 or outside.size == 0:
        return None
    return float(inside.mean() - outside.mean())


def detector_percentile_for_score(score: np.ndarray, percentile: float = DETECTOR_PERCENTILE) -> float:
    """Percentile for the segment mask, raised when the score bulk is wide.

    The spread is the median absolute deviation from the file median.
    At or below DETECTOR_CALM_MAD the requested default stays. At or above
    DETECTOR_NOISY_MAD the cut is the cap. Between them it follows the MAD.
    A percentile other than the default 99.2 is returned unchanged.
    """
    requested = float(percentile)
    if abs(requested - DETECTOR_PERCENTILE) > 1e-9:
        return requested
    values = np.asarray(score, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return requested
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if not np.isfinite(mad) or mad <= DETECTOR_CALM_MAD:
        return requested
    if mad >= DETECTOR_NOISY_MAD:
        return float(DETECTOR_PERCENTILE_CAP)
    frac = (mad - DETECTOR_CALM_MAD) / (DETECTOR_NOISY_MAD - DETECTOR_CALM_MAD)
    return requested + frac * (float(DETECTOR_PERCENTILE_CAP) - requested)


def segment_anomaly_mask(
    score: np.ndarray,
    percentile: float,
    segment_rows: int = SEGMENT_ROWS,
) -> np.ndarray:
    """True where the score is in the upper tail of its own time segment."""
    values = np.asarray(score, dtype=np.float64)
    mask = np.zeros(values.shape, dtype=bool)
    if values.size == 0:
        return mask
    rows = max(int(segment_rows), 1)
    for start in range(0, values.shape[0], rows):
        end = min(values.shape[0], start + rows)
        chunk = values[start:end]
        finite = np.isfinite(chunk)
        if not finite.any():
            continue
        finite_values = chunk[finite]
        median = float(np.median(finite_values))
        threshold = float(np.percentile(finite_values, percentile))
        if not np.isfinite(threshold):
            continue
        if threshold > median:
            selected = finite & (chunk >= threshold)
        else:
            # The percentile still sits on a flat background. Keep only the
            # strict upper tail so a rare plateau is not lost and a flat
            # segment does not light up.
            selected = finite & (chunk > median)
        mask[start:end] = selected
    return mask


class FilteredEvents(list):
    """Kept events plus how many local candidates the junk filter removed."""

    def __init__(self, events: Optional[Iterable[dict]] = None, tally: Optional[dict] = None) -> None:
        super().__init__(events or [])
        self.tally = {
            "duration": 0,
            "amplitude": 0,
            "confidence": 0,
        }
        if tally:
            self.tally.update(tally)


def filter_junk_events(
    events: Sequence[dict],
    min_duration_steps: int = MIN_DURATION_STEPS,
    min_local_amplitude_c: float = MIN_LOCAL_AMPLITUDE_C,
) -> tuple[list[dict], dict]:
    """Drop short or weak local events. A broad label is kept.

    A local candidate that fails both checks is counted once, duration first.
    Wide events are already typed, so dropping them here would hide a fire,
    a broad heat, or a weak drift. `zone_cooling` uses the neighbor-ring
    amplitude. The 0.6 °C cut is the 3×241 median, which sits inside a wide
    cold block and would drop that zone.
    """
    tally = {"duration": 0, "amplitude": 0, "confidence": 0}
    kept: list[dict] = []
    for event in events:
        if event.get("type_hypothesis") in BROAD_HYPOTHESES or event.get("type_hypothesis") == ZONE_COOLING:
            kept.append(event)
            continue
        if int(event["duration_steps"]) < min_duration_steps:
            tally["duration"] += 1
            continue
        if abs(float(event["amplitude"])) < min_local_amplitude_c:
            tally["amplitude"] += 1
            continue
        kept.append(event)
    return kept, tally


def format_filter_line(tally: dict) -> str:
    duration = int(tally.get("duration", 0))
    amplitude = int(tally.get("amplitude", 0))
    confidence = int(tally.get("confidence", 0))
    total = duration + amplitude + confidence
    return (
        f"Filtered {total} events: duration={duration}, "
        f"amplitude={amplitude}, confidence={confidence}."
    )


def category_counts(events: Sequence[dict]) -> dict[str, int]:
    """How many kept events fell into each local and broad label."""
    counts = {label: 0 for label in ALL_TYPE_LABELS}
    for event in events:
        label = str(event.get("type_hypothesis", ""))
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def format_category_line(events: Sequence[dict]) -> str:
    counts = category_counts(events)
    parts = [f"{label}={counts[label]}" for label in ALL_TYPE_LABELS]
    return "Categories: " + ", ".join(parts) + "."


# Optional second look at the ensemble score. These cuts are not the
# detector: extract_events keeps its own percentile, close kernel, and labels.
OPENCV_UINT8_THRESHOLD = 200
OPENCV_MIN_PIXELS = 15
OPENCV_CLOSE_SIZE = (3, 3)
# A 16-sample ring stays inside the 6 m plateaus on calm 4.csv, so the
# interior never reaches 1.2 °C. 58 samples reaches past that plateau.
# Noisy files keep the 16-sample ring.
OPENCV_CALM_HALF_POINTS = 58
OPENCV_CALM_FRAGMENT_MINUTES = 90.0
OPENCV_CALM_LONG_MINUTES = 200.0
# Same-sign OpenCV boxes this close on both axes become one frame.
# Exact 2 m and exact 30 min qualify. Opposite contrast signs do not.
OPENCV_NEAR_METER_GAP_M = 2.0
OPENCV_NEAR_TIME_GAP_MIN = 30.0
# Contrast of the frame against its neighbor background, not the raw temperature.
# Exactly ±0.5 °C stays unknown.
OPENCV_CONTRAST_C = 0.5
OPENCV_HEATING = "heating"
OPENCV_COOLING = "cooling"
OPENCV_UNKNOWN = "unknown"
OPENCV_BOX_COLOR = "magenta"
COMBINED_BOTH_COLOR = "green"


def _score_to_uint8(score_map: np.ndarray) -> np.ndarray:
    """Stretch a finite score map into 0..255. A flat map stays black."""
    score = np.asarray(score_map, dtype=np.float64)
    image = np.zeros(score.shape, dtype=np.uint8)
    finite = np.isfinite(score)
    if not finite.any():
        return image
    values = score[finite]
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.5))
    if hi - lo <= 1e-12:
        return image
    scaled = (np.clip(score, lo, hi) - lo) / (hi - lo)
    scaled = np.where(finite, scaled, 0.0)
    return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)


def _neighbor_ring_delta(
    temperatures: np.ndarray,
    half_points: Optional[int] = None,
) -> np.ndarray:
    """Per-pixel temperature minus the same neighbor ring as local amplitude.

    For one pixel the ring is `half_points` samples to each side on that row
    (default `NEIGHBOR_HALF_POINTS`), plus up to `NEIGHBOR_PRE_STEPS` earlier
    rows in that column. The pixel itself is not in the ring.
    """
    import warnings

    values = np.asarray(temperatures, dtype=np.float64)
    n_time, n_space = values.shape
    if half_points is None:
        half_points = NEIGHBOR_HALF_POINTS
    half_points = int(half_points)
    pre_steps = NEIGHBOR_PRE_STEPS
    delta = np.zeros((n_time, n_space), dtype=np.float64)
    n_samples = half_points * 2 + pre_steps
    chunk = 128
    for start in range(0, n_time, chunk):
        stop = min(n_time, start + chunk)
        hist = max(0, start - pre_steps)
        window = values[hist:stop]
        local_rows = stop - start
        base = start - hist
        stack = np.full((local_rows, n_space, n_samples), np.nan, dtype=np.float64)
        slot = 0
        rows = window[base : base + local_rows]
        for offset in range(1, half_points + 1):
            if offset < n_space:
                stack[:, offset:, slot] = rows[:, :-offset]
            slot += 1
        for offset in range(1, half_points + 1):
            if offset < n_space:
                stack[:, :-offset, slot] = rows[:, offset:]
            slot += 1
        for offset in range(1, pre_steps + 1):
            src = base - offset
            if src >= 0:
                stack[:, :, slot] = window[src : src + local_rows]
            else:
                skip = -src
                if skip < local_rows:
                    stack[skip:, :, slot] = window[: local_rows - skip]
            slot += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reference = np.nanmedian(stack, axis=-1)
        step = rows - reference
        step[~np.isfinite(step)] = 0.0
        delta[start:stop] = step
    return delta


def _opencv_components_from_mask(mask: np.ndarray) -> list[dict]:
    """Connected components of one binary mask. Specks under 15 pixels drop."""
    if not np.any(mask):
        return []
    binary = np.zeros(mask.shape, dtype=np.uint8)
    binary[mask] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, OPENCV_CLOSE_SIZE)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        closed,
        connectivity=8,
    )
    components: list[dict] = []
    for label_id in range(1, int(stats.shape[0])):
        left = int(stats[label_id, cv2.CC_STAT_LEFT])
        top = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < OPENCV_MIN_PIXELS or width < 1 or height < 1:
            continue
        t0 = top
        t1 = top + height - 1
        s0 = left
        s1 = left + width - 1
        owned = labels[t0 : t1 + 1, s0 : s1 + 1] == label_id
        pre_close = mask[t0 : t1 + 1, s0 : s1 + 1] & owned
        span_cells = int(pre_close.size)
        occupied_cells = int(np.count_nonzero(pre_close))
        components.append(
            {
                "source": "opencv",
                "t_range": (t0, t1),
                "s_range": (s0, s1),
                "n_pixels": area,
                "row_fill": float(occupied_cells / span_cells),
            }
        )
    return components


def extract_events_opencv(
    score_map: np.ndarray,
    temperatures: np.ndarray,
    score_percentile: float = 95.0,
    contrast_min_c: float = 1.2,
    half_points: int = NEIGHBOR_HALF_POINTS,
) -> list[dict]:
    """Connected components of the score mask AND the contrast mask.

    The score mask keeps finite scores at or above `score_percentile`
    (default 95). The contrast mask keeps pixels whose absolute neighbor-ring
    contrast is at least `contrast_min_c` degrees (default 1.2). The ring
    width is `half_points` (default `NEIGHBOR_HALF_POINTS`). A flat score or
    a flat temperature returns no components. Specks under 15 pixels are dropped.
    `row_fill`
    is the share of the bounding rectangle covered by this component on the
    anomaly mask before close. A close-filled gap counts as zero. The
    detector event list is not an argument, and this cut is not passed into
    the detector.
    """
    if not HAS_CV2:
        raise RuntimeError("opencv-python is not installed")
    score = np.asarray(score_map, dtype=np.float64)
    temps = np.asarray(temperatures, dtype=np.float64)
    if score.ndim != 2:
        raise ValueError("score_map must be 2-D")
    if temps.shape != score.shape:
        raise ValueError("temperatures must match score_map")
    finite_score = np.isfinite(score)
    if not finite_score.any():
        return []
    score_values = score[finite_score]
    if float(score_values.max() - score_values.min()) <= 1e-12:
        return []
    finite_temp = np.isfinite(temps)
    if not finite_temp.any():
        return []
    temp_values = temps[finite_temp]
    if float(temp_values.max() - temp_values.min()) <= 1e-12:
        return []
    percentile = float(score_percentile)
    contrast_min = float(contrast_min_c)
    score_level = float(np.percentile(score_values, percentile))
    score_on = finite_score & (score >= score_level)
    contrast = np.abs(_neighbor_ring_delta(temps, half_points=half_points))
    contrast_on = finite_temp & (contrast >= contrast_min)
    mask = score_on & contrast_on
    components = _opencv_components_from_mask(mask)
    if not components:
        return []
    components.sort(
        key=lambda item: (-int(item["n_pixels"]), item["t_range"][0], item["s_range"][0])
    )
    return components


def opencv_type_from_temperatures(inside, background: float) -> str:
    """Label one frame from its mean contrast against a background.

    Above +0.5 °C is heating, below −0.5 °C is cooling. Exactly ±0.5 °C,
    and anything in between, is unknown.
    """
    values = np.asarray(inside, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0 or not np.isfinite(background):
        return OPENCV_UNKNOWN
    contrast = float(values.mean() - float(background))
    if contrast > OPENCV_CONTRAST_C:
        return OPENCV_HEATING
    if contrast < -OPENCV_CONTRAST_C:
        return OPENCV_COOLING
    return OPENCV_UNKNOWN


def annotate_opencv_components(
    components: Sequence[dict],
    temperatures: np.ndarray,
    dists: np.ndarray,
    time_step_min: float,
) -> list[dict]:
    """Fill contrast type, meters, and minutes on components already extracted.

    The contrast is the neighbor-ring amplitude of the rectangle. This does
    not threshold the score again.
    """
    values = np.asarray(temperatures, dtype=np.float64)
    n_time, n_space = values.shape
    annotated: list[dict] = []
    for component in components:
        s0, s1 = int(component["s_range"][0]), int(component["s_range"][1])
        t0, t1 = int(component["t_range"][0]), int(component["t_range"][1])
        if s0 > s1:
            s0, s1 = s1, s0
        if t0 > t1:
            t0, t1 = t1, t0
        s0 = int(np.clip(s0, 0, n_space - 1))
        s1 = int(np.clip(s1, 0, n_space - 1))
        t0 = int(np.clip(t0, 0, n_time - 1))
        t1 = int(np.clip(t1, 0, n_time - 1))
        mask = np.zeros(values.shape, dtype=bool)
        mask[t0 : t1 + 1, s0 : s1 + 1] = True
        contrast = neighbor_local_amplitude(values, mask, t0, t1, s0, s1)
        inside = values[t0 : t1 + 1, s0 : s1 + 1]
        finite = inside[np.isfinite(inside)]
        mean = float(finite.mean()) if finite.size else 0.0
        component["amplitude"] = float(contrast)
        component["type_hypothesis"] = opencv_type_from_temperatures(inside, mean - contrast)
        component["duration_min"] = float((t1 - t0 + 1) * float(time_step_min))
        start_m = float(dists[s0])
        end_m = float(dists[s1])
        if start_m > end_m:
            start_m, end_m = end_m, start_m
        component["distance_start_m"] = start_m
        component["distance_end_m"] = end_m
        component["meters"] = [start_m, end_m]
        component["source"] = "opencv"
        annotated.append(component)
    return annotated


def _median_value(values: Sequence[float]) -> float:
    """Middle value, or the mean of the two central values when the count is even."""
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    mid = count // 2
    if count % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _opencv_fixed_keep_count(components: Sequence[dict]) -> int:
    """How many boxes the old floors would keep. Not the adaptive result."""
    floor_area = 40
    floor_duration_min = 4.0
    kept = 0
    for component in components:
        area = _component_area(component)
        duration = float(component["duration_min"])
        if area >= floor_area and duration >= floor_duration_min:
            kept += 1
    return kept


def filter_opencv_components(components: Sequence[dict]) -> list[dict]:
    """Drop OpenCV boxes under an adaptive area or duration cut.

    Medians are taken on the full incoming list, including boxes this step
    drops. The area cut is max(40, 0.25 × median area). The duration cut is
    max(4, 0.3 × median duration). A value exactly at a cut stays. The
    detector event list is not an argument.
    """
    floor_area = 40
    floor_duration_min = 4.0
    area_fraction = 0.25
    duration_fraction = 0.3
    if not components:
        return []
    areas = [float(_component_area(component)) for component in components]
    durations = [float(component["duration_min"]) for component in components]
    area_cut = max(floor_area, area_fraction * _median_value(areas))
    duration_cut = max(floor_duration_min, duration_fraction * _median_value(durations))
    kept: list[dict] = []
    for component in components:
        area = float(_component_area(component))
        duration = float(component["duration_min"])
        if area < area_cut or duration < duration_cut:
            continue
        kept.append(component)
    return kept


def _ensemble_score_mad(score: np.ndarray) -> float:
    """Median absolute deviation of finite ensemble scores from their median."""
    values = np.asarray(score, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if not np.isfinite(mad):
        return 0.0
    return mad


def filter_opencv_row_fill(components: Sequence[dict], score: np.ndarray) -> list[dict]:
    """Drop a thin OpenCV box. The cut follows the file's score spread.

    The spread is the median absolute deviation of finite `ensemble_score`
    from its median, the same spread the detector reads. At or below 0.04
    the cut is 0.6. Strictly above 0.04 the cut is 0.45. There is no ramp
    between those two cuts. A share exactly on the cut stays. Called after
    the area and duration filter. The fill is the share of the bounding
    rectangle on the pre-close anomaly mask. An empty row opened by close
    counts as zero. A speck on an otherwise empty row does not make the row
    full. `duration_min` is left as the full span. The detector event list
    is not an argument, and this cut is not passed into the detector.
    """
    if _ensemble_score_mad(score) <= DETECTOR_CALM_MAD:
        row_fill_min = 0.6
    else:
        row_fill_min = 0.45
    kept: list[dict] = []
    for component in components:
        if float(component["row_fill"]) < row_fill_min:
            continue
        kept.append(component)
    return kept


def _opencv_unknown_count(components: Sequence[dict]) -> int:
    return sum(
        1 for component in components if component.get("type_hypothesis") == OPENCV_UNKNOWN
    )


def retype_opencv_components(components: Sequence[dict]) -> list[dict]:
    """Drop a weak neighbor contrast, then label the boxes that remain.

    Called after the area and duration filter. Absolute contrast strictly
    under 1.2 °C is omitted regardless of area or duration. Exactly +1.2 °C
    is heating and exactly −1.2 °C is cooling. Contrast of at least +0.8 °C
    is heating and at most −0.8 °C is cooling; those cuts only see a box the
    1.2 °C drop already kept. A shorter weak box under 10 minutes is still
    omitted. The detector event list is not an argument.
    """
    contrast_min_c = 1.2
    contrast_c = 0.8
    weak_max_duration_min = 10.0
    kept: list[dict] = []
    for component in components:
        contrast = float(component["amplitude"])
        duration = float(component["duration_min"])
        if abs(contrast) < contrast_min_c:
            continue
        if contrast >= contrast_c:
            component["type_hypothesis"] = OPENCV_HEATING
        elif contrast <= -contrast_c:
            component["type_hypothesis"] = OPENCV_COOLING
        elif abs(contrast) < contrast_c and duration < weak_max_duration_min:
            continue
        else:
            component["type_hypothesis"] = OPENCV_UNKNOWN
        kept.append(component)
    return kept


def _record_meters(record: dict) -> tuple[float, float]:
    if "meters" in record:
        return float(record["meters"][0]), float(record["meters"][1])
    start_m = float(record.get("distance_start_m", 0.0))
    end_m = float(record.get("distance_end_m", record.get("spatial_extent_m", 0.0)))
    return start_m, end_m


def _ordered_pair(pair) -> tuple[float, float]:
    start, end = float(pair[0]), float(pair[1])
    if start > end:
        start, end = end, start
    return start, end


def _ordered_index_pair(pair) -> tuple[int, int]:
    start, end = int(pair[0]), int(pair[1])
    if start > end:
        start, end = end, start
    return start, end


def _component_area(component: dict) -> int:
    if "n_pixels" in component:
        return int(component["n_pixels"])
    return int(component["area"])


def _opencv_strong_overlap(left: dict, right: dict) -> bool:
    """True when both axes overlap by at least half the smaller side."""
    overlap_fraction = 0.5
    left_t0, left_t1 = _ordered_index_pair(left["t_range"])
    right_t0, right_t1 = _ordered_index_pair(right["t_range"])
    overlap_steps = min(left_t1, right_t1) - max(left_t0, right_t0) + 1
    if overlap_steps <= 0:
        return False
    left_steps = left_t1 - left_t0 + 1
    right_steps = right_t1 - right_t0 + 1
    left_duration = float(left["duration_min"])
    right_duration = float(right["duration_min"])
    shorter = min(left_duration, right_duration)
    if shorter <= 0 or left_steps <= 0 or right_steps <= 0:
        return False
    left_step = left_duration / left_steps
    right_step = right_duration / right_steps
    if abs(left_step - right_step) <= 1e-6 * max(1.0, abs(left_step)):
        shorter_steps = min(left_steps, right_steps)
        if overlap_steps < overlap_fraction * shorter_steps:
            return False
    else:
        step = left_step if left_duration <= right_duration else right_step
        if overlap_steps * step < overlap_fraction * shorter:
            return False
    left_m0, left_m1 = _record_meters(left)
    right_m0, right_m1 = _record_meters(right)
    left_width = left_m1 - left_m0
    right_width = right_m1 - right_m0
    narrower = min(left_width, right_width)
    if narrower <= 0:
        return False
    overlap_m = min(left_m1, right_m1) - max(left_m0, right_m0)
    if overlap_m < 0:
        overlap_m = 0.0
    return overlap_m + 1e-9 >= overlap_fraction * narrower


def merge_opencv_components(components: Sequence[dict]) -> list[dict]:
    """Merge OpenCV boxes that overlap by half on both time and distance.

    Clusters are the connected components of that pairwise test on the input
    list. A leftover is not compared again to the merged hull. The detector
    event list is not an argument.
    """
    items = list(components)
    count = len(items)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            if not _opencv_strong_overlap(items[left_index], items[right_index]):
                continue
            left_root = find(left_index)
            right_root = find(right_index)
            if left_root != right_root:
                parent[right_root] = left_root
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    merged: list[dict] = []
    for indexes in groups.values():
        if len(indexes) == 1:
            merged.append(items[indexes[0]])
            continue
        members = [items[index] for index in indexes]
        t_bounds = [_ordered_index_pair(member["t_range"]) for member in members]
        s_bounds = [_ordered_index_pair(member["s_range"]) for member in members]
        meter_bounds = [_record_meters(member) for member in members]
        t0 = min(bound[0] for bound in t_bounds)
        t1 = max(bound[1] for bound in t_bounds)
        s0 = min(bound[0] for bound in s_bounds)
        s1 = max(bound[1] for bound in s_bounds)
        start_m = min(bound[0] for bound in meter_bounds)
        end_m = max(bound[1] for bound in meter_bounds)
        sample = members[0]
        sample_t0, sample_t1 = _ordered_index_pair(sample["t_range"])
        sample_steps = sample_t1 - sample_t0 + 1
        step = float(sample["duration_min"]) / sample_steps
        area = sum(_component_area(member) for member in members)
        types = {str(member.get("type_hypothesis") or OPENCV_UNKNOWN) for member in members}
        box = dict(sample)
        box["t_range"] = (t0, t1)
        box["s_range"] = (s0, s1)
        box["meters"] = [start_m, end_m]
        box["distance_start_m"] = start_m
        box["distance_end_m"] = end_m
        box["duration_min"] = float((t1 - t0 + 1) * step)
        box["n_pixels"] = area
        box["area"] = area
        box["type_hypothesis"] = next(iter(types)) if len(types) == 1 else OPENCV_UNKNOWN
        box["source"] = "opencv"
        merged.append(box)
    return merged


def _opencv_meter_gap(left: dict, right: dict) -> float:
    """Gap between closed meter intervals. Overlap and a shared edge are 0."""
    left_m0, left_m1 = _record_meters(left)
    right_m0, right_m1 = _record_meters(right)
    if left_m0 > left_m1:
        left_m0, left_m1 = left_m1, left_m0
    if right_m0 > right_m1:
        right_m0, right_m1 = right_m1, right_m0
    if left_m1 < right_m0:
        return right_m0 - left_m1
    if right_m1 < left_m0:
        return left_m0 - right_m1
    return 0.0


def _opencv_time_gap_minutes(left: dict, right: dict) -> float:
    """Minutes between inclusive time spans. Adjacent indexes are a gap of 0."""
    left_t0, left_t1 = _ordered_index_pair(left["t_range"])
    right_t0, right_t1 = _ordered_index_pair(right["t_range"])
    gap_steps = _interval_gap(left_t0, left_t1, right_t0, right_t1)
    if gap_steps <= 0:
        return 0.0
    left_steps = left_t1 - left_t0 + 1
    right_steps = right_t1 - right_t0 + 1
    if left_steps <= 0 or right_steps <= 0:
        return float("inf")
    left_step = float(left["duration_min"]) / left_steps
    right_step = float(right["duration_min"]) / right_steps
    if abs(left_step - right_step) <= 1e-6 * max(1.0, abs(left_step)):
        step = left_step
    else:
        step = left_step if float(left["duration_min"]) <= float(right["duration_min"]) else right_step
    return gap_steps * step


def _opencv_near_pair(left: dict, right: dict) -> bool:
    """True when both boxes share a contrast sign and both gaps are within the limits."""
    if _amp_sign(float(left.get("amplitude", 0.0))) != _amp_sign(float(right.get("amplitude", 0.0))):
        return False
    if _opencv_meter_gap(left, right) > OPENCV_NEAR_METER_GAP_M + 1e-9:
        return False
    return _opencv_time_gap_minutes(left, right) <= OPENCV_NEAR_TIME_GAP_MIN + 1e-9


def merge_opencv_nearby(components: Sequence[dict]) -> list[dict]:
    """Join same-sign OpenCV boxes that overlap or nearly touch.

    A pair qualifies when the closed meter gap is at most 2 m and the closed
    time gap is at most 30 min. Clusters are connected components of that
    test on the input list. A leftover is not compared to the merged hull.
    Meters and duration are the union interval, so an interior gap counts in
    the duration. The detector list is not an argument.
    """
    items = list(components)
    count = len(items)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            if not _opencv_near_pair(items[left_index], items[right_index]):
                continue
            left_root = find(left_index)
            right_root = find(right_index)
            if left_root != right_root:
                parent[right_root] = left_root
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    merged: list[dict] = []
    for indexes in groups.values():
        if len(indexes) == 1:
            merged.append(items[indexes[0]])
            continue
        members = [items[index] for index in indexes]
        t_bounds = [_ordered_index_pair(member["t_range"]) for member in members]
        s_bounds = [_ordered_index_pair(member["s_range"]) for member in members]
        meter_bounds = [_record_meters(member) for member in members]
        t0 = min(bound[0] for bound in t_bounds)
        t1 = max(bound[1] for bound in t_bounds)
        s0 = min(bound[0] for bound in s_bounds)
        s1 = max(bound[1] for bound in s_bounds)
        start_m = min(bound[0] for bound in meter_bounds)
        end_m = max(bound[1] for bound in meter_bounds)
        sample = members[0]
        sample_t0, sample_t1 = _ordered_index_pair(sample["t_range"])
        sample_steps = sample_t1 - sample_t0 + 1
        step = float(sample["duration_min"]) / sample_steps
        area = sum(_component_area(member) for member in members)
        types = {str(member.get("type_hypothesis") or OPENCV_UNKNOWN) for member in members}
        box = dict(sample)
        box["t_range"] = (t0, t1)
        box["s_range"] = (s0, s1)
        box["meters"] = [start_m, end_m]
        box["distance_start_m"] = start_m
        box["distance_end_m"] = end_m
        box["duration_min"] = float((t1 - t0 + 1) * step)
        box["n_pixels"] = area
        box["area"] = area
        box["type_hypothesis"] = next(iter(types)) if len(types) == 1 else OPENCV_UNKNOWN
        box["source"] = "opencv"
        box.pop("overlaps_detector", None)
        merged.append(box)
    return merged


def _fraction_overlap(left: dict, right: dict, overlap_fraction: float) -> bool:
    """True when time and meters each cover the fraction of the smaller side.

    Time uses the inclusive index overlap. A zero or negative meter width
    does not count. Exactly the fraction on both axes is true.
    """
    left_t0, left_t1 = _ordered_index_pair(left["t_range"])
    right_t0, right_t1 = _ordered_index_pair(right["t_range"])
    overlap_steps = min(left_t1, right_t1) - max(left_t0, right_t0) + 1
    if overlap_steps <= 0:
        return False
    left_steps = left_t1 - left_t0 + 1
    right_steps = right_t1 - right_t0 + 1
    left_duration = float(left["duration_min"])
    right_duration = float(right["duration_min"])
    shorter = min(left_duration, right_duration)
    if shorter <= 0 or left_steps <= 0 or right_steps <= 0:
        return False
    left_step = left_duration / left_steps
    right_step = right_duration / right_steps
    if abs(left_step - right_step) <= 1e-6 * max(1.0, abs(left_step)):
        shorter_steps = min(left_steps, right_steps)
        if overlap_steps + 1e-9 < overlap_fraction * shorter_steps:
            return False
    else:
        step = left_step if left_duration <= right_duration else right_step
        if overlap_steps * step + 1e-9 < overlap_fraction * shorter:
            return False
    left_m0, left_m1 = _record_meters(left)
    right_m0, right_m1 = _record_meters(right)
    if left_m0 > left_m1:
        left_m0, left_m1 = left_m1, left_m0
    if right_m0 > right_m1:
        right_m0, right_m1 = right_m1, right_m0
    narrower = min(left_m1 - left_m0, right_m1 - right_m0)
    if narrower <= 0:
        return False
    overlap_m = min(left_m1, right_m1) - max(left_m0, right_m0)
    if overlap_m < 0:
        overlap_m = 0.0
    return overlap_m + 1e-9 >= overlap_fraction * narrower


def mark_opencv_detector_overlaps(
    components: Sequence[dict],
    events: Sequence[dict],
) -> list[dict]:
    """Set overlaps_detector on the boxes that are drawn and written.

    Called after the size filter, the type rule, and the merge. True when
    the stored box and a detector event overlap by at least 30% of the
    shorter duration and 30% of the narrower width. The detector list is
    not modified. A touch on only one axis stays false.
    """
    overlap_fraction = 0.3
    labeled: list[dict] = []
    for component in components:
        component["overlaps_detector"] = any(
            _fraction_overlap(component, event, overlap_fraction) for event in events
        )
        labeled.append(component)
    return labeled


def combine_detector_and_opencv(
    events: Sequence[dict],
    components: Sequence[dict],
) -> list[dict]:
    """One list of detector events plus OpenCV boxes that meet none of them.

    A detector event becomes ``both`` when `_fraction_overlap` at 30% hits any
    OpenCV component. Its meters, duration, and type stay the detector event.
    An OpenCV component with `overlaps_detector` true is left out. The inputs
    are not modified, and boxes are not joined into one hull.
    """
    overlap_fraction = 0.3
    combined: list[dict] = []
    for event in events:
        row = dict(event)
        matched = any(
            _fraction_overlap(component, event, overlap_fraction) for component in components
        )
        row["source"] = "both" if matched else "detector"
        row.pop("overlaps_detector", None)
        combined.append(row)
    for component in components:
        if bool(component.get("overlaps_detector")):
            continue
        row = dict(component)
        row["source"] = "opencv"
        combined.append(row)
    return combined


def opencv_legend_label(component: dict, rank: int) -> str:
    """One OpenCV legend line: number, type, signed contrast, meter span."""
    type_name = str(component.get("type_hypothesis") or OPENCV_UNKNOWN)
    contrast = float(component.get("amplitude", 0.0))
    start_m, end_m = _record_meters(component)
    return f"{rank}. {type_name} {contrast:+.1f} C  {start_m:.1f}-{end_m:.1f} m"


def _anomaly_plot_paths(plot_path: str) -> dict[str, str]:
    from pathlib import Path

    path = Path(plot_path)
    suffix = path.suffix or ".png"
    return {
        "model": str(path),
        "opencv": str(path.with_name(f"{path.stem}_opencv{suffix}")),
        "compare": str(path.with_name(f"{path.stem}_compare{suffix}")),
        "combined": str(path.with_name(f"{path.stem}_combined{suffix}")),
    }


def format_opencv_components(
    components: Sequence[dict],
    dists: Optional[np.ndarray] = None,
) -> str:
    """Meter spans for the optional OpenCV list. Not part of the typed report."""
    lines = [f"OpenCV components: {len(components)}."]
    for index, component in enumerate(components, start=1):
        s0, s1 = (int(component["s_range"][0]), int(component["s_range"][1]))
        t0, t1 = (int(component["t_range"][0]), int(component["t_range"][1]))
        if dists is not None and len(dists):
            start_m = float(dists[s0])
            end_m = float(dists[s1])
            if start_m > end_m:
                start_m, end_m = end_m, start_m
            meters = f"{start_m:.1f}-{end_m:.1f}"
        else:
            meters = f"{s0}-{s1}"
        type_name = component.get("type_hypothesis")
        type_bit = f", type={type_name}" if type_name else ""
        if "overlaps_detector" in component:
            overlap_bit = f", overlaps_detector={str(bool(component['overlaps_detector'])).lower()}"
        else:
            overlap_bit = ""
        lines.append(
            f"  opencv {index}: meters={meters}, t_range={t0}-{t1}, "
            f"s_range={s0}-{s1}, pixels={int(component['n_pixels'])}, source=opencv"
            f"{type_bit}{overlap_bit}"
        )
    return "\n".join(lines)


def _morphology_labels(binary: np.ndarray, use_cv2: Optional[bool]) -> np.ndarray:
    if use_cv2 is None:
        use_cv2 = HAS_CV2
    if use_cv2:
        if not HAS_CV2:
            raise RuntimeError("opencv-python is not installed")
        image = binary.astype(np.uint8)
        # Close in time only. A one-column gap is 0.25 m; filling it glues
        # neighboring warm sections into one box wider than 15 m.
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, close_kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3, 2), np.uint8))
        _count, labeled, _stats, _centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        return labeled
    closed = binary_closing(binary, structure=np.ones((3, 5), dtype=bool))
    opened = binary_opening(closed, structure=np.ones((2, 3), dtype=bool))
    labeled, _count = label(opened)
    return labeled


def _amp_sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _describe_event(
    mask: np.ndarray,
    temperatures: np.ndarray,
    background: np.ndarray,
    dists: np.ndarray,
    time_step_min: float,
    n_space: int,
    local_broad_fraction: float,
    stat_score: Optional[np.ndarray],
    ae_score: Optional[np.ndarray],
    neighbor_half_points: int,
    neighbor_pre_steps: int,
) -> dict:
    """One event from a boolean mask. `amplitude` stays the rule field.

    `local_amplitude` is the neighbor-ring contrast and is what confidence uses.
    The mask is kept under `_mask` until merge finishes.
    """
    time_idx, space_idx = np.where(mask)
    t0 = int(time_idx.min())
    t1 = int(time_idx.max())
    space_min = int(space_idx.min())
    space_max = int(space_idx.max())
    duration = t1 - t0 + 1
    spatial_points = space_max - space_min + 1
    extent_m = float(abs(dists[space_max] - dists[space_min]))
    meter_lo = float(min(dists[space_min], dists[space_max]))
    meter_hi = float(max(dists[space_min], dists[space_max]))
    amplitude = float(temperatures[mask].mean() - background[mask].mean())
    spatial_fraction = float(spatial_points) / float(n_space) if n_space else 0.0
    local_amplitude = neighbor_local_amplitude(
        temperatures,
        mask,
        t0,
        t1,
        space_min,
        space_max,
        half_points=neighbor_half_points,
        pre_steps=neighbor_pre_steps,
    )
    event = {
        "duration_steps": duration,
        "duration_min": float(duration * time_step_min),
        "spatial_points": spatial_points,
        "spatial_extent_m": extent_m,
        "spatial_fraction": spatial_fraction,
        "distance_start_m": meter_lo,
        "distance_end_m": meter_hi,
        "amplitude": amplitude,
        "local_amplitude": local_amplitude,
        "n_pixels": int(mask.sum()),
        "t_range": (t0, t1),
        "s_range": (space_min, space_max),
        "t_center": float(time_idx.mean()),
        "s_center": float(space_idx.mean()),
        "_mask": mask,
    }
    event.update(
        _event_shape_measures(temperatures, mask, t0, t1, space_min, space_max, time_step_min)
    )
    if spatial_fraction >= local_broad_fraction:
        outside_amplitude = outside_cable_amplitude(temperatures, mask, t0, t1)
        if outside_amplitude is not None:
            event["amplitude"] = outside_amplitude
        event["type_hypothesis"] = classify_broad_event(event)
    else:
        event["type_hypothesis"] = classify_event_type(event)
    if event["type_hypothesis"] in (LOCAL_HEATING, LOCAL_COOLING):
        event["periodic"] = event_is_periodic(
            temperatures,
            mask,
            t0,
            t1,
            space_min,
            space_max,
            time_step_min,
        )
    event["confidence"] = event_confidence(
        local_amplitude,
        mask,
        t0,
        t1,
        stat_score=stat_score,
        ae_score=ae_score,
    )
    return event


def _can_merge(left: dict, right: dict, gap_points: int, gap_steps: int) -> bool:
    if _amp_sign(float(left["local_amplitude"])) != _amp_sign(float(right["local_amplitude"])):
        return False
    time_gap = _interval_gap(left["t_range"][0], left["t_range"][1], right["t_range"][0], right["t_range"][1])
    space_gap = _interval_gap(left["s_range"][0], left["s_range"][1], right["s_range"][0], right["s_range"][1])
    return time_gap < gap_steps and space_gap < gap_points


def merge_nearby_events(
    events: Sequence[dict],
    temperatures: np.ndarray,
    background: np.ndarray,
    dists: np.ndarray,
    time_step_min: float,
    n_space: int,
    local_broad_fraction: float,
    stat_score: Optional[np.ndarray],
    ae_score: Optional[np.ndarray],
    neighbor_half_points: int = NEIGHBOR_HALF_POINTS,
    neighbor_pre_steps: int = NEIGHBOR_PRE_STEPS,
    merge_gap_points: int = MERGE_MAX_GAP_POINTS,
    merge_gap_steps: int = MERGE_MAX_GAP_STEPS,
) -> list[dict]:
    """Union pieces that sit on nearly the same cable span, close in time, same sign.

    The merged mask is the union of the pixels, not the cold gap between them.
    The reported time and meter span is the bounding union. Amplitude and
    confidence are recomputed on that union.
    """
    pending = [dict(event) for event in events]
    while len(pending) > 1:
        best: Optional[tuple] = None
        for i in range(len(pending)):
            for j in range(i + 1, len(pending)):
                if not _can_merge(pending[i], pending[j], merge_gap_points, merge_gap_steps):
                    continue
                time_gap = _interval_gap(
                    pending[i]["t_range"][0],
                    pending[i]["t_range"][1],
                    pending[j]["t_range"][0],
                    pending[j]["t_range"][1],
                )
                space_gap = _interval_gap(
                    pending[i]["s_range"][0],
                    pending[i]["s_range"][1],
                    pending[j]["s_range"][0],
                    pending[j]["s_range"][1],
                )
                key = (time_gap, space_gap, pending[i]["t_range"][0], pending[i]["s_range"][0], i, j)
                if best is None or key < best[0]:
                    best = (key, i, j)
        if best is None:
            break
        _key, i, j = best
        mask = np.logical_or(pending[i]["_mask"], pending[j]["_mask"])
        merged = _describe_event(
            mask,
            temperatures,
            background,
            dists,
            time_step_min,
            n_space,
            local_broad_fraction,
            stat_score,
            ae_score,
            neighbor_half_points,
            neighbor_pre_steps,
        )
        pending = [event for index, event in enumerate(pending) if index != i and index != j]
        pending.append(merged)
    return pending


def extract_events(
    score_map: np.ndarray,
    original: np.ndarray,
    dists: Optional[np.ndarray] = None,
    percentile: float = DETECTOR_PERCENTILE,
    min_pixels: int = 15,
    min_duration: int = 3,
    min_spatial: int = 5,
    time_step_min: float = 2.0,
    timestamps: Optional[Sequence[str]] = None,
    use_cv2: Optional[bool] = None,
    segment_rows: int = SEGMENT_ROWS,
    min_duration_steps: int = MIN_DURATION_STEPS,
    min_local_amplitude_c: float = MIN_LOCAL_AMPLITUDE_C,
    local_broad_fraction: float = LOCAL_BROAD_FRACTION,
    stat_score: Optional[np.ndarray] = None,
    ae_score: Optional[np.ndarray] = None,
    neighbor_half_points: int = NEIGHBOR_HALF_POINTS,
    neighbor_pre_steps: int = NEIGHBOR_PRE_STEPS,
    merge_gap_points: int = MERGE_MAX_GAP_POINTS,
    merge_gap_steps: int = MERGE_MAX_GAP_STEPS,
    min_confidence: float = 0.0,
) -> FilteredEvents:
    """Segment-threshold, clean, type, then lightly merge nearby pieces.

    A component whose spatial-sample share is below `local_broad_fraction`
    keeps the local labels. A wider component is typed broad and is not
    dropped for width. `min_duration` is the morphology gate (default 3).
    `min_duration_steps` drops only local flashes (default 6).
    `min_confidence` is off at 0. Raise it to drop the weakest kept events.
    """
    score = np.asarray(score_map, dtype=np.float64)
    temperatures = np.asarray(original, dtype=np.float64)
    if score.shape != temperatures.shape:
        raise ValueError("score_map and original must have the same shape")
    if dists is None:
        dists_arr = np.arange(score.shape[1], dtype=np.float64) * 0.25
    else:
        dists_arr = np.asarray(dists, dtype=np.float64)
        if dists_arr.shape != (score.shape[1],):
            raise ValueError("dists length must match the spatial axis")
    if timestamps is not None:
        parsed_step = median_timestamp_step_minutes(timestamps)
        if parsed_step is not None:
            time_step_min = parsed_step
    applied_percentile = detector_percentile_for_score(score, percentile)
    binary = segment_anomaly_mask(score, applied_percentile, segment_rows=segment_rows)
    labeled = _morphology_labels(binary, use_cv2)
    labels = [int(value) for value in np.unique(labeled) if int(value) != 0]
    background = local_spacetime_background(temperatures)
    n_space = int(score.shape[1])
    events = []
    for marker in labels:
        mask = labeled == marker
        n_pixels = int(mask.sum())
        if n_pixels < min_pixels:
            continue
        time_idx, space_idx = np.where(mask)
        duration = int(time_idx.max() - time_idx.min() + 1)
        spatial_points = int(space_idx.max() - space_idx.min() + 1)
        if duration < min_duration or spatial_points < min_spatial:
            continue
        event = _describe_event(
            mask,
            temperatures,
            background,
            dists_arr,
            time_step_min,
            n_space,
            local_broad_fraction,
            stat_score,
            ae_score,
            neighbor_half_points,
            neighbor_pre_steps,
        )
        events.append(event)
    kept, tally = filter_junk_events(
        events,
        min_duration_steps=min_duration_steps,
        min_local_amplitude_c=min_local_amplitude_c,
    )
    kept = merge_nearby_events(
        kept,
        temperatures,
        background,
        dists_arr,
        time_step_min,
        n_space,
        local_broad_fraction,
        stat_score,
        ae_score,
        neighbor_half_points=neighbor_half_points,
        neighbor_pre_steps=neighbor_pre_steps,
        merge_gap_points=merge_gap_points,
        merge_gap_steps=merge_gap_steps,
    )
    if min_confidence > 0:
        confident = []
        for event in kept:
            if float(event["confidence"]) < min_confidence:
                tally["confidence"] = int(tally.get("confidence", 0)) + 1
                continue
            confident.append(event)
        kept = confident
    for event in kept:
        event.pop("_mask", None)
    kept.sort(
        key=lambda item: (
            -float(item["confidence"]),
            -item["n_pixels"],
            -abs(item["amplitude"]),
            item["t_range"][0],
            item["s_range"][0],
        )
    )
    return FilteredEvents(kept, tally)


def format_event_report(events: Sequence[dict]) -> str:
    tally = getattr(events, "tally", None) or {
        "duration": 0,
        "amplitude": 0,
        "confidence": 0,
    }
    lines = [
        format_filter_line(tally),
        format_category_line(events),
        f"Detected {len(events)} events.",
    ]
    for index, event in enumerate(events, start=1):
        start_m = float(event.get("distance_start_m", 0.0))
        end_m = float(event.get("distance_end_m", event.get("spatial_extent_m", 0.0)))
        lines.append(
            f"  {index}: duration={event['duration_steps']} steps "
            f"({event['duration_min']:.2f} min), "
            f"spatial extent={event['spatial_extent_m']:.2f} m, "
            f"amplitude={event['amplitude']:.2f} C, "
            f"local_amplitude={float(event.get('local_amplitude', event['amplitude'])):.2f} C, "
            f"confidence={float(event.get('confidence', 0.0)):.2f}, "
            f"meters={start_m:.1f}-{end_m:.1f}, "
            f"type={event['type_hypothesis']}"
            + (
                f", periodic={str(bool(event['periodic'])).lower()}"
                if "periodic" in event
                else ""
            )
        )
    return "\n".join(lines)


def _short_type(label: str) -> str:
    return str(label).split(" (", 1)[0]


def _event_label(event: dict) -> str:
    amp = float(event.get("local_amplitude", event.get("amplitude", 0.0)))
    start_m = float(event.get("distance_start_m", 0.0))
    end_m = float(event.get("distance_end_m", event.get("spatial_extent_m", 0.0)))
    text = (
        f"{_short_type(event.get('type_hypothesis', ''))} {amp:+.1f} C"
        f"  conf {float(event.get('confidence', 0.0)):.2f}"
        f"  {start_m:.1f}-{end_m:.1f} m"
    )
    if event.get("type_hypothesis") in BROAD_HYPOTHESES and event.get("max_dt_per_min") is not None:
        text += f"  {float(event['max_dt_per_min']):.2f} C/min"
    return text


def _mark_anchor(t0: int, t1: int, s1: int, placed: list[tuple[float, float]]) -> tuple[float, float]:
    """Box-top center, nudged right when another mark already sits on the same corner."""
    cx = 0.5 * (float(t0) + float(t1))
    sy = float(s1)
    for _ in range(6):
        crowded = any(abs(cx - px) < 28.0 and abs(sy - py) < 22.0 for px, py in placed)
        if not crowded:
            break
        cx += 30.0
    placed.append((cx, sy))
    return cx, sy


def _legend(axis, handles, title: str, labelcolor: Optional[str] = None) -> None:
    if not handles:
        return
    kwargs = dict(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=7,
        framealpha=0.95,
        title=title,
    )
    if labelcolor is not None:
        kwargs["labelcolor"] = labelcolor
    axis.legend(**kwargs)


def _number_mark(axis, rank: int, t0: int, t1: int, s1: int, placed: list, face: str) -> None:
    cx, sy = _mark_anchor(int(t0), int(t1), int(s1), placed)
    axis.text(
        cx,
        sy,
        str(rank),
        color="white",
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=5,
        bbox={
            "boxstyle": "circle,pad=0.2",
            "facecolor": face,
            "edgecolor": "white",
            "alpha": 0.9,
            "linewidth": 0.6,
        },
    )


def _detector_handles(events: Sequence[dict]) -> list:
    from matplotlib.lines import Line2D

    handles = []
    for rank, event in enumerate(events, start=1):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color="none",
                markerfacecolor="black",
                markeredgecolor="white",
                markersize=7,
                label=f"{rank}. {_event_label(event)}",
            )
        )
    return handles


def _opencv_handles(components: Sequence[dict]) -> list:
    from matplotlib.lines import Line2D

    handles = []
    for rank, component in enumerate(components, start=1):
        handles.append(
            Line2D(
                [0],
                [0],
                color=OPENCV_BOX_COLOR,
                linewidth=2.0,
                label=opencv_legend_label(component, rank),
            )
        )
    return handles


def _draw_detector_on(axis, events: Sequence[dict], n_space: int, numbered: bool) -> None:
    placed: list[tuple[float, float]] = []
    for rank, event in enumerate(events, start=1):
        t0, t1 = event["t_range"]
        s0, s1 = event["s_range"]
        axis.plot([t0, t1, t1, t0, t0], [s0, s0, s1, s1, s0], color="white", linewidth=1.2)
        ring0 = max(0, int(s0) - NEIGHBOR_HALF_POINTS)
        ring1 = min(n_space - 1, int(s1) + NEIGHBOR_HALF_POINTS)
        axis.plot(
            [t0, t1, t1, t0, t0],
            [ring0, ring0, ring1, ring1, ring0],
            color="white",
            linewidth=0.7,
            linestyle="--",
        )
        if numbered:
            _number_mark(axis, rank, int(t0), int(t1), int(s1), placed, "black")


def _draw_detector_score(axis, events: Sequence[dict]) -> None:
    for event in events:
        t0, t1 = event["t_range"]
        s0, s1 = event["s_range"]
        axis.plot([t0, t1, t1, t0, t0], [s0, s0, s1, s1, s0], "c-", linewidth=1.2)


def _combined_outline_color(source: str) -> str:
    """Outline color for one combined record. Detector stays white, OpenCV stays magenta."""
    if source == "opencv":
        return OPENCV_BOX_COLOR
    if source == "both":
        return COMBINED_BOTH_COLOR
    return "white"


def _combined_legend_label(record: dict, rank: int) -> str:
    start_m, end_m = _record_meters(record)
    return f"{rank}. {record['source']}  {start_m:.1f}-{end_m:.1f} m"


def _combined_handles(records: Sequence[dict]) -> list:
    from matplotlib.lines import Line2D

    handles = []
    for rank, record in enumerate(records, start=1):
        handles.append(
            Line2D(
                [0],
                [0],
                color=_combined_outline_color(str(record["source"])),
                linewidth=2.0,
                label=_combined_legend_label(record, rank),
            )
        )
    return handles


def _draw_combined_on(axis, records: Sequence[dict], numbered: bool) -> None:
    """One solid outline per combined record. No neighbor ring and no box cap."""
    placed: list[tuple[float, float]] = []
    for rank, record in enumerate(records, start=1):
        t0, t1 = record["t_range"]
        s0, s1 = record["s_range"]
        source = str(record["source"])
        axis.plot(
            [t0, t1, t1, t0, t0],
            [s0, s0, s1, s1, s0],
            color=_combined_outline_color(source),
            linewidth=1.2,
            linestyle="-",
        )
        if numbered:
            face = "black" if source == "detector" else _combined_outline_color(source)
            _number_mark(axis, rank, int(t0), int(t1), int(s1), placed, face)


def _draw_opencv_on(axis, components: Sequence[dict], numbered: bool) -> None:
    placed: list[tuple[float, float]] = []
    for rank, component in enumerate(components, start=1):
        t0, t1 = component["t_range"]
        s0, s1 = component["s_range"]
        axis.plot(
            [t0, t1, t1, t0, t0],
            [s0, s0, s1, s1, s0],
            color=OPENCV_BOX_COLOR,
            linewidth=1.0,
            linestyle="--" if component.get("overlaps_detector") else "-",
        )
        if numbered:
            _number_mark(axis, rank, int(t0), int(t1), int(s1), placed, OPENCV_BOX_COLOR)


def save_anomaly_plot(
    matrix: np.ndarray,
    score: np.ndarray,
    events: Sequence[dict],
    plot_path: str,
    max_boxes: int = 8,
    opencv_events: Optional[Sequence[dict]] = None,
    mode: str = "model",
    combined_events: Optional[Sequence[dict]] = None,
) -> None:
    """Write one figure. mode is model, opencv, compare, or combined."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if mode not in {"model", "opencv", "compare", "combined"}:
        raise ValueError(f"unknown plot mode {mode}")
    n_space = int(np.asarray(matrix).shape[1])
    temperatures = np.asarray(matrix).T
    scores = np.asarray(score).T
    if mode == "combined":
        records = list(combined_events or [])
        figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].imshow(temperatures, aspect="auto", cmap="viridis", origin="lower")
        axes[0].set_title("Temperature, combined sources")
        axes[0].set_ylabel("distance index")
        _draw_combined_on(axes[0], records, numbered=True)
        _legend(axes[0], _combined_handles(records), "source")
        axes[1].imshow(scores, aspect="auto", cmap="hot", origin="lower")
        axes[1].set_title("Ensemble anomaly score")
        axes[1].set_xlabel("time index")
        axes[1].set_ylabel("distance index")
        figure.tight_layout()
        figure.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        return

    shown = list(events)[:max_boxes]
    components = list(opencv_events or [])

    if mode == "compare":
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].imshow(temperatures, aspect="auto", cmap="viridis", origin="lower")
        axes[0].set_title("Model")
        axes[0].set_ylabel("distance index")
        _draw_detector_on(axes[0], shown, n_space, numbered=True)
        _legend(axes[0], _detector_handles(shown), "top events")
        axes[1].imshow(temperatures, aspect="auto", cmap="viridis", origin="lower")
        axes[1].set_title("OpenCV")
        axes[1].set_xlabel("time index")
        axes[1].set_ylabel("distance index")
        _draw_opencv_on(axes[1], components, numbered=True)
        _legend(axes[1], _opencv_handles(components), "OpenCV", labelcolor=OPENCV_BOX_COLOR)
        figure.tight_layout()
        figure.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(figure)
        return

    broad = [event for event in shown if event.get("type_hypothesis") in BROAD_HYPOTHESES]
    if mode == "model" and broad:
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(12, 9),
            gridspec_kw={"height_ratios": [3, 3, 1.2]},
        )
    else:
        figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].imshow(temperatures, aspect="auto", cmap="viridis", origin="lower")
    axes[0].set_ylabel("distance index")
    axes[1].imshow(scores, aspect="auto", cmap="hot", origin="lower")
    axes[1].set_title("Ensemble anomaly score")
    axes[1].set_xlabel("time index")
    axes[1].set_ylabel("distance index")
    if mode == "model":
        axes[0].set_title("Temperature, event boxes, neighbor ring")
        _draw_detector_on(axes[0], shown, n_space, numbered=True)
        _draw_detector_score(axes[1], shown)
        _legend(axes[0], _detector_handles(shown), "top events")
        if broad:
            rates = [float(event.get("max_dt_per_min") or 0.0) for event in broad]
            labels = [
                f"{_short_type(event.get('type_hypothesis', ''))} "
                f"{float(event.get('local_amplitude', 0.0)):+.1f} C"
                for event in broad
            ]
            axes[2].barh(range(len(broad)), rates, color="firebrick")
            axes[2].axvline(FIRE_MAX_DT_PER_MIN, color="gray", linestyle="--", linewidth=0.8)
            axes[2].set_yticks(range(len(broad)))
            axes[2].set_yticklabels(labels, fontsize=7)
            axes[2].set_xlabel("max dT/dt, C per min")
            axes[2].set_title("Wide events")
    else:
        axes[0].set_title("Temperature, OpenCV boxes")
        _draw_opencv_on(axes[0], components, numbered=True)
        _draw_opencv_on(axes[1], components, numbered=False)
        _legend(axes[0], _opencv_handles(components), "OpenCV", labelcolor=OPENCV_BOX_COLOR)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def add_opencv_narrow_ring_boxes(
    components: Sequence[dict],
    score: np.ndarray,
    denoised: np.ndarray,
    temperatures: np.ndarray,
    dists: np.ndarray,
    time_step_min: float,
    score_percentile: float = 95.0,
    contrast_min_c: float = 1.2,
) -> list[dict]:
    """On a calm file, keep dense 16-sample boxes the wide ring did not keep.

    A box is added when its rectangle share is at least 0.6, its duration is
    at least 90 minutes, and it does not overlap by half on both axes a box
    already kept that itself has share at least 0.6. A noisy file is returned
    unchanged. The score percentile and the 1.2 °C cut stay as passed.
    """
    if _ensemble_score_mad(score) > DETECTOR_CALM_MAD:
        return list(components)
    fresh = extract_events_opencv(
        score,
        denoised,
        score_percentile=score_percentile,
        contrast_min_c=contrast_min_c,
        half_points=NEIGHBOR_HALF_POINTS,
    )
    annotate_opencv_components(fresh, temperatures, dists, time_step_min)
    kept = list(components)
    dense = [item for item in kept if float(item["row_fill"]) >= 0.6]
    for item in fresh:
        if float(item["row_fill"]) < 0.6:
            continue
        if float(item["duration_min"]) < OPENCV_CALM_FRAGMENT_MINUTES:
            continue
        if int(item["n_pixels"]) < OPENCV_MIN_PIXELS:
            continue
        if any(_opencv_strong_overlap(item, other) for other in dense):
            continue
        kept.append(item)
        dense.append(item)
    kept.sort(key=lambda item: (-int(item["n_pixels"]), item["t_range"][0], item["s_range"][0]))
    return kept


def drop_opencv_fragments_covered_by_long_box(
    components: Sequence[dict],
    score: np.ndarray,
) -> list[dict]:
    """On a calm file, drop a short box that sits inside a long box's meters.

    Duration strictly under 90 minutes is dropped only when both meter ends
    lie inside a box of at least 200 minutes. A noisy file is returned
    unchanged. Share, contrast, and the score percentile are not read.
    """
    if _ensemble_score_mad(score) > DETECTOR_CALM_MAD:
        return list(components)
    long_boxes = [
        item
        for item in components
        if float(item["duration_min"]) >= OPENCV_CALM_LONG_MINUTES
    ]
    kept: list[dict] = []
    for component in components:
        if float(component["duration_min"]) >= OPENCV_CALM_FRAGMENT_MINUTES:
            kept.append(component)
            continue
        start_m, end_m = _record_meters(component)
        covered = False
        for long_box in long_boxes:
            long_start, long_end = _record_meters(long_box)
            if start_m + 1e-9 >= long_start and end_m <= long_end + 1e-9:
                covered = True
                break
        if not covered:
            kept.append(component)
    return kept


def collect_opencv_components(
    score: np.ndarray,
    denoised: np.ndarray,
    temperatures: np.ndarray,
    dists: np.ndarray,
    time_step_min: float,
    score_percentile: float = 95.0,
    contrast_min_c: float = 1.2,
) -> list[dict]:
    """Build the OpenCV list up to, but not including, detector overlap marks.

    A calm file (ensemble-score MAD at or below 0.04) segments the wide ring
    and then adds dense 16-sample boxes. A noisy file segments only the
    16-sample ring. Both still require score percentile and contrast cuts.
    """
    calm = _ensemble_score_mad(score) <= DETECTOR_CALM_MAD
    half_points = OPENCV_CALM_HALF_POINTS if calm else NEIGHBOR_HALF_POINTS
    opencv_events = extract_events_opencv(
        score,
        denoised,
        score_percentile=score_percentile,
        contrast_min_c=contrast_min_c,
        half_points=half_points,
    )
    annotate_opencv_components(opencv_events, temperatures, dists, time_step_min)
    if calm:
        opencv_events = add_opencv_narrow_ring_boxes(
            opencv_events,
            score,
            denoised,
            temperatures,
            dists,
            time_step_min,
            score_percentile=score_percentile,
            contrast_min_c=contrast_min_c,
        )
    before_filter = len(opencv_events)
    fixed_kept = _opencv_fixed_keep_count(opencv_events)
    opencv_events = filter_opencv_components(opencv_events)
    print(f"OpenCV components before filter: {before_filter}.", flush=True)
    print(f"OpenCV fixed cuts would keep: {fixed_kept}.", flush=True)
    print(
        f"OpenCV after adaptive filter: {len(opencv_events)} components, "
        f"unknown {_opencv_unknown_count(opencv_events)}.",
        flush=True,
    )
    opencv_events = filter_opencv_row_fill(opencv_events, score)
    print(f"OpenCV after row fill: {len(opencv_events)}.", flush=True)
    opencv_events = retype_opencv_components(opencv_events)
    print(
        f"OpenCV after type rule: {len(opencv_events)} components, "
        f"unknown {_opencv_unknown_count(opencv_events)}.",
        flush=True,
    )
    before_merge = len(opencv_events)
    opencv_events = merge_opencv_components(opencv_events)
    print(f"OpenCV before merge: {before_merge}.", flush=True)
    print(f"OpenCV after merge: {len(opencv_events)}.", flush=True)
    if calm:
        opencv_events = drop_opencv_fragments_covered_by_long_box(opencv_events, score)
        print(f"OpenCV after calm fragment drop: {len(opencv_events)}.", flush=True)
    before_near = len(opencv_events)
    opencv_events = merge_opencv_nearby(opencv_events)
    print(f"OpenCV before near merge: {before_near}.", flush=True)
    print(f"OpenCV after near merge: {len(opencv_events)}.", flush=True)
    return opencv_events


def run_pipeline(
    csv_path: str,
    use_ae: bool = True,
    percentile: float = DETECTOR_PERCENTILE,
    max_rows: Optional[int] = None,
    plot_path: Optional[str] = None,
    seed: int = 0,
    epochs: int = 6,
    device: str = "cpu",
    window_steps: int = TEMPORAL_MEDIAN_STEPS,
    segment_rows: int = SEGMENT_ROWS,
    use_opencv: bool = False,
    opencv_score_percentile: float = 95.0,
    opencv_contrast_min: float = 1.2,
) -> dict:
    set_seed(seed)
    print(f"loading {csv_path}", flush=True)
    matrix, dists, timestamps = load_dts_to_matrix(csv_path, max_rows=max_rows)
    applied_window = temporal_median_window(matrix.shape[0], window_steps)
    print(
        f"matrix={matrix.shape[0]}x{matrix.shape[1]} device={device} seed={seed} "
        f"epochs={epochs if use_ae else 0} temporal_window={applied_window} "
        f"segment_rows={int(segment_rows)}",
        flush=True,
    )
    denoised = denoise_matrix(matrix)
    print("scoring ensemble", flush=True)
    parts = compute_score_parts(
        denoised,
        use_ae=use_ae,
        seed=seed,
        epochs=epochs,
        device=device,
        window_steps=window_steps,
    )
    score = parts["ensemble"]
    if not np.isfinite(score).all():
        raise RuntimeError("ensemble score is not finite")
    applied_percentile = detector_percentile_for_score(score, percentile)
    print(f"detector_percentile={applied_percentile:.2f}", flush=True)
    events = extract_events(
        score,
        matrix,
        dists=dists,
        percentile=percentile,
        timestamps=timestamps,
        segment_rows=segment_rows,
        stat_score=parts["statistical"],
        ae_score=parts["autoencoder"],
    )
    print(format_event_report(events), flush=True)
    opencv_events: list[dict] = []
    if use_opencv:
        step = median_timestamp_step_minutes(timestamps)
        if step is None:
            step = 2.0
        opencv_events = collect_opencv_components(
            score,
            denoised,
            matrix,
            dists,
            step,
            score_percentile=opencv_score_percentile,
            contrast_min_c=opencv_contrast_min,
        )
        opencv_events = mark_opencv_detector_overlaps(opencv_events, events)
        n_overlap = sum(1 for component in opencv_events if component.get("overlaps_detector"))
        print(
            f"OpenCV overlaps detector: {n_overlap}, unique {len(opencv_events) - n_overlap}.",
            flush=True,
        )
        print(format_opencv_components(opencv_events, dists), flush=True)
    if plot_path is not None:
        if use_opencv:
            paths = _anomaly_plot_paths(plot_path)
            save_anomaly_plot(matrix, score, events, paths["model"])
            print(f"wrote {paths['model']}", flush=True)
            save_anomaly_plot(
                matrix,
                score,
                events,
                paths["opencv"],
                mode="opencv",
                opencv_events=opencv_events,
            )
            print(f"wrote {paths['opencv']}", flush=True)
            save_anomaly_plot(
                matrix,
                score,
                events,
                paths["compare"],
                mode="compare",
                opencv_events=opencv_events,
            )
            print(f"wrote {paths['compare']}", flush=True)
            combined_records = combine_detector_and_opencv(events, opencv_events)
            save_anomaly_plot(
                matrix,
                score,
                events,
                paths["combined"],
                mode="combined",
                combined_events=combined_records,
            )
            print(f"wrote {paths['combined']}", flush=True)
        else:
            save_anomaly_plot(matrix, score, events, plot_path)
            print(f"wrote {plot_path}", flush=True)
    return {
        "matrix": matrix,
        "dists": dists,
        "timestamps": timestamps,
        "denoised": denoised,
        "ensemble_score": score,
        "events": events,
        "opencv_events": opencv_events,
        "filter_tally": dict(getattr(events, "tally", {})),
        "category_counts": category_counts(events),
        "plot_path": plot_path,
    }


def iter_csvs(path: str) -> Iterable[str]:
    from pathlib import Path

    candidate = Path(path)
    if candidate.is_dir():
        files = sorted(candidate.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no CSV files in {candidate}")
        return [str(item) for item in files]
    if not candidate.is_file():
        raise FileNotFoundError(path)
    return [str(candidate)]
