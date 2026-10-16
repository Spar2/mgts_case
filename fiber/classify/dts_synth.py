"""Windows from the dts_synth_v1 HDF5 set.

Train and validation stay on their own sequences. Tracing-only crops are left
out: that class is not one of the six names. Periodic_Thermal and Sensor_Fault
have no pixels in this set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fiber.classify.agree_model import agree_table
from fiber.classify.bulk import bulk_table
from fiber.classify.context import context_table, wide_context_crops
from fiber.classify.contrast import contrast_table
from fiber.classify.excess import excess_table
from fiber.classify.neighbor import neighbor_table
from fiber.classify.persist import persist_table
from fiber.classify.physical import FEATURE_NAMES as PHYSICAL_FEATURES
from fiber.classify.physical import shape_features
from fiber.classify.plateau import plateau_table
from fiber.classify.quantile import quantile_table
from fiber.classify.regime import regime_table
from fiber.classify.residual import meter_crops
from fiber.classify.rules import BROAD_WIDTH_M, LABELS
from fiber.classify.scan import MIN_CLASS_CELLS, deviation_crops
from fiber.classify.stability import stability_table
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP

PRESENT = ("Background", "Local_Heating", "Local_Cooling", "Broad_Shift")
FEATURE_TABLES = {
    "context": context_table,
    "agree": agree_table,
    "contrast": contrast_table,
    "bulk": bulk_table,
    "neighbor": neighbor_table,
    "excess": excess_table,
    "persist": persist_table,
    "quantile": quantile_table,
    "regime": regime_table,
    "stability": stability_table,
}
def read_h5(path: Path) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        data = {key: handle[key][()] for key in (
            "noisy", "clean", "mask_event", "mask_nuisance", "x", "t_unix", "sigma_x", "n_valid",
        )}
    data["valid"] = np.isfinite(data["noisy"])
    data["x"] = np.asarray(data["x"], dtype=np.float64)
    data["t_unix"] = np.asarray(data["t_unix"], dtype=np.int64)
    data["sigma_x"] = np.asarray(data["sigma_x"], dtype=np.float32)
    return data


def join_h5(left: dict, right: dict) -> dict:
    """Glue two time-halves. Static distance arrays stay from the first half."""
    out = {}
    rows = left["noisy"].shape[0]
    for key, value in left.items():
        other = right[key]
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == rows and key not in ("x", "sigma_x"):
            out[key] = np.concatenate([value, other], axis=0)
        else:
            out[key] = value
    out["valid"] = np.isfinite(out["noisy"])
    return out


def iter_split(root: Path, split: str):
    """Yield (seq_id, record) with _a/_b halves joined."""
    folder = Path(root) / "h5" / split
    halves: dict[str, dict[str, Path]] = {}
    wholes: list[Path] = []
    for path in sorted(folder.glob("*.h5")):
        stem = path.stem
        if stem.endswith("_a") or stem.endswith("_b"):
            halves.setdefault(stem[:-2], {})[stem[-1]] = path
        else:
            wholes.append(path)
    for path in wholes:
        yield path.stem, read_h5(path)
    for seq_id, parts in sorted(halves.items()):
        if "a" not in parts or "b" not in parts:
            raise FileNotFoundError(f"{seq_id} is missing one time half in {folder}")
        yield seq_id, join_h5(read_h5(parts["a"]), read_h5(parts["b"]))


def crop_label(mask: np.ndarray, section_width_m: float) -> str | None:
    """Map one crop. None means the only event is tracing.

    `section_width_m` is the full run of section heating touched by the crop,
    not the width of the crop itself. A 40-column crop is often shorter than 40 m.
    """
    heat = int(np.count_nonzero(mask == 2))
    cool = int(np.count_nonzero((mask == 5) | (mask == 6)))
    section = int(np.count_nonzero((mask == 3) | (mask == 4)))
    trace = int(np.count_nonzero(mask == 1))
    if heat + cool + section == 0 and trace > 0:
        return None
    best = max(heat, cool, section)
    if best < MIN_CLASS_CELLS:
        return "Background"
    if section == best:
        if section_width_m > BROAD_WIDTH_M:
            return "Broad_Shift"
        return "Local_Heating"
    if heat == best:
        return "Local_Heating"
    return "Local_Cooling"


def _section_width(mask, x, t0: int, x0: int, x1: int) -> float:
    rows = np.asarray(mask[t0:t0 + TIME_CROP])
    touched = ((rows == 3) | (rows == 4)).any(axis=0)
    if not np.any(touched[x0:x1]):
        return 0.0
    left = x0
    while left > 0 and touched[left - 1]:
        left -= 1
    right = x1 - 1
    while right + 1 < len(touched) and touched[right + 1]:
        right += 1
    return float(x[right] - x[left])


def _slice_label(mask, valid, x, t0: int, x0: int, x1: int) -> str | None:
    view = valid[t0:t0 + TIME_CROP, x0:x1]
    if view.size == 0 or float(view.mean()) < 0.5:
        return None
    width = _section_width(mask, x, t0, x0, x1)
    return crop_label(mask[t0:t0 + TIME_CROP, x0:x1], width)


def _column_label(mask, valid, x, t0: int, x0: int) -> str | None:
    return _slice_label(mask, valid, x, t0, x0, x0 + SPACE_CROP)


def _metre_label(mask, valid, x, t0: int, x0: int) -> str | None:
    end = float(x[x0]) + 64.0
    x1 = int(np.searchsorted(x, end, side="left"))
    x1 = min(max(x1, x0 + 1), len(x))
    return _slice_label(mask, valid, x, t0, x0, x1)


def _pools(mask: np.ndarray, nuisance: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Local_Heating": np.argwhere((mask == 2) | (mask == 3) | (mask == 4)),
        "Local_Cooling": np.argwhere((mask == 5) | (mask == 6)),
        "Broad_Shift": np.argwhere((mask == 3) | (mask == 4)),
        "nuisance": np.argwhere(np.asarray(nuisance) > 0),
    }


def _empty_buckets() -> dict[str, list[tuple[int, int]]]:
    return {name: [] for name in PRESENT}


def sample_windows(record: dict, rng: np.random.Generator, per_class: int, mode: str) -> dict[str, np.ndarray]:
    """Up to `per_class` origins per present class. About half the draws start on an event."""
    mask = record["mask_event"]
    valid = record["valid"]
    x = record["x"]
    nt, nx = mask.shape
    buckets = _empty_buckets()
    if nt < TIME_CROP or (mode == "columns" and nx < SPACE_CROP) or (mode == "metres" and nx < 2):
        return {name: np.zeros((0, 2), dtype=np.int64) for name in PRESENT}
    pools = _pools(mask, record["mask_nuisance"])
    label_at = _column_label if mode == "columns" else _metre_label
    open_names = [name for name in PRESENT if name == "Background" or len(pools[name])]
    attempts = 0
    while attempts < per_class * 60 and any(len(buckets[name]) < per_class for name in open_names):
        attempts += 1
        waiting = [name for name in open_names if len(buckets[name]) < per_class]
        event_waiting = [name for name in waiting if name != "Background"]
        if "Background" in waiting and (not event_waiting or rng.random() < 0.5):
            want = "Background"
        else:
            want = event_waiting[int(rng.integers(0, len(event_waiting)))]
        if want == "Background":
            t0 = int(rng.integers(0, nt - TIME_CROP + 1))
            if mode == "columns":
                x0 = int(rng.integers(0, nx - SPACE_CROP + 1))
            else:
                x0 = int(rng.integers(0, nx))
            if len(pools["nuisance"]) and rng.random() < 0.5:
                pixel = pools["nuisance"][int(rng.integers(0, len(pools["nuisance"])))]
                t0 = int(np.clip(int(pixel[0]) - int(rng.integers(0, TIME_CROP)), 0, nt - TIME_CROP))
                if mode == "columns":
                    x0 = int(np.clip(int(pixel[1]) - int(rng.integers(0, SPACE_CROP)), 0, nx - SPACE_CROP))
                else:
                    x0 = int(np.clip(int(pixel[1]), 0, nx - 1))
        else:
            pixel = pools[want][int(rng.integers(0, len(pools[want])))]
            t0 = int(np.clip(int(pixel[0]) - int(rng.integers(0, TIME_CROP)), 0, nt - TIME_CROP))
            if mode == "columns":
                x0 = int(np.clip(int(pixel[1]) - int(rng.integers(0, SPACE_CROP)), 0, nx - SPACE_CROP))
            else:
                x0 = int(np.clip(int(pixel[1]) - int(rng.integers(0, 8)), 0, nx - 1))
        label = label_at(mask, valid, x, t0, x0)
        if label is None or len(buckets[label]) >= per_class:
            continue
        buckets[label].append((t0, x0))
    return {
        name: np.asarray(rows, dtype=np.int64).reshape(-1, 2) if rows else np.zeros((0, 2), dtype=np.int64)
        for name, rows in buckets.items()
    }


def _stack_origins(grouped: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    names = []
    for name in PRESENT:
        rows = grouped[name]
        if len(rows) == 0:
            continue
        parts.append(rows)
        names.append(np.full(len(rows), LABELS.index(name), dtype=np.int64))
    if not parts:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(parts), np.concatenate(names)


def _physical_rows(field, valid, x, times, origins: np.ndarray) -> np.ndarray:
    out = np.full((len(origins), len(PHYSICAL_FEATURES)), np.nan, dtype=np.float64)
    for index, (t0, x0) in enumerate(origins):
        t0 = int(t0)
        x0 = int(x0)
        sl = (slice(t0, t0 + TIME_CROP), slice(x0, x0 + SPACE_CROP))
        try:
            out[index] = shape_features(
                field[sl], valid[sl], x[x0:x0 + SPACE_CROP], np.asarray(times[t0:t0 + TIME_CROP], dtype=np.float64),
            )
        except ValueError:
            continue
    return out


def _feature_rows(field, valid, x, origins: np.ndarray) -> dict[str, np.ndarray]:
    rows = {name: table(field, valid, origins) for name, table in FEATURE_TABLES.items()}
    rows["plateau"] = plateau_table(field, valid, origins, x)
    return rows


def _denoise_norm(noisy, clean, valid, sigma) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract the per-column time median, then divide by sigma. Invalid cells stay 0."""
    sig = np.maximum(np.asarray(sigma, dtype=np.float32), 1e-3)
    base = np.nanmedian(np.where(valid, noisy, np.nan), axis=0)
    base = np.where(np.isfinite(base), base, 0.0).astype(np.float32)
    base_grid = np.broadcast_to(base, noisy.shape)
    sig_grid = np.broadcast_to(sig, noisy.shape)
    xin = np.zeros(noisy.shape, dtype=np.float32)
    yin = np.zeros(noisy.shape, dtype=np.float32)
    clean32 = np.asarray(clean, dtype=np.float32)
    xin[valid] = ((noisy[valid] - base_grid[valid]) / sig_grid[valid]).astype(np.float32)
    yin[valid] = ((clean32[valid] - base_grid[valid]) / sig_grid[valid]).astype(np.float32)
    return xin, yin, valid.astype(np.float32)


def _row_norm(noisy_row, clean_row, valid, sigma) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One trace: subtract its own valid median, then divide by sigma."""
    sig = np.maximum(np.asarray(sigma, dtype=np.float32), 1e-3)
    base = float(np.median(noisy_row[valid])) if np.any(valid) else 0.0
    xin = np.zeros(noisy_row.shape, dtype=np.float32)
    yin = np.zeros(clean_row.shape, dtype=np.float32)
    xin[valid] = ((noisy_row[valid] - base) / sig[valid]).astype(np.float32)
    yin[valid] = ((np.asarray(clean_row, dtype=np.float32)[valid] - base) / sig[valid]).astype(np.float32)
    return xin, yin, valid.astype(np.float32)


def sample_denoise(record: dict, rng: np.random.Generator, n_unet: int, n_conv: int, n_ae: int) -> dict[str, list]:
    noisy = record["noisy"]
    clean = record["clean"]
    valid = record["valid"]
    sigma = record["sigma_x"]
    mask = record["mask_event"]
    nt, nx = noisy.shape
    out = {"unet": [], "conv": [], "ae": []}
    tries = 0
    while len(out["unet"]) < n_unet and tries < n_unet * 40 and nt >= 32 and nx >= 128:
        tries += 1
        t0 = int(rng.integers(0, nt - 31))
        x0 = int(rng.integers(0, nx - 127))
        sl = (slice(t0, t0 + 32), slice(x0, x0 + 128))
        if float(valid[sl].mean()) < 0.9:
            continue
        xin, yin, ok = _denoise_norm(noisy[sl], clean[sl], valid[sl], sigma[x0:x0 + 128])
        out["unet"].append((xin, yin, ok))
    tries = 0
    while len(out["conv"]) < n_conv and tries < n_conv * 40 and nx >= 1024:
        tries += 1
        row = int(rng.integers(0, nt))
        x0 = int(rng.integers(0, nx - 1023))
        sl = (row, slice(x0, x0 + 1024))
        ok = valid[sl]
        if float(ok.mean()) < 0.7:
            continue
        xin, yin, mask_row = _row_norm(noisy[sl], clean[sl], ok, sigma[x0:x0 + 1024])
        out["conv"].append((xin, yin, mask_row))
    tries = 0
    while len(out["ae"]) < n_ae and tries < n_ae * 50 and nt >= 8 and nx >= 64:
        tries += 1
        t0 = int(rng.integers(0, nt - 7))
        x0 = int(rng.integers(0, nx - 63))
        sl = (slice(t0, t0 + 8), slice(x0, x0 + 64))
        if float(valid[sl].mean()) < 0.9 or int(mask[sl].max()) != 0:
            continue
        xin, yin, ok = _denoise_norm(noisy[sl], clean[sl], valid[sl], sigma[x0:x0 + 64])
        out["ae"].append((xin, yin, ok))
    return out


def build_cache(root: Path, split: str, dest: Path, rng: np.random.Generator, *, per_class: int, cap: int, limit: int | None) -> dict:
    """Sample windows and write numpy caches. Test sequences are not read."""
    if split == "test":
        raise ValueError("the test split is not a training cache")
    dest.mkdir(parents=True, exist_ok=True)
    class_patch: dict[str, list] = {name: [] for name in PRESENT}
    class_wide: dict[str, list] = {name: [] for name in PRESENT}
    class_feat: dict[str, dict[str, list]] = {name: {} for name in PRESENT}
    class_physical: dict[str, list] = {name: [] for name in PRESENT}
    res_x: dict[str, list] = {name: [] for name in PRESENT}
    denoise = {"unet_x": [], "unet_y": [], "unet_m": [], "conv_x": [], "conv_y": [], "conv_m": [], "ae_x": [], "ae_y": [], "ae_m": []}
    seen = 0
    for seq_id, record in iter_split(root, split):
        if limit is not None and seen >= limit:
            break
        seen += 1
        field = np.asarray(record["noisy"], dtype=np.float32)
        valid = record["valid"]
        grouped = sample_windows(record, rng, per_class, "columns")
        origins, target = _stack_origins(grouped)
        if len(origins):
            patch = deviation_crops(field, valid, origins)
            wide = wide_context_crops(field, valid, origins)
            features = _feature_rows(field, valid, record["x"], origins)
            physical = _physical_rows(field, valid, record["x"], record["t_unix"], origins)
            for name in PRESENT:
                sel = target == LABELS.index(name)
                if not np.any(sel):
                    continue
                class_patch[name].append(patch[sel])
                class_wide[name].append(wide[sel])
                class_physical[name].append(physical[sel])
                for key, value in features.items():
                    class_feat[name].setdefault(key, []).append(value[sel])
        residual_grouped = sample_windows(record, rng, per_class, "metres")
        residual_origins, residual_target = _stack_origins(residual_grouped)
        if len(residual_origins):
            images = meter_crops(field, valid, record["x"], residual_origins)
            for name in PRESENT:
                sel = residual_target == LABELS.index(name)
                if np.any(sel):
                    res_x[name].append(images[sel])
        crops = sample_denoise(record, rng, 8, 8, 4)
        for xin, yin, ok in crops["unet"]:
            denoise["unet_x"].append(xin)
            denoise["unet_y"].append(yin)
            denoise["unet_m"].append(ok)
        for xin, yin, ok in crops["conv"]:
            denoise["conv_x"].append(xin)
            denoise["conv_y"].append(yin)
            denoise["conv_m"].append(ok)
        for xin, yin, ok in crops["ae"]:
            denoise["ae_x"].append(xin)
            denoise["ae_y"].append(yin)
            denoise["ae_m"].append(ok)
        counts = {name: int(sum(len(part) for part in class_patch[name])) for name in PRESENT}
        print(f"cache {split} {seen} {seq_id} {counts}", flush=True)
    saved = {"split": split, "sequences": seen, "classes": {}, "labels": list(LABELS)}
    patch_parts = []
    wide_parts = []
    y_parts = []
    physical_parts = []
    feat_parts: dict[str, list] = {}
    for name in PRESENT:
        if not class_patch[name]:
            saved["classes"][name] = 0
            continue
        stacked_patch = np.concatenate(class_patch[name], axis=0)
        take = min(cap, len(stacked_patch))
        pick = rng.choice(len(stacked_patch), size=take, replace=False)
        patch_parts.append(stacked_patch[pick])
        wide_parts.append(np.concatenate(class_wide[name], axis=0)[pick])
        y_parts.append(np.full(take, LABELS.index(name), dtype=np.int64))
        physical_parts.append(np.concatenate(class_physical[name], axis=0)[pick])
        for key, parts in class_feat[name].items():
            feat_parts.setdefault(key, []).append(np.concatenate(parts, axis=0)[pick])
        saved["classes"][name] = int(take)
    if not patch_parts:
        raise RuntimeError(f"no classifier windows in {split}")
    order = rng.permutation(sum(len(part) for part in y_parts))
    patch = np.concatenate(patch_parts, axis=0)[order]
    wide = np.concatenate(wide_parts, axis=0)[order]
    target = np.concatenate(y_parts, axis=0)[order]
    physical = np.concatenate(physical_parts, axis=0)[order]
    np.save(dest / "patch.npy", patch)
    np.save(dest / "wide.npy", wide)
    np.save(dest / "y.npy", target)
    np.save(dest / "physical.npy", physical)
    for key, parts in feat_parts.items():
        np.save(dest / f"feat_{key}.npy", np.concatenate(parts, axis=0)[order])
    res_parts = []
    res_y = []
    for name in PRESENT:
        if not res_x[name]:
            continue
        stacked = np.concatenate(res_x[name], axis=0)
        take = min(cap, len(stacked))
        pick = rng.choice(len(stacked), size=take, replace=False)
        res_parts.append(stacked[pick])
        res_y.append(np.full(take, LABELS.index(name), dtype=np.int64))
    if res_parts:
        order = rng.permutation(sum(len(part) for part in res_y))
        np.save(dest / "residual.npy", np.concatenate(res_parts, axis=0)[order])
        np.save(dest / "residual_y.npy", np.concatenate(res_y, axis=0)[order])
    denoise_cap = {"unet": cap, "conv": cap, "ae": max(1, cap // 2)}
    for kind, limit_n in denoise_cap.items():
        xs = denoise[f"{kind}_x"]
        if not xs:
            continue
        stacked_x = np.stack(xs)
        take = min(limit_n, len(stacked_x))
        pick = rng.choice(len(stacked_x), size=take, replace=False)
        np.save(dest / f"{kind}_x.npy", stacked_x[pick])
        np.save(dest / f"{kind}_y.npy", np.stack(denoise[f"{kind}_y"])[pick])
        np.save(dest / f"{kind}_m.npy", np.stack(denoise[f"{kind}_m"])[pick])
        saved[kind] = int(take)
    (dest / "ready.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print(f"cache {split} ready {saved}", flush=True)
    return saved


def load_ready(dest: Path) -> dict:
    return json.loads((dest / "ready.json").read_text(encoding="utf-8"))
