"""Train every haklolarp model on dts_synth_v1 and write experimental weights.

Published weight files are not opened for writing. The test split is scored
once after the weights are saved. Periodic_Thermal and Sensor_Fault have no
pixels in this set, so those two names are not taught.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.classify.agree_model import AGREE_FEATURE_NAMES, AGREE_LEARNING_RATE, AGREE_LEAVES, AGREE_MIN_CHILD_SAMPLES, AGREE_TREES
from fiber.classify.bulk import dump_bulk_model
from fiber.classify.context import FEATURE_NAMES, dump_context_model
from fiber.classify.contrast import dump_contrast_model
from fiber.classify.dts_synth import PRESENT, build_cache, iter_split, load_ready
from fiber.classify.excess import dump_excess_model
from fiber.classify.lgb_context import LGB_LEARNING_RATE, LGB_LEAVES, LGB_MIN_CHILD_SAMPLES, LGB_TREES
from fiber.classify.neighbor import dump_neighbor_model
from fiber.classify.persist import dump_persist_model
from fiber.classify.physical import FEATURE_NAMES as PHYSICAL_FEATURES
from fiber.classify.physical import SCHEMA
from fiber.classify.plant import SoftmaxSceneModel
from fiber.classify.plateau import dump_plateau_model
from fiber.classify.quantile import dump_quantile_model
from fiber.classify.regime import dump_regime_model
from fiber.classify.rules import BROAD_WIDTH_M, LABELS
from fiber.classify.stability import dump_stability_model
from fiber.models.patch_classifier import SPACE_CROP, TIME_CROP

EXPERIMENTAL = ROOT / "models" / "weights" / "experimental"
REPORT = ROOT / "artifacts" / "classify" / "experimental_dts_report.json"
RECIPES = (
    ("context", dump_context_model, "context_logistic.json"),
    ("contrast", dump_contrast_model, "contrast.json"),
    ("bulk", dump_bulk_model, "bulk.json"),
    ("neighbor", dump_neighbor_model, "neighbor.json"),
    ("excess", dump_excess_model, "excess.json"),
    ("persist", dump_persist_model, "persist.json"),
    ("plateau", dump_plateau_model, "plateau.json"),
    ("quantile", dump_quantile_model, "quantile.json"),
    ("regime", dump_regime_model, "regime.json"),
    ("stability", dump_stability_model, "stability.json"),
)


def macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for label in sorted(set(int(item) for item in truth)):
        tp = int(np.sum((truth == label) & (predicted == label)))
        fp = int(np.sum((truth != label) & (predicted == label)))
        fn = int(np.sum((truth == label) & (predicted != label)))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def _channel(crops: np.ndarray) -> np.ndarray:
    if crops.ndim == 3:
        return crops[:, None, :, :]
    return crops


def fit_cnn(factory, train_x, train_y, val_x, val_y, dest: Path, epochs: int, batch: int) -> dict:
    import torch
    from torch import nn

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("cuda is required")
    torch.manual_seed(0)
    model = factory().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_x = _channel(np.asarray(train_x, dtype=np.float32))
    val_x = _channel(np.asarray(val_x, dtype=np.float32))
    train_y = np.asarray(train_y, dtype=np.int64)
    val_y = np.asarray(val_y, dtype=np.int64)
    best_f1 = -1.0
    best_state = None
    stale = 0
    history = []
    rng = np.random.default_rng(0)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_y))
        losses = []
        step = max(1, min(batch, len(train_y)))
        for start in range(0, len(train_y) - step + 1, step):
            pick = order[start:start + step]
            xb = torch.from_numpy(train_x[pick]).to(device)
            yb = torch.from_numpy(train_y[pick]).to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        predicted = _predict_torch(model, val_x, batch, device)
        score = macro_f1(val_y, predicted)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else None, "val_macro_f1": score})
        print(f"train {dest.stem} epoch {epoch} loss {history[-1]['loss']:.4f} val_f1 {score:.4f}", flush=True)
        if score > best_f1 + 1e-4:
            best_f1 = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is None:
        raise RuntimeError(f"{dest.stem} produced no checkpoint")
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "n_class": len(LABELS)}, dest)
    return {"val_macro_f1": best_f1, "epochs_ran": history[-1]["epoch"], "history": history, "path": str(dest)}


def _predict_torch(model, crops: np.ndarray, batch: int, device: str) -> np.ndarray:
    import torch

    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(crops), batch):
            xb = torch.from_numpy(crops[start:start + batch]).to(device)
            pieces.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)


def _fit_softmax(features: np.ndarray, target: np.ndarray, dump, dest: Path) -> SoftmaxSceneModel:
    model = SoftmaxSceneModel()
    finite = np.isfinite(features).all(axis=1)
    model.fit(features[finite], target[finite], steps=500)
    payload = dump(model, {
        "experimental": True,
        "source": "dts_synth_v1",
        "classes_without_positives": ["Periodic_Thermal", "Sensor_Fault"],
    })
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return model


def _indexes(names: list[str]) -> np.ndarray:
    table = {name: index for index, name in enumerate(LABELS)}
    return np.asarray([table[name] for name in names], dtype=np.int64)


def _fit_booster(features, target, names, trees, leaves, lr, min_child, dest: Path):
    import lightgbm as lgb

    finite = np.isfinite(features).all(axis=1)
    dataset = lgb.Dataset(
        np.asarray(features[finite], dtype=np.float64),
        label=np.asarray(target[finite], dtype=np.int64),
        feature_name=list(names),
        free_raw_data=False,
    )
    booster = lgb.train(
        {
            "objective": "multiclass",
            "num_class": len(LABELS),
            "learning_rate": lr,
            "num_leaves": leaves,
            "min_child_samples": min_child,
            "verbosity": -1,
            "seed": 0,
            "deterministic": True,
            "num_threads": 1,
            "feature_pre_filter": False,
        },
        dataset,
        num_boost_round=trees,
    )
    booster.save_model(str(dest))
    return booster


def _booster_index(booster, features: np.ndarray) -> np.ndarray:
    finite = np.isfinite(features).all(axis=1)
    predicted = np.zeros(len(features), dtype=np.int64)
    if finite.any():
        prob = np.asarray(booster.predict(features[finite]))
        predicted[finite] = np.argmax(prob, axis=1)
    return predicted


def fit_denoise(kind: str, train: Path, val: Path, dest: Path, epochs: int, batch: int) -> dict:
    import torch
    from torch import nn

    from fiber.models.conv1d_denoise import Conv1dDenoiser
    from fiber.models.conv_ae import ConvBackgroundAE
    from fiber.models.unet2d_denoise import UNet2DDenoiser

    factory = {"conv": Conv1dDenoiser, "unet": UNet2DDenoiser, "ae": ConvBackgroundAE}[kind]
    train_x = np.load(train / f"{kind}_x.npy")
    train_y = np.load(train / f"{kind}_y.npy")
    train_m = np.load(train / f"{kind}_m.npy")
    val_x = np.load(val / f"{kind}_x.npy")
    val_y = np.load(val / f"{kind}_y.npy")
    val_m = np.load(val / f"{kind}_m.npy")
    device = "cuda"
    model = factory().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = None
    best_loss = 1e9
    stale = 0
    history = []

    def batches(x, y, mask):
        x = torch.from_numpy(x[:, None] if kind == "conv" else x[:, None]).to(device)
        y = torch.from_numpy(y[:, None] if kind == "conv" else y[:, None]).to(device)
        mask = torch.from_numpy(mask[:, None] if kind == "conv" else mask[:, None]).to(device)
        return x, y, mask

    def loss_of(pred, target, mask):
        err = nn.functional.smooth_l1_loss(pred, target, reduction="none")
        weight = mask.sum().clamp(min=1.0)
        return (err * mask).sum() / weight

    rng = np.random.default_rng(0)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_x))
        losses = []
        step = max(1, min(batch, len(train_x)))
        for start in range(0, len(train_x) - step + 1, step):
            pick = order[start:start + step]
            xb, yb, mb = batches(train_x[pick], train_y[pick], train_m[pick])
            opt.zero_grad()
            pred = model(xb)
            if kind == "ae":
                pred = pred[..., : yb.shape[-2], : yb.shape[-1]]
            loss = loss_of(pred, yb, mb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            xb, yb, mb = batches(val_x, val_y, val_m)
            pred = model(xb)
            if kind == "ae":
                pred = pred[..., : yb.shape[-2], : yb.shape[-1]]
            val_loss = float(loss_of(pred, yb, mb))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else None, "val_huber": val_loss})
        print(f"train {dest.stem} epoch {epoch} loss {history[-1]['loss']:.4f} val_huber {val_loss:.4f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    blob = {"state_dict": best, "mean_c": 0.0, "std_c": 1.0, "hidden": 32, "depth": 4, "kernel": 5}
    torch.save(blob, dest)
    return {"val_huber": best_loss, "epochs_ran": history[-1]["epoch"], "history": history, "path": str(dest)}


def _cache_dir(root: Path, split: str, cap: int, limit: int | None) -> Path:
    tag = "all" if limit is None else f"lim{limit}"
    return root / "cache" / f"{split}_{tag}_cap{cap}"


def ensure_cache(root: Path, split: str, cap: int, per_class: int, limit: int | None, rebuild: bool) -> Path:
    dest = _cache_dir(root, split, cap, limit)
    if dest.exists() and (dest / "ready.json").is_file() and not rebuild:
        print(f"cache {split} reuse {dest}", flush=True)
        return dest
    if dest.exists() and rebuild:
        for path in dest.glob("*"):
            path.unlink()
    seed = 0 if split == "train" else 1
    build_cache(
        root, split, dest, np.random.default_rng(seed), per_class=per_class, cap=cap, limit=limit,
    )
    return dest


def _load_xy(folder: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    y_name = "residual_y.npy" if stem == "residual" else "y.npy"
    return np.load(folder / f"{stem}.npy"), np.load(folder / y_name)


def train_all(train: Path, val: Path, epochs: int, denoise_epochs: int) -> dict:
    import torch

    from fiber.models.column_graph import ColumnGraph
    from fiber.models.column_transformer import ColumnTransformer
    from fiber.models.depth_cnn import DepthCNN
    from fiber.models.patch_classifier import PatchClassifier
    from fiber.models.residual_net import ResidualNet

    if not torch.cuda.is_available():
        raise RuntimeError("cuda is required")
    print(f"device {torch.cuda.get_device_name(0)}", flush=True)
    EXPERIMENTAL.mkdir(parents=True, exist_ok=True)
    report = {}
    nets = (
        ("patch", PatchClassifier, "patch", "patch_classifier.pt", 128),
        ("wide", PatchClassifier, "wide", "wide_context.pt", 64),
        ("depth", DepthCNN, "wide", "depth_cnn.pt", 64),
        ("graph", ColumnGraph, "wide", "column_graph.pt", 64),
        ("transformer", ColumnTransformer, "wide", "column_transformer.pt", 32),
        ("residual", ResidualNet, "residual", "residual_net.pt", 64),
    )
    for name, factory, stem, filename, batch in nets:
        x, y = _load_xy(train, stem)
        xv, yv = _load_xy(val, stem)
        report[name] = fit_cnn(factory, x, y, xv, yv, EXPERIMENTAL / filename, epochs, batch)
    train_y = np.load(train / "y.npy")
    val_y = np.load(val / "y.npy")
    for key, dump, filename in RECIPES:
        features = np.load(train / f"feat_{key}.npy")
        val_features = np.load(val / f"feat_{key}.npy")
        model = _fit_softmax(features, train_y, dump, EXPERIMENTAL / filename)
        predicted = _indexes(model.predict(val_features))
        score = macro_f1(val_y, predicted)
        report[filename] = {"val_macro_f1": score, "path": str(EXPERIMENTAL / filename)}
        print(f"train {filename} val_f1 {score:.4f}", flush=True)
    context = np.load(train / "feat_context.npy")
    context_val = np.load(val / "feat_context.npy")
    agree = np.load(train / "feat_agree.npy")
    agree_val = np.load(val / "feat_agree.npy")
    lgb_model = _fit_booster(
        context, train_y, FEATURE_NAMES, LGB_TREES, LGB_LEAVES, LGB_LEARNING_RATE, LGB_MIN_CHILD_SAMPLES,
        EXPERIMENTAL / "context_lgb.txt",
    )
    lgb_score = macro_f1(val_y, _booster_index(lgb_model, context_val))
    report["context_lgb.txt"] = {"val_macro_f1": lgb_score, "path": str(EXPERIMENTAL / "context_lgb.txt")}
    print(f"train context_lgb.txt val_f1 {lgb_score:.4f}", flush=True)
    agree_model = _fit_booster(
        agree, train_y, AGREE_FEATURE_NAMES, AGREE_TREES, AGREE_LEAVES, AGREE_LEARNING_RATE, AGREE_MIN_CHILD_SAMPLES,
        EXPERIMENTAL / "context_agree.txt",
    )
    agree_score = macro_f1(val_y, _booster_index(agree_model, agree_val))
    report["context_agree.txt"] = {"val_macro_f1": agree_score, "path": str(EXPERIMENTAL / "context_agree.txt")}
    print(f"train context_agree.txt val_f1 {agree_score:.4f}", flush=True)
    physical = _fit_physical(train, val)
    report["physical_shape_v1.json"] = physical
    for kind, filename, batch in (
        ("conv", "conv1d_supervised.pt", 64),
        ("unet", "unet2d_supervised.pt", 32),
        ("ae", "conv_ae_bg.pt", 64),
    ):
        report[filename] = fit_denoise(kind, train, val, EXPERIMENTAL / filename, denoise_epochs, batch)
    report["omitted"] = {
        "Periodic_Thermal": "dts_synth_v1 has no pixels for this name",
        "Sensor_Fault": "dts_synth_v1 has no pixels for this name",
        "texture": "same five context features as context_logistic, so the file would be a copy",
        "room": "same five context features as context_logistic, so the file would be a copy",
        "patch_scanner_profile_persistent": "same PatchClassifier as patch_classifier.pt",
        "ml_test_cnn": "per-file audit nets from another repository",
    }
    return report


def _fit_physical(train: Path, val: Path) -> dict:
    features = np.load(train / "physical.npy")
    target = np.load(train / "y.npy")
    finite = np.isfinite(features).all(axis=1)
    model = SoftmaxSceneModel()
    model.fit(features[finite], target[finite], steps=500)
    payload = {
        "schema": SCHEMA,
        "features": list(PHYSICAL_FEATURES),
        "labels": list(LABELS),
        "mu": model.mu.tolist(),
        "sig": model.sig.tolist(),
        "weight": model.weight.tolist(),
        "bias": model.bias.tolist(),
        "threshold": 0.0,
        "serve": "research_offline",
        "expert_confirmed": False,
        "experimental": True,
        "source": "dts_synth_v1",
    }
    dest = EXPERIMENTAL / "physical_shape_v1.json"
    dest.write_text(json.dumps(payload), encoding="utf-8")
    val_x = np.load(val / "physical.npy")
    val_y = np.load(val / "y.npy")
    keep = np.isfinite(val_x).all(axis=1)
    predicted = np.zeros(len(val_y), dtype=np.int64)
    if keep.any():
        predicted[keep] = _indexes(model.predict(val_x[keep]))
    score = macro_f1(val_y[keep], predicted[keep]) if keep.any() else 0.0
    print(f"train physical_shape_v1.json val_f1 {score:.4f}", flush=True)
    return {"val_macro_f1": score, "path": str(dest), "rows": int(finite.sum())}


def _unix(text: str) -> float:
    raw = text.strip()
    if raw.endswith("+00"):
        raw = raw[:-3] + "+0000"
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S%z").timestamp()


def _box_name(row: dict) -> str | None:
    label = row["label"]
    if label == "tracing":
        return None
    if label == "hotspot":
        return "Local_Heating"
    if label in ("coldspot", "cooling"):
        return "Local_Cooling"
    if label in ("section_heating", "slow_warming"):
        width = float(row["dist_end_m"]) - float(row["dist_start_m"])
        return "Broad_Shift" if width > BROAD_WIDTH_M else "Local_Heating"
    return None


def _cover(t_unix, x, i0: int, i1: int, j0: int, j1: int) -> list[tuple[int, int]]:
    nt = len(t_unix)
    nx = len(x)
    if nt < TIME_CROP or nx < SPACE_CROP:
        return []
    t_lo = max(0, min(i0, nt - TIME_CROP))
    t_hi = min(max(i1, 0), nt - TIME_CROP)
    x_lo = max(0, min(j0, nx - SPACE_CROP))
    x_hi = min(max(j1, 0), nx - SPACE_CROP)
    if t_hi < t_lo:
        t_lo = t_hi
    if x_hi < x_lo:
        x_lo = x_hi
    t_starts = list(range(t_lo, t_hi + 1, 14)) or [t_lo]
    x_starts = list(range(x_lo, x_hi + 1, 10)) or [x_lo]
    if len(t_starts) * len(x_starts) > 36:
        t_starts = t_starts[:: max(1, len(t_starts) // 6)]
        x_starts = x_starts[:: max(1, len(x_starts) // 6)]
    center = (int(np.clip((i0 + i1) // 2, 0, nt - TIME_CROP)), int(np.clip((j0 + j1) // 2, 0, nx - SPACE_CROP)))
    origins = [(t0, x0) for t0 in t_starts for x0 in x_starts]
    if center not in origins:
        origins.append(center)
    return origins[:36]


def _load_models():
    import torch

    from fiber.models.column_graph import ColumnGraph
    from fiber.models.column_transformer import ColumnTransformer
    from fiber.models.depth_cnn import DepthCNN
    from fiber.models.patch_classifier import PatchClassifier
    from fiber.models.residual_net import ResidualNet

    device = "cuda"
    torch_models = {}
    for name, factory, filename in (
        ("patch", PatchClassifier, "patch_classifier.pt"),
        ("wide", PatchClassifier, "wide_context.pt"),
        ("depth", DepthCNN, "depth_cnn.pt"),
        ("graph", ColumnGraph, "column_graph.pt"),
        ("transformer", ColumnTransformer, "column_transformer.pt"),
        ("residual", ResidualNet, "residual_net.pt"),
    ):
        blob = torch.load(EXPERIMENTAL / filename, map_location=device, weights_only=True)
        model = factory()
        model.load_state_dict(blob["state_dict"])
        model.to(device)
        model.eval()
        torch_models[name] = model
    softmax = {}
    for key, _dump, filename in RECIPES:
        blob = json.loads((EXPERIMENTAL / filename).read_text(encoding="utf-8"))
        model = SoftmaxSceneModel()
        model.mu = np.asarray(blob["mu"], dtype=np.float64)
        model.sig = np.asarray(blob["sig"], dtype=np.float64)
        model.weight = np.asarray(blob["weight"], dtype=np.float64)
        model.bias = np.asarray(blob["bias"], dtype=np.float64)
        softmax[filename] = model
    physical_blob = json.loads((EXPERIMENTAL / "physical_shape_v1.json").read_text(encoding="utf-8"))
    physical = SoftmaxSceneModel()
    physical.mu = np.asarray(physical_blob["mu"], dtype=np.float64)
    physical.sig = np.asarray(physical_blob["sig"], dtype=np.float64)
    physical.weight = np.asarray(physical_blob["weight"], dtype=np.float64)
    physical.bias = np.asarray(physical_blob["bias"], dtype=np.float64)
    softmax["physical_shape_v1.json"] = physical
    import lightgbm as lgb

    boosters = {
        "context_lgb.txt": lgb.Booster(model_file=str(EXPERIMENTAL / "context_lgb.txt")),
        "context_agree.txt": lgb.Booster(model_file=str(EXPERIMENTAL / "context_agree.txt")),
    }
    return torch_models, softmax, boosters, device


def score_test(root: Path) -> dict:
    """One pass over test boxes. The numbers are not used to change weights."""
    from fiber.classify.context import wide_context_crops
    from fiber.classify.dts_synth import _feature_rows, _physical_rows
    from fiber.classify.residual import meter_crops
    from fiber.classify.scan import deviation_crops

    labels = list(csv.DictReader((root / "labels.csv").open(newline="", encoding="utf-8")))
    by_seq: dict[str, list] = {}
    for row in labels:
        if row["split"] == "test":
            by_seq.setdefault(row["seq_id"], []).append(row)
    torch_models, softmax, boosters, device = _load_models()
    names = list(torch_models) + list(softmax) + list(boosters)
    hits = {name: {} for name in names}
    difficulty = {name: {} for name in names}
    quiet_wrong = {name: 0 for name in names}
    quiet_n = 0
    nuisance_wrong = {name: 0 for name in names}
    nuisance_n = 0
    tracing = 0
    rng = np.random.default_rng(2)
    for seq_id, record in iter_split(root, "test"):
        rows = by_seq.get(seq_id, [])
        field = np.asarray(record["noisy"], dtype=np.float32)
        valid = record["valid"]
        x = record["x"]
        t_unix = record["t_unix"]
        jobs = []
        for row in rows:
            if row["kind"] == "event" and row["label"] == "tracing":
                tracing += 1
                continue
            mapped = _box_name(row) if row["kind"] == "event" else None
            if row["kind"] == "event" and mapped is None:
                continue
            i0 = int(np.searchsorted(t_unix, _unix(row["t_start"])))
            i1 = int(np.searchsorted(t_unix, _unix(row["t_end"]), side="right") - 1)
            j0 = int(np.searchsorted(x, float(row["dist_start_m"]) - 1.0))
            j1 = int(np.searchsorted(x, float(row["dist_end_m"]) + 1.0, side="right") - 1)
            i0 = max(0, i0 - 2)
            i1 = min(len(t_unix) - 1, max(i0, i1 + 2))
            j0 = min(max(0, j0), len(x) - 1)
            j1 = min(max(j0, j1), len(x) - 1)
            origins = _cover(t_unix, x, i0, i1, j0, j1)
            if not origins:
                continue
            jobs.append((origins, row["kind"], mapped, row.get("difficulty", "")))
        nt, nx = field.shape
        quiet = []
        tries = 0
        while len(quiet) < 12 and tries < 200 and nt >= TIME_CROP and nx >= SPACE_CROP:
            tries += 1
            t0 = int(rng.integers(0, nt - TIME_CROP + 1))
            x0 = int(rng.integers(0, nx - SPACE_CROP + 1))
            if int(record["mask_event"][t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP].max()) != 0:
                continue
            if float(valid[t0:t0 + TIME_CROP, x0:x0 + SPACE_CROP].mean()) < 0.5:
                continue
            quiet.append((t0, x0))
        all_origins = [pair for origins, _kind, _mapped, _diff in jobs for pair in origins]
        all_origins.extend(quiet)
        if not all_origins:
            continue
        origin_arr = np.asarray(all_origins, dtype=np.int64)
        patch = _channel(deviation_crops(field, valid, origin_arr))
        wide = _channel(wide_context_crops(field, valid, origin_arr))
        residual = meter_crops(field, valid, x, origin_arr)
        features = _feature_rows(field, valid, x, origin_arr)
        physical = _physical_rows(field, valid, x, t_unix, origin_arr)
        predicted = {
            "patch": _predict_torch(torch_models["patch"], patch, 128, device),
            "wide": _predict_torch(torch_models["wide"], wide, 64, device),
            "depth": _predict_torch(torch_models["depth"], wide, 64, device),
            "graph": _predict_torch(torch_models["graph"], wide, 64, device),
            "transformer": _predict_torch(torch_models["transformer"], wide, 32, device),
            "residual": _predict_torch(torch_models["residual"], residual, 64, device),
        }
        for filename, model in softmax.items():
            matrix = physical if filename.startswith("physical") else features["context" if filename.startswith("context_logistic") else filename.replace(".json", "")]
            if filename == "context_logistic.json":
                matrix = features["context"]
            keep = np.isfinite(matrix).all(axis=1)
            pred = np.zeros(len(origin_arr), dtype=np.int64)
            if keep.any():
                pred[keep] = _indexes(model.predict(matrix[keep]))
            predicted[filename] = pred
        predicted["context_lgb.txt"] = _booster_index(boosters["context_lgb.txt"], features["context"])
        predicted["context_agree.txt"] = _booster_index(boosters["context_agree.txt"], features["agree"])
        cursor = 0
        for origins, kind, mapped, level in jobs:
            span = len(origins)
            block = slice(cursor, cursor + span)
            cursor += span
            for name, pred in predicted.items():
                called = pred[block]
                if kind == "event":
                    index = LABELS.index(mapped)
                    hit = bool(np.any(called == index))
                    hits[name].setdefault(mapped, {"hit": 0, "total": 0})
                    hits[name][mapped]["total"] += 1
                    hits[name][mapped]["hit"] += int(hit)
                    difficulty[name].setdefault(level, {"hit": 0, "total": 0})
                    difficulty[name][level]["total"] += 1
                    difficulty[name][level]["hit"] += int(hit)
                else:
                    nuisance_wrong[name] += int(np.any(called != 0))
            if kind != "event":
                nuisance_n += 1
        quiet_pred = {name: pred[cursor:] for name, pred in predicted.items()}
        quiet_n += len(quiet)
        for name, pred in quiet_pred.items():
            quiet_wrong[name] += int(np.sum(pred != 0))
        print(f"test {seq_id} boxes {len(jobs)} quiet {len(quiet)}", flush=True)
    summary = {}
    for name in names:
        summary[name] = {
            "box_hit": hits[name],
            "by_difficulty": difficulty[name],
            "quiet_false_alarm": (quiet_wrong[name] / quiet_n) if quiet_n else None,
            "nuisance_called_event": (nuisance_wrong[name] / nuisance_n) if nuisance_n else None,
        }
    summary["tracing_boxes_excluded"] = tracing
    summary["quiet_windows"] = quiet_n
    summary["nuisance_boxes"] = nuisance_n
    return summary


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts" / "dts_synth_v1")
    parser.add_argument("--cap", type=int, default=1500)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--denoise-epochs", type=int, default=8)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--stage", choices=("all", "cache", "train", "test"), default="all")
    args = parser.parse_args()
    train = ensure_cache(args.root, "train", args.cap, args.per_class, args.limit, args.rebuild) if args.stage != "test" else _cache_dir(args.root, "train", args.cap, args.limit)
    val = ensure_cache(args.root, "val", args.cap, args.per_class, args.limit, args.rebuild) if args.stage != "test" else _cache_dir(args.root, "val", args.cap, args.limit)
    report = {
        "source": "dts_synth_v1",
        "present_classes": list(PRESENT),
        "train_cache": load_ready(train) if args.stage != "test" else {},
        "val_cache": load_ready(val) if args.stage != "test" else {},
    }
    if args.stage in ("all", "train"):
        report["models"] = train_all(train, val, args.epochs, args.denoise_epochs)
    if args.stage in ("all", "test"):
        report["test"] = score_test(args.root)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ABORT {exc}", flush=True)
        raise
