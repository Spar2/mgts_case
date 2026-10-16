"""Inventory of research models and their seed recovery.

The function reads weight files and calls the forward that already ships.
It does not fit, it does not write weights, and it does not set
anomaly_status. A fixed scan window is not an expert rectangle.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from fiber.classify.rules import LABELS

_ROOT = Path(__file__).resolve().parents[1]

# stem -> (family, model name, kind)
# kind "score" calls the shipped card forward. Other kinds are reasons.
_BY_STEM = {
    "research_residual_cnn.pt": ("research_residual_cnn", "research residual CNN", "residual_cnn"),
    "research_context_cnn.pt": ("research_context_cnn", "research context CNN", "context_cnn"),
    "physical_shape_v1.json": ("physical_shape", "physical shape", "physical"),
    "gadf_efficientnet.pt": ("gadf_efficientnet", "GADF EfficientNet", "gadf"),
    "zone_xgb.json": ("zone_xgb", "zone XGB", "zone_xgb"),
    "column_graph.pt": ("column_graph", "column graph", "wide"),
    "column_transformer.pt": ("column_transformer", "column transformer", "wide"),
    "wide_context.pt": ("wide_context", "wide context CNN", "wide"),
    "depth_cnn.pt": ("depth_cnn", "depth CNN", "wide"),
    "gradient_fusion.pt": ("gradient_fusion", "gradient fusion", "fusion"),
    "patch_scanner.pt": ("patch_scanner", "patch scanner", "patch"),
    "patch_profile.pt": ("patch_profile", "patch profile", "patch"),
    "patch_persistent.pt": ("patch_persistent", "patch persistent", "patch"),
    "patch_classifier.pt": ("patch_classifier", "patch classifier", "patch"),
    "residual_net.pt": ("residual_net", "residual net", "metre"),
    "residual_net_box.pt": ("residual_net_box", "residual net box", "metre"),
    "context_logistic.json": ("context_logistic", "context logistic", "window_features"),
    "context_lgb.txt": ("context_lgb", "LightGBM", "window_features"),
    "context_agree.txt": ("context_agree", "context agree", "window_features"),
    "bulk.json": ("bulk", "bulk logistic", "window_features"),
    "contrast.json": ("contrast", "contrast logistic", "window_features"),
    "excess.json": ("excess", "excess logistic", "window_features"),
    "neighbor.json": ("neighbor", "neighbor logistic", "window_features"),
    "persist.json": ("persist", "persist logistic", "window_features"),
    "plateau.json": ("plateau", "plateau logistic", "window_features"),
    "quantile.json": ("quantile", "quantile logistic", "window_features"),
    "regime.json": ("regime", "regime logistic", "window_features"),
    "stability.json": ("stability", "stability logistic", "window_features"),
    "blindspot_x.pt": ("blindspot_x", "blindspot denoiser", "field"),
    "conv1d_n2self.pt": ("conv1d_n2self", "conv1d Noise2Self", "field"),
    "conv1d_supervised.pt": ("conv1d_supervised", "conv1d supervised denoiser", "field"),
    "conv_ae_bg.pt": ("conv_ae", "conv autoencoder", "field"),
    "dilated1d_supervised.pt": ("dilated1d_supervised", "dilated1d denoiser", "field"),
    "n2n_tsr.pt": ("n2n_tsr", "noise2noise TSR", "field"),
    "unet2d_n2self.pt": ("unet2d_n2self", "unet2d Noise2Self", "field"),
    "unet2d_supervised.pt": ("unet2d_supervised", "unet2d supervised denoiser", "field"),
    "vmd_dilated1d.pt": ("vmd_dilated1d", "VMD dilated denoiser", "field"),
    "zsn2n.pt": ("zsn2n", "zero-shot noise2noise", "field"),
    "seg_unet.pt": ("seg_unet", "segmentation U-Net", "label_map"),
    "yolo_waterfall.pt": ("yolo_waterfall", "YOLO waterfall", "boxes"),
    "1.pt": ("ml_test_cnn", "ml-test ConvAutoencoder 1", "score_map"),
    "2.pt": ("ml_test_cnn", "ml-test ConvAutoencoder 2", "score_map"),
    "3.pt": ("ml_test_cnn", "ml-test ConvAutoencoder 3", "score_map"),
    "4.pt": ("ml_test_cnn", "ml-test ConvAutoencoder 4", "score_map"),
    "5.pt": ("ml_test_cnn", "ml-test ConvAutoencoder 5", "score_map"),
    "6.pt": ("ml_test_cnn", "ml-test ConvAutoencoder 6", "score_map"),
    "real_masked_ridge.json": ("real_masked_ridge", "real masked ridge", "reconstruction"),
}

_REASONS = {
    "wide": (
        "Вход, который уже есть у модели, не принимает эти карточки: "
        "он берёт фиксированный кроп скана 28×120, а прямоугольник expert seed им не является."
    ),
    "fusion": (
        "Вход, который уже есть у модели, не принимает эти карточки: "
        "он берёт кроп 28×40 с тремя каналами, а прямоугольник expert seed им не является."
    ),
    "patch": (
        "Вход, который уже есть у модели, не принимает эти карточки: "
        "он берёт кроп отклонения 28×40 от начала окна скана, а прямоугольник expert seed им не является."
    ),
    "metre": (
        "Вход, который уже есть у модели, не принимает эти карточки: "
        "он берёт остаточный кроп 28×64 м от начала окна, а прямоугольник expert seed им не является."
    ),
    "window_features": (
        "Вход, который уже есть у модели, не принимает эти карточки: "
        "признаки считаются на окне скана 28×40, а прямоугольник expert seed им не является."
    ),
    "field": "Прямой проход отдаёт поле, а не одно из шести имён класса.",
    "label_map": "Прямой проход отдаёт карту меток, а не одно из шести имён класса.",
    "boxes": "Прямой проход отдаёт detection box, а не одно из шести имён класса.",
    "score_map": (
        "Прямой проход отдаёт карту reconstruction error, а не одно из шести имён класса. "
        "Вход, который уже есть у модели, — внешний CV residual патчами 32×32, а не прямоугольник expert seed."
    ),
    "reconstruction": (
        "Прямой проход отдаёт восстановленную температуру по соседним столбцам, "
        "а не одно из шести имён класса."
    ),
    "zones": (
        "Вход, который уже есть у скорера, не принимает эти карточки: "
        "он подписывает зоны детектора, а не прямоугольник expert seed. Файла весов нет."
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_OUTSIDE_WEIGHTS = (
    "research/ml-test-review/weights/1.pt",
    "research/ml-test-review/weights/2.pt",
    "research/ml-test-review/weights/3.pt",
    "research/ml-test-review/weights/4.pt",
    "research/ml-test-review/weights/5.pt",
    "research/ml-test-review/weights/6.pt",
    "research/real-data-v2/real_masked_ridge.json",
)


def _weight_files(root: Path) -> list[Path]:
    base = root / "models" / "weights"
    files = [
        path for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".json", ".txt"}
    ]
    for relative in _OUTSIDE_WEIGHTS:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"weight file missing: {relative}")
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _jobs(root: Path) -> tuple[list[dict], list[str]]:
    from fiber.weak_labels import EXPERT_SEED_BY_ID, EXPERT_SEEDS, EXPERT_WEIGHT, map_expert_class

    jobs = []
    for seed in EXPERT_SEEDS:
        if seed["id"] == "next-0003" and seed["class_name"] != "Local_Heating":
            raise ValueError("next-0003 must stay Local_Heating")
        if seed["class_name"] not in LABELS:
            raise ValueError(f"{seed['id']} left the six class names")
        jobs.append({
            "id": str(seed["id"]),
            "record": str(seed["record"]),
            "rows": (int(seed["rows_half_open"][0]), int(seed["rows_half_open"][1])),
            "x_span": (float(seed["x_m"][0]), float(seed["x_m"][1])),
            "expert_class": str(seed["class_name"]),
            "group": "previous",
        })
    if len(jobs) != 15:
        raise ValueError(f"previous expert seeds are {len(jobs)}, not 15")
    ask_path = root / "docs" / "evidence" / "expert_seeds_ask.jsonl"
    seen = set()
    ask = []
    for line in ask_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seed_id = str(row["id"])
        if seed_id in seen or seed_id in EXPERT_SEED_BY_ID or seed_id.startswith("gap-"):
            raise ValueError(f"ask seed id is not a new expert seed: {seed_id}")
        if float(row["weight"]) < float(EXPERT_WEIGHT) or row.get("expert_confirmed") is not True:
            raise ValueError(f"ask seed {seed_id} is not a weight-10 expert seed")
        expert = str(row["class_name"])
        mapped = map_expert_class(str(row["presence"]), str(row.get("recorded_shape") or ""))
        if expert not in LABELS or mapped != expert:
            raise ValueError(f"ask seed {seed_id} class drifted")
        if str(row["presence"]) == "unclear":
            raise ValueError(f"unclear ask seed was stored: {seed_id}")
        seen.add(seed_id)
        ask.append({
            "id": seed_id,
            "record": str(row["record"]),
            "rows": (int(row["rows_half_open"][0]), int(row["rows_half_open"][1])),
            "x_span": (float(row["x_m"][0]), float(row["x_m"][1])),
            "expert_class": expert,
            "group": "ask",
        })
    if len(ask) != 110:
        raise ValueError(f"ask expert seeds are {len(ask)}, not 110")
    unclear = []
    answer_path = root / "docs" / "evidence" / "zakhar_ask_shortlist_answers.jsonl"
    for line in answer_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("presence")) == "unclear":
            unclear.append(str(row["id"]))
    if len(unclear) != 40:
        raise ValueError(f"unclear answers are {len(unclear)}, not 40")
    scored = {job["id"] for job in jobs + ask}
    if scored.intersection(unclear):
        raise ValueError("an unclear answer entered the recovery cards")
    return jobs + ask, unclear


def _records(root: Path, jobs: list[dict]) -> dict:
    from fiber.io import Record
    from fiber.pipeline import analyze_field

    names = []
    for job in jobs:
        if job["record"] not in names:
            names.append(job["record"])
    loaded = {}
    for name in names:
        record = Record.load(root / "artifacts" / "records" / name)
        print(f"inventory record {name}", flush=True)
        found = analyze_field(
            record.raw_c,
            record.x_m,
            record.time_s,
            valid=record.valid,
            time_iso=record.time_iso,
        )
        manifest = found["manifest"]
        config = manifest["config"]
        if (
            manifest["detector"] != "spatial_z_halo"
            or manifest["clean"] != "median_x3"
            or float(config["threshold_z"]) != 3.5
            or manifest["trained_classifier_operational"] is not False
        ):
            raise RuntimeError("analyze_field is not the frozen operational pair")
        loaded[name] = {"record": record, "found": found}
    return loaded


def _box(job: dict, record) -> tuple[int, int, int, int]:
    from fiber.research_cnn import meter_columns

    t0, t1 = job["rows"]
    c0, c1 = meter_columns(record.x_m, job["x_span"][0], job["x_span"][1])
    if t1 <= t0 or c1 <= c0:
        raise ValueError(f"{job['id']} rectangle is empty")
    return t0, t1, c0, c1


def _class_index(logits) -> str:
    import numpy as np

    scores = np.asarray(logits).reshape(-1)
    if scores.shape != (len(LABELS),):
        raise ValueError("forward did not return six class scores")
    name = LABELS[int(scores.argmax())]
    if name not in LABELS:
        raise ValueError("forward left the six class names")
    return name


def _score_residual(path: Path, jobs: list[dict], loaded: dict) -> list[str]:
    import torch

    from fiber.research_cnn import load_research_cnn, residual_crop

    model, _blob, device = load_research_cnn(path)
    names = []
    with torch.inference_mode():
        for job in jobs:
            pack = loaded[job["record"]]
            record = pack["record"]
            found = pack["found"]
            t0, t1, c0, c1 = _box(job, record)
            crop = residual_crop(
                found["clean"], found["score_available"], t0, t1, c0, c1,
            )
            if crop is None:
                raise ValueError(f"{job['id']} residual crop is empty")
            logits = model(torch.from_numpy(crop[None, None]).to(device))
            names.append(_class_index(logits[0].detach().cpu().numpy()))
    return names


def _score_context(path: Path, jobs: list[dict], loaded: dict) -> list[str]:
    import numpy as np
    import torch

    from fiber.research_cnn import load_research_context_cnn, read_context_crop

    model, _blob, device = load_research_context_cnn(path)
    names = []
    with torch.inference_mode():
        for job in jobs:
            pack = loaded[job["record"]]
            record = pack["record"]
            found = pack["found"]
            t0, t1, _c0, _c1 = _box(job, record)
            crop, vector, measured = read_context_crop(
                found, record.x_m, record.time_s, (t0, t1), job["x_span"],
            )
            if crop is None:
                raise ValueError(f"{job['id']} context crop is empty")
            if measured["far_cable_median_c"] is None and np.isfinite(vector[1]):
                raise ValueError(f"{job['id']} absent far side was filled with a median")
            crops = torch.from_numpy(crop[None, None]).to(device)
            context = torch.from_numpy(vector[None].copy()).to(device)
            logits = model(crops, context)
            names.append(_class_index(logits[0].detach().cpu().numpy()))
    return names


def _score_physical(path: Path, jobs: list[dict], loaded: dict) -> list[str]:
    from fiber.classify.physical import classify_physical_shape, load_physical_model

    model = load_physical_model(path)
    names = []
    for job in jobs:
        record = loaded[job["record"]]["record"]
        t0, t1, c0, c1 = _box(job, record)
        called = classify_physical_shape(
            model,
            record.raw_c[t0:t1, c0:c1],
            record.valid[t0:t1, c0:c1],
            record.x_m[c0:c1],
            record.time_s[t0:t1],
        )
        name = called["shape_hypothesis"]
        if name not in LABELS:
            raise ValueError(f"{job['id']} physical forward left the six class names")
        names.append(name)
    return names


def _summary_module(root: Path):
    path = root / "scripts" / "train_summary_pipeline.py"
    spec = importlib.util.spec_from_file_location("inventory_summary_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score_gadf(path: Path, jobs: list[dict], loaded: dict, root: Path) -> list[str]:
    import timm

    summary = _summary_module(root)
    model = summary._load_torch(
        path,
        lambda _blob: timm.create_model(
            "efficientnet_b0", pretrained=False, in_chans=1, num_classes=6,
        ),
    )
    images = []
    for job in jobs:
        record = loaded[job["record"]]["record"]
        t0, t1, c0, c1 = _box(job, record)
        series = summary._spatial_mean(record.raw_c, record.valid, t0, t1 - 1, c0, c1 - 1)
        images.append(summary._gadf(series)[None, :])
    predicted = summary._predict_windows(model, images, 16)
    if len(predicted) != len(jobs):
        raise ValueError("GADF forward dropped a card")
    names = []
    for index in predicted:
        name = LABELS[int(index)]
        if name not in LABELS:
            raise ValueError("GADF forward left the six class names")
        names.append(name)
    return names


def _zone_cache(loaded: dict) -> dict:
    from fiber.anomaly.detect import spatial_z
    from fiber.denoise.dsp import median_x

    cache = {}
    for name, pack in loaded.items():
        record = pack["record"]
        cleaned = median_x(record.raw_c, record.valid, k=3)
        score, _bg, _scale = spatial_z(
            cleaned, record.valid, record.x_m,
            inner_m=8.0, outer_m=40.0, bin_m=2.0, floor_c=0.30,
        )
        cache[name] = (cleaned, score)
    return cache


def _score_zone_xgb(path: Path, jobs: list[dict], loaded: dict, root: Path) -> list[str]:
    import xgboost as xgb

    summary = _summary_module(root)
    model = xgb.Booster()
    model.load_model(str(path))
    cache = _zone_cache(loaded)
    features = []
    for job in jobs:
        record = loaded[job["record"]]["record"]
        t0, t1, c0, c1 = _box(job, record)
        cleaned, score = cache[job["record"]]
        features.append(summary._zone_features(
            cleaned, record.valid, score, record.x_m, record.time_s,
            t0, t1 - 1, c0, c1 - 1,
        ))
    import numpy as np

    predicted = summary._xgb_predict(model, np.stack(features))
    if len(predicted) != len(jobs):
        raise ValueError("zone XGB forward dropped a card")
    names = []
    for index in predicted:
        name = LABELS[int(index)]
        if name not in LABELS:
            raise ValueError("zone XGB forward left the six class names")
        names.append(name)
    return names


def _counts(jobs: list[dict], names: list[str]) -> dict:
    if len(names) != len(jobs):
        raise ValueError("a recovery card was dropped")
    previous = [job for job, name in zip(jobs, names, strict=True) if job["group"] == "previous"]
    ask = [job for job, name in zip(jobs, names, strict=True) if job["group"] == "ask"]
    previous_hits = sum(name == job["expert_class"] for job, name in zip(jobs, names, strict=True) if job["group"] == "previous")
    ask_hits = sum(name == job["expert_class"] for job, name in zip(jobs, names, strict=True) if job["group"] == "ask")
    if len(previous) != 15 or len(ask) != 110:
        raise ValueError("recovery denominators left 15 and 110")
    return {
        "previous_match_count": int(previous_hits),
        "previous_n_cards": 15,
        "previous_match_label": f"{previous_hits}/15",
        "ask_match_count": int(ask_hits),
        "ask_n_cards": 110,
        "ask_match_label": f"{ask_hits}/110",
    }


def _blank_counts() -> dict:
    return {
        "previous_match_count": None,
        "previous_n_cards": None,
        "previous_match_label": None,
        "ask_match_count": None,
        "ask_n_cards": None,
        "ask_match_label": None,
    }


def _row(path: Path | None, root: Path, family: str, model: str, kind: str) -> dict:
    relative = None if path is None else path.relative_to(root).as_posix()
    digest = None if path is None else _sha256(path)
    return {
        "model": model,
        "family": family,
        "weight_path": relative,
        "weights_sha256": digest,
        "kind": kind,
        "runnable": False,
        "reason": None if kind in {"residual_cnn", "context_cnn", "physical", "gadf", "zone_xgb"} else _REASONS[kind],
        **_blank_counts(),
    }


def inventory_models(root: Path | None = None) -> dict:
    """Score every research weight file that can name these cards.

    A second call reads the same files and returns the same fractions.
    Unclear answers are not cards. Weight bytes are only read.
    """
    base = _ROOT if root is None else Path(root)
    jobs, unclear = _jobs(base)
    rows = []
    for path in _weight_files(base):
        stem = path.name
        if stem not in _BY_STEM:
            raise ValueError(f"weight file has no inventory row: {path}")
        family, model, kind = _BY_STEM[stem]
        rows.append(_row(path, base, family, model, kind))
    rows.append(_row(
        None, base, "classify_zones", "classify_zones", "zones",
    ))
    print(
        f"inventory rows {len(rows)} files {sum(row['weight_path'] is not None for row in rows)}",
        flush=True,
    )
    needs_score = any(row["kind"] in {"residual_cnn", "context_cnn", "physical", "gadf", "zone_xgb"} for row in rows)
    loaded = _records(base, jobs) if needs_score else {}
    scorers = {
        "residual_cnn": _score_residual,
        "context_cnn": _score_context,
        "physical": _score_physical,
        "gadf": lambda path, jobs, loaded: _score_gadf(path, jobs, loaded, base),
        "zone_xgb": lambda path, jobs, loaded: _score_zone_xgb(path, jobs, loaded, base),
    }
    for row in rows:
        scorer = scorers.get(row["kind"])
        if scorer is None:
            continue
        path = base / row["weight_path"]
        print(f"inventory score {row['weight_path']}", flush=True)
        before = _sha256(path)
        try:
            names = scorer(path, jobs, loaded)
            counted = _counts(jobs, names)
        except Exception as exc:
            row["runnable"] = False
            row["reason"] = f"Запуск на этих карточках не прошёл: {exc}"
            print(f"inventory failed {row['weight_path']}: {exc}", flush=True)
            continue
        after = _sha256(path)
        if after != before or after != row["weights_sha256"]:
            raise RuntimeError(f"scoring changed weight bytes: {row['weight_path']}")
        row.update(counted)
        row["runnable"] = True
        row["reason"] = None
        print(
            f"inventory done {row['weight_path']} {row['previous_match_label']} {row['ask_match_label']}",
            flush=True,
        )
    by_sha: dict[str, list[dict]] = {}
    for row in rows:
        if row["weights_sha256"]:
            by_sha.setdefault(row["weights_sha256"], []).append(row)
    for group in by_sha.values():
        if len(group) < 2:
            continue
        labels = {(row["previous_match_label"], row["ask_match_label"], row["reason"]) for row in group}
        if len(labels) != 1:
            raise RuntimeError("weight files with the same sha256 disagreed")
    return {
        "schema": "model_inventory_v1",
        "serve": "research",
        "operational": False,
        "trained_classifier_operational": False,
        "sets_anomaly_status": False,
        "detector": "spatial_z_halo",
        "clean": "median_x3",
        "threshold_z": 3.5,
        "note": (
            "Recovery is class match on the previous 15 expert seeds and, separately, "
            "on the 110 weight-10 ask seeds. It is not operational accuracy. "
            "Context CNN ask recovery is in-sample. The other classifiers were fit on planted scenes. "
            "A miss stays a miss. presence=unclear is not a match or a miss."
        ),
        "previous_ids": [job["id"] for job in jobs if job["group"] == "previous"],
        "ask_ids": [job["id"] for job in jobs if job["group"] == "ask"],
        "unclear_ids": unclear,
        "rows": rows,
    }


def decision_section(table: dict) -> str:
    """One DECISIONS record rendered from the inventory table."""
    lines = [
        "## 2026-10-16 — инвентарь research-моделей",
        "",
        "Одна таблица [docs/evidence/model_inventory.json](evidence/model_inventory.json) по каждому файлу весов в `models/weights`, включая `experimental`, по шести чекпоинтам `research/ml-test-review/weights`, по `research/real-data-v2/real_masked_ridge.json`, и по кодовому скореру без файла весов. Оценка без переобучения. Класс эксперта сравнивается с классом, который отдаёт shipped forward. Прежние 15 expert seeds и 110 ask seeds веса 10 считаются отдельно. 40 ответов `presence=unclear` в знаменатели не входят: это не совпадение и не промах. Промах остаётся промахом. Дроби не являются операционной точностью. Восстановление context CNN на ask seeds in-sample: эти карточки были fit seeds. Остальные классификаторы учились на посаженных сценах. `serve=research`, operational false. `trained_classifier_operational` false. `sets_anomaly_status` false. Порог spatial_z и чистка median_x3 не менялись. Операционная пара остаётся `median_x3` и `spatial_z_halo` при `|Z| >= 3.5`. `next-0003` остаётся экспертом `Local_Heating`. Веса не переписывались.",
        "",
    ]
    seen: dict[str, int] = {}
    for row in table["rows"]:
        digest = row["weights_sha256"]
        if digest:
            seen[digest] = seen.get(digest, 0) + 1
    shared = [digest for digest, count in seen.items() if count > 1]
    if shared:
        lines.append("Одинаковый sha256 у двух путей — это одни и те же байты. Если обе строки посчитаны, дроби совпадают.")
        lines.append("")
    for row in table["rows"]:
        if row["weight_path"]:
            head = f"`{row['weight_path']}`: модель `{row['model']}`. sha256 `{row['weights_sha256']}`."
        else:
            head = f"код `{row['model']}`: файла весов нет."
        if row["runnable"]:
            tail = f" Прежние 15: `{row['previous_match_label']}`. Ask seeds: `{row['ask_match_label']}`."
        else:
            tail = f" Дроби нет. Причина: {row['reason']}"
        lines.append(f"- {head}{tail}")
    lines.append("")
    return "\n".join(lines)
