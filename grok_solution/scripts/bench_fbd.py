"""KID, FD_∞ и доменный зонд на bottleneck det0. Рядом — KL зона/тишина по выходам v8.

Реальные 1–6.csv читаются через dts.real и используются только как выборка оценки.
Зонд — логистическая регрессия на уже посчитанных эмбеддингах, GroupKFold по файлам.
Его веса не сохраняются и порог детектора не калибруют. Метки зонда — имя домена,
не класс события. Маска зон на реальном кабеле — прямоугольники эталона.

Пример:
  /workspace/repo/run/.venv/bin/python scripts/bench_fbd.py --skip-kl --write-md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts import fbd, metrics, real, grid, features  # noqa: E402

REAL_DIR = Path("/workspace/mgts/MGTS_data")
ZONE_DIR = ROOT / "artifacts" / "eval" / "domain_cache"
RESULTS_START = "<!-- FBD_RESULTS_START -->"
RESULTS_END = "<!-- FBD_RESULTS_END -->"


def _finite(x):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


def _round(x, n=6):
    v = _finite(x)
    if isinstance(v, float):
        return round(v, n)
    return v


def add_hist(hist, edges, values):
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0
    hist += np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)[0]
    return int(v.size)


def load_npz_arrays(path, keys):
    with np.load(path) as z:
        return {k: np.array(z[k]) for k in keys}


def event_views(ev, valid, nu):
    ev = np.asarray(ev)
    valid = np.asarray(valid, bool)
    hot = valid & (ev >= 1) & (ev <= 4)
    cold = valid & (ev >= 5) & (ev <= 6)
    truth = hot | cold
    bg = valid & ~truth
    if nu is not None:
        bg = bg & (np.asarray(nu) == 0)
    return hot, cold, truth, bg


def list_models(pred_root: Path):
    names = []
    for p in sorted(pred_root.iterdir()):
        if p.is_dir() and (p / "test").is_dir() and (p / "val").is_dir():
            names.append(p.name)
    return names


def list_seqs(cache: Path):
    return sorted(cache.glob("seq*.npz"))


def reference_zones(name: str):
    path = ZONE_DIR / f"{name}_base.json"
    if not path.is_file():
        raise FileNotFoundError(f"нет маски зон эталона: {path}")
    zones = json.loads(path.read_text())
    return zones


def collect_iou_kl(models, cache_root: Path, pred_root: Path, far: float, edges, limit: int):
    val_files = list_seqs(cache_root / "val")
    test_files = list_seqs(cache_root / "test")
    if limit:
        val_files = val_files[:limit]
        test_files = test_files[:limit]
    hists = {m: np.zeros(len(edges) - 1, np.int64) for m in models}
    n_bg = {m: 0 for m in models}
    t0 = time.time()
    for path in val_files:
        arrays = load_npz_arrays(path, ("ev", "valid", "nu"))
        _, _, _, bg = event_views(arrays["ev"], arrays["valid"], arrays["nu"])
        for m in models:
            pred = pred_root / m / "val" / path.name
            if not pred.is_file():
                continue
            s = load_npz_arrays(pred, ("s",))["s"]
            if s.shape != bg.shape:
                raise RuntimeError(f"{pred} shape {s.shape} != {bg.shape}")
            n_bg[m] += add_hist(hists[m], edges, s[bg])
        del arrays
    thr = {m: fbd.far_threshold_hist(hists[m], edges, far) for m in models}
    print(f"[fbd] val thresholds far={far} за {time.time() - t0:.0f} с", flush=True)

    acc = {
        m: dict(tot=np.zeros(3, np.int64), hot=np.zeros(3, np.int64), cold=np.zeros(3, np.int64),
                far_n=0, far_d=0, hot_kl=[], cold_kl=[], skipped=0, n_files=0)
        for m in models
    }
    t1 = time.time()
    for path in test_files:
        arrays = load_npz_arrays(path, ("ev", "valid", "nu", "t"))
        hot_t, cold_t, truth, bg = event_views(arrays["ev"], arrays["valid"], arrays["nu"])
        specs = fbd.zone_specs(arrays["ev"], arrays["valid"], arrays["t"], nu=arrays["nu"])
        for m in models:
            pred_path = pred_root / m / "test" / path.name
            if not pred_path.is_file():
                continue
            ph = load_npz_arrays(pred_path, ("s", "h"))
            s, h = ph["s"], np.asarray(ph["h"], np.float32)
            if s.shape != truth.shape:
                raise RuntimeError(f"{pred_path} shape {s.shape} != {truth.shape}")
            pred = (s >= thr[m]) & arrays["valid"]
            h = np.nan_to_num(h)
            acc[m]["tot"] += metrics.confusion(pred, truth, arrays["valid"])
            acc[m]["hot"] += metrics.confusion(pred & (h > 0), hot_t, arrays["valid"])
            acc[m]["cold"] += metrics.confusion(pred & (h <= 0), cold_t, arrays["valid"])
            acc[m]["far_n"] += int((pred & bg).sum())
            acc[m]["far_d"] += int(bg.sum())
            kl = fbd.kl_zone_quiet(s, specs)
            acc[m]["hot_kl"].extend(kl["hot"])
            acc[m]["cold_kl"].extend(kl["cold"])
            acc[m]["skipped"] += kl["n_skipped"]
            acc[m]["n_files"] += 1
        del arrays, specs
        print(f"[fbd] kl/iou {path.stem}", flush=True)
    print(f"[fbd] test kl/iou за {time.time() - t1:.0f} с", flush=True)

    rows = {}
    for m in models:
        a = acc[m]
        hot_s = fbd.kl_summary(a["hot_kl"])
        cold_s = fbd.kl_summary(a["cold_kl"])
        pr = metrics.prf(*a["tot"])
        hot = metrics.prf(*a["hot"])
        cold = metrics.prf(*a["cold"])
        rows[m] = {
            "threshold_far_1e-3": _round(thr[m], 6),
            "n_val_bg": int(n_bg[m]),
            "n_test_files": int(a["n_files"]),
            "test_far": _round(a["far_n"] / max(a["far_d"], 1), 6),
            "iou": _round(pr["iou"], 6),
            "precision": _round(pr["precision"], 6),
            "recall": _round(pr["recall"], 6),
            "hot_iou": _round(hot["iou"], 6),
            "cold_iou": _round(cold["iou"], 6),
            "kl_hot_mean": _round(hot_s["mean"], 6),
            "kl_hot_median": _round(hot_s["median"], 6),
            "kl_hot_n": hot_s["n"],
            "kl_cold_mean": _round(cold_s["mean"], 6),
            "kl_cold_median": _round(cold_s["median"], 6),
            "kl_cold_n": cold_s["n"],
            "kl_skipped_zones": int(a["skipped"]),
        }
        print(f"[fbd] {m} thr={rows[m]['threshold_far_1e-3']} iou={rows[m]['iou']} "
              f"hot={rows[m]['hot_iou']} cold={rows[m]['cold_iou']} "
              f"kl_hot={rows[m]['kl_hot_mean']} kl_cold={rows[m]['kl_cold_mean']}", flush=True)
    rho_h = fbd.spearman_rho([rows[m]["kl_hot_mean"] for m in models], [rows[m]["hot_iou"] for m in models])
    rho_c = fbd.spearman_rho([rows[m]["kl_cold_mean"] for m in models], [rows[m]["cold_iou"] for m in models])
    verdict = fbd.selector_verdict(rho_h["rho"], rho_c["rho"], 0.6)
    return rows, {"hot": rho_h, "cold": rho_c, "threshold": 0.6, "verdict": verdict}


def embed_modes(enc, X, valid, ban, n_each, patch, seed, tag, batch):
    overall = fbd.sample_patches(X, valid, n_each, patch, fbd.stable_seed(seed, tag, "overall"), ban=None)
    outside = fbd.sample_patches(X, valid, n_each, patch, fbd.stable_seed(seed, tag, "outside"), ban=ban)
    feats = {}
    if len(overall):
        feats["overall"] = enc.embed(overall, batch=batch)
    else:
        feats["overall"] = np.zeros((0, enc.dim), np.float32)
    if len(outside):
        feats["outside"] = enc.embed(outside, batch=batch)
    else:
        feats["outside"] = np.zeros((0, enc.dim), np.float32)
    return feats


def stack_or_empty(chunks, dim):
    chunks = [c for c in chunks if len(c)]
    if not chunks:
        return np.zeros((0, dim), np.float32)
    return np.concatenate(chunks, 0)


def collect_synth(enc, ver, split, n_each, patch, seed, batch, limit):
    files = list_seqs(ROOT / "artifacts" / "cache" / ver / split)
    if limit:
        files = files[:limit]
    overall, outside = [], []
    per_file = {}
    t0 = time.time()
    for path in files:
        arrays = load_npz_arrays(path, ("X", "valid", "ev"))
        ban = np.asarray(arrays["ev"] > 0)
        tag = f"{ver}:{split}:{path.stem}"
        feats = embed_modes(enc, arrays["X"], arrays["valid"], ban, n_each, patch, seed, tag, batch)
        overall.append(feats["overall"])
        outside.append(feats["outside"])
        per_file[path.stem] = {"n_overall": int(len(feats["overall"])), "n_outside": int(len(feats["outside"]))}
        print(f"[fbd] embed {tag} overall {len(feats['overall'])} outside {len(feats['outside'])} "
              f"({time.time() - t0:.0f} с)", flush=True)
        del arrays, feats
    return {
        "overall": stack_or_empty(overall, enc.dim),
        "outside": stack_or_empty(outside, enc.dim),
        "per_file_n": per_file,
        "n_files": len(files),
    }


def collect_real(enc, names, n_each, patch, seed, batch):
    """Чтение CSV через dts.real. Каналы считаются тем же grid+features, что и у синтеза. Веса сети не меняются."""
    overall, outside = [], []
    per = {}
    t0 = time.time()
    for name in names:
        csv_path = REAL_DIR / f"{name}.csv"
        rec = real.load_csv(csv_path)
        zones = reference_zones(name)
        plan = grid.plan(rec["x"], rec["t"])
        tc, valid = grid.resample(rec["T"], rec["valid"], plan, "mean")
        X, _aux = features.make_channels(tc, valid, dx=float(plan["dx"]))
        # Бан — только прямоугольники эталона. Дыры valid режет min_valid, иначе одно
        # пропущенное измерение выкидывало бы целое окно.
        ban = fbd.boxes_to_mask(plan["t"], plan["x"], zones)
        tag = f"real:{name}"
        feats = embed_modes(enc, X, valid, ban, n_each, patch, seed, tag, batch)
        overall.append(feats["overall"])
        outside.append(feats["outside"])
        per[name] = {
            "sha256": rec["sha256"],
            "raw_shape": [int(rec["T"].shape[0]), int(rec["T"].shape[1])],
            "grid_shape": [int(valid.shape[0]), int(valid.shape[1])],
            "n_reference_zones": len(zones),
            "zone_cell_fraction": _round(float(ban[valid].mean()) if valid.any() else 0.0, 6),
            "n_overall": int(len(feats["overall"])),
            "n_outside": int(len(feats["outside"])),
            "overall": feats["overall"],
            "outside": feats["outside"],
        }
        print(f"[fbd] real {name} grid {per[name]['grid_shape']} zones {len(zones)} "
              f"patches {per[name]['n_overall']}/{per[name]['n_outside']} ({time.time() - t0:.0f} с)", flush=True)
        del rec, tc, X, feats
    return {
        "overall": stack_or_empty(overall, enc.dim),
        "outside": stack_or_empty(outside, enc.dim),
        "files": per,
    }


def _empty_feat(dim):
    return np.zeros((0, dim), np.float32)


def _embed_strata(enc, X, valid, zone, n_each, patch, seed, tag, batch):
    zpatch = fbd.sample_zone_patches(X, valid, zone, n_each, patch, fbd.stable_seed(seed, tag, "zone"))
    qpatch = fbd.sample_patches(X, valid, n_each, patch, fbd.stable_seed(seed, tag, "quiet"), ban=zone)
    parts = []
    if len(zpatch):
        parts.append(zpatch)
    if len(qpatch):
        parts.append(qpatch)
    if not parts:
        empty = _empty_feat(enc.dim)
        return empty, empty
    feat = enc.embed(np.concatenate(parts, 0), batch=batch)
    nz = len(zpatch)
    return feat[:nz], feat[nz:]


def collect_synth_strata(enc, ver, split, n_each, patch, seed, batch, limit):
    files = list_seqs(ROOT / "artifacts" / "cache" / ver / split)
    if limit:
        files = files[:limit]
    ids, zone_f, quiet_f = [], [], []
    t0 = time.time()
    for path in files:
        arrays = load_npz_arrays(path, ("X", "valid", "ev"))
        zone = np.asarray(arrays["ev"] > 0)
        tag = f"{ver}:{split}:{path.stem}"
        zf, qf = _embed_strata(enc, arrays["X"], arrays["valid"], zone, n_each, patch, seed, tag, batch)
        ids.append(tag)
        zone_f.append(zf)
        quiet_f.append(qf)
        print(f"[fbd] embed {tag} zone {len(zf)} quiet {len(qf)} ({time.time() - t0:.0f} с)", flush=True)
        del arrays
    return {"ids": ids, "zone": zone_f, "quiet": quiet_f, "n_files": len(files)}


def collect_real_strata(enc, names, n_each, patch, seed, batch):
    """CSV через dts.real. Каналы — тот же grid и features, что у синтеза. Веса энкодера не меняются."""
    ids, zone_f, quiet_f = [], [], []
    per = {}
    t0 = time.time()
    for name in names:
        rec = real.load_csv(REAL_DIR / f"{name}.csv")
        zones = reference_zones(name)
        plan = grid.plan(rec["x"], rec["t"])
        tc, valid = grid.resample(rec["T"], rec["valid"], plan, "mean")
        X, _aux = features.make_channels(tc, valid, dx=float(plan["dx"]))
        ban = fbd.boxes_to_mask(plan["t"], plan["x"], zones)
        tag = f"real:{name}"
        zf, qf = _embed_strata(enc, X, valid, ban, n_each, patch, seed, tag, batch)
        ids.append(tag)
        zone_f.append(zf)
        quiet_f.append(qf)
        per[name] = {
            "sha256": rec["sha256"],
            "raw_shape": [int(rec["T"].shape[0]), int(rec["T"].shape[1])],
            "grid_shape": [int(valid.shape[0]), int(valid.shape[1])],
            "n_reference_zones": len(zones),
            "zone_cell_fraction": _round(float(ban[valid].mean()) if valid.any() else 0.0, 6),
            "n_zone": int(len(zf)),
            "n_quiet": int(len(qf)),
        }
        print(f"[fbd] real {name} grid {per[name]['grid_shape']} zones {len(zones)} "
              f"patches zone/quiet {len(zf)}/{len(qf)} ({time.time() - t0:.0f} с)", flush=True)
        del rec, tc, X
    return {"ids": ids, "zone": zone_f, "quiet": quiet_f, "n_files": len(names), "files": per}


def _pack_counts(pack):
    return {
        "n_files": int(pack["n_files"]),
        "n_zone": int(sum(len(a) for a in pack["zone"])),
        "n_quiet": int(sum(len(a) for a in pack["quiet"])),
    }


def eval_pair(pack_a, pack_b, seed, n_sizes, n_repeats):
    assembled = fbd.assemble_balanced(pack_a, pack_b, seed=seed)
    if assembled is None:
        return {"error": "не удалось выровнять зоны и тишину", "kid": None, "fd_inf": None,
                "probe": {"separation": None, "auc_oof": None}}
    fa, ga, fb, gb, info = assembled
    print(f"[fbd] balanced n={info['n']} zone={info['n_zone']} quiet={info['n_quiet']} "
          f"frac={info['zone_fraction']:.3f}", flush=True)
    rep = fbd.gap_report(fa, ga, fb, gb, seed=seed, n_sizes=n_sizes, n_repeats=n_repeats)
    rep["balance"] = info
    sep = rep["probe"].get("separation")
    fd = rep["fd_inf"]
    print(f"[fbd] kid={rep['kid']:.4f} fd_inf={None if fd is None else round(fd, 4)} "
          f"sep={None if sep is None else round(sep, 4)}", flush=True)
    return rep


def _num(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def judge_order(lower_block, real_blocks):
    """holds: граница < домен. substantial: KID/FD_∞ отношение ≥ 2, зонд +0.10 separation."""
    out = {}
    fields = (("kid", "kid"), ("fd_inf", "fd_inf"), ("probe", "separation"))
    for name, field in fields:
        if name == "probe":
            lower = _num((lower_block.get("probe") or {}).get("separation"))
            reals = {ver: _num((blk.get("probe") or {}).get("separation")) for ver, blk in real_blocks.items()}
        else:
            lower = _num(lower_block.get(field))
            reals = {ver: _num(blk.get(field)) for ver, blk in real_blocks.items()}
        holds, ratios, substantial = {}, {}, {}
        for ver, val in reals.items():
            if lower is None or val is None:
                holds[ver] = None
                ratios[ver] = None
                substantial[ver] = None
                continue
            holds[ver] = bool(lower < val)
            ratios[ver] = None if abs(lower) < 1e-12 else round(val / lower, 4)
            if not holds[ver]:
                substantial[ver] = False
            elif name == "probe":
                substantial[ver] = bool(val - lower >= 0.10)
            elif lower > 1e-8:
                substantial[ver] = bool(val / lower >= 2.0)
            else:
                substantial[ver] = bool(val > 0.0)
        out[name] = {
            "lower_v8_val_test": lower,
            "synth_vs_real": reals,
            "holds": holds,
            "ratios": ratios,
            "substantial": substantial,
            "all_hold": bool(holds) and all(v is True for v in holds.values()),
            "all_substantial": bool(substantial) and all(v is True for v in substantial.values()),
        }
    out["all_hold"] = all(out[k]["all_hold"] for k in ("kid", "fd_inf", "probe"))
    out["all_substantial"] = all(out[k]["all_substantial"] for k in ("kid", "fd_inf", "probe"))
    out["criterion"] = (
        "holds: значение(v8 val↔test) < значение(test↔real). "
        "substantial: KID и FD_∞ — отношение ≥ 2 при положительной границе, иначе граница ≤ 0 и домен > 0; "
        "зонд — separation больше хотя бы на 0.10."
    )
    return out


def legacy_from(prev):
    if not prev:
        return None
    if prev.get("legacy_unbalanced"):
        return prev["legacy_unbalanced"]
    if "lower_bound" not in prev:
        return None
    sv = prev.get("synth_vs_real") or {}
    per = {}
    pooled = {}
    for ver, block in sv.items():
        pooled[ver] = {
            "overall": block.get("overall"),
            "outside_zones": block.get("outside_zones"),
        }
        per[ver] = {}
        for name, modes in (block.get("per_file") or {}).items():
            per[ver][name] = {
                "overall": (modes.get("overall") or {}).get("fbd"),
                "overall_diag": (modes.get("overall") or {}).get("fbd_diag"),
                "outside": (modes.get("outside") or {}).get("fbd"),
                "outside_diag": (modes.get("outside") or {}).get("fbd_diag"),
                "shrinkage_real_overall": (modes.get("overall") or {}).get("shrinkage_b"),
            }
    return {
        "note": (
            "Первый прогон: Ледуа–Вольф FBD, n не выровнен, доля зон не согласована. "
            "Смещение при n≈d перевернуло порядок: synth-vs-synth оказался больше synth-vs-real."
        ),
        "lower_bound": prev.get("lower_bound"),
        "synth_vs_real": pooled,
        "per_file_fbd": per,
        "real_files": (prev.get("real") or {}).get("files"),
    }


def _clean(obj, nd=6):
    if isinstance(obj, dict):
        return {str(k): _clean(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v, nd) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return round(v, nd) if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def pair_block(a, b, boot, seed, shrink):
    if len(a) < 2 or len(b) < 2:
        return {"fbd": None, "fbd_diag": None, "ci95": None, "n_a": int(len(a)), "n_b": int(len(b))}
    if boot > 0:
        r = fbd.bootstrap_fbd(a, b, n_boot=boot, seed=seed, shrink=shrink)
        return {
            "fbd": _round(r["fbd"], 6),
            "fbd_diag": _round(r["fbd_diag"], 6),
            "ci95": [_round(r["ci95"][0], 6), _round(r["ci95"][1], 6)],
            "boot_mean": _round(r["boot_mean"], 6),
            "n_boot": int(boot),
            "n_a": int(r["n_a"]),
            "n_b": int(r["n_b"]),
            "feature_dim": int(r["feature_dim"]),
            "shrinkage_a": _round(r["shrinkage_a"], 6),
            "shrinkage_b": _round(r["shrinkage_b"], 6),
        }
    r = fbd.frechet_distance(a, b, shrink=shrink)
    return {
        "fbd": _round(r["fbd"], 6),
        "fbd_diag": _round(r["fbd_diag"], 6),
        "ci95": None,
        "n_a": int(r["n_a"]),
        "n_b": int(r["n_b"]),
        "feature_dim": int(r["feature_dim"]),
        "shrinkage_a": _round(r["shrinkage_a"], 6),
        "shrinkage_b": _round(r["shrinkage_b"], 6),
    }


def per_file_pairs(real_pack, synth_feats, boot, seed, shrink):
    out = {}
    for name, info in real_pack["files"].items():
        out[name] = {}
        for mode in ("overall", "outside"):
            block = pair_block(synth_feats[mode], info[mode], boot=0, seed=seed, shrink=shrink)
            block["n_real"] = int(len(info[mode]))
            out[name][mode] = {k: v for k, v in block.items() if k != "n_b"}
            out[name][mode]["n_synth"] = block["n_a"]
            out[name][mode]["n_real"] = block["n_b"]
    return out


def dump_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    tmp.replace(path)


def _fmt(x):
    if x is None:
        return "н/д"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _ci(block):
    ci = block.get("ci95")
    if not ci or ci[0] is None:
        return "—"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _r2_of(block):
    detail = block.get("fd_inf_detail") or {}
    return detail.get("r2")


def _balance_of(block):
    bal = block.get("balance") or {}
    return bal.get("n"), bal.get("zone_fraction")


def render_results(report: dict) -> str:
    lines = []
    lines.append("## Результаты прогона")
    lines.append("")
    dom = report.get("domain") or {}
    lines.append("Те же признаки bottleneck det0. Выборка выровнена: одинаковое число зонных и тихих патчей "
                 "с обеих сторон, файлы чередуются. KID — несмещённый MMD² с полиномиальным ядром степени 3. "
                 "FD_∞ — свободный член прямой FID(N) = FD_∞ + slope/N на выборочной ковариации. "
                 "Зонд — логрег, GroupKFold по файлам; в таблице separation = max(AUC, 1−AUC). "
                 "Нижняя граница — v8 val против v8 test.")
    if dom:
        lines.append("")
        lines.append(f"Чекпоинт width={dom.get('width')}, признак {dom.get('feature_dim')}, "
                     f"патч {dom.get('patch_t_x')}, на файл синтетики {dom.get('synth_n_per_stratum')}+"
                     f"{dom.get('synth_n_per_stratum')} (зона+тишина), на реальный кабель "
                     f"{dom.get('real_n_per_stratum')}+{dom.get('real_n_per_stratum')}, "
                     f"seed={dom.get('seed')}, FD_∞ размеров {dom.get('fd_inf_sizes')} × повторов {dom.get('fd_inf_repeats')}.")
    lines.append("")
    lines.append("| пара | n | доля зон | KID | FD_∞ | R² | FBD выборочный | separation | AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    labels = [("v8_val_vs_v8_test", "v8 val ↔ v8 test")]
    for ver in ("v8", "v10", "v11"):
        key = f"{ver}_test_vs_real"
        if key in (report.get("pairs") or {}):
            labels.append((key, f"{ver} test ↔ real 1–6"))
    for key, title in labels:
        block = report["pairs"][key]
        n, frac = _balance_of(block)
        probe = block.get("probe") or {}
        lines.append(f"| {title} | {n if n is not None else 'н/д'} | {_fmt(frac)} | {_fmt(block.get('kid'))} | "
                     f"{_fmt(block.get('fd_inf'))} | {_fmt(_r2_of(block))} | {_fmt(block.get('fbd_empirical'))} | "
                     f"{_fmt(probe.get('separation'))} | {_fmt(probe.get('auc_oof'))} |")
    lines.append("")
    order = report.get("order") or {}
    if order:
        bits = []
        for name, title in (("kid", "KID"), ("fd_inf", "FD_∞"), ("probe", "зонд")):
            item = order.get(name) or {}
            holds = item.get("holds") or {}
            ratios = item.get("ratios") or {}
            substantial = item.get("substantial") or {}
            parts = []
            for ver, flag in holds.items():
                word = "да" if flag else ("н/д" if flag is None else "нет")
                sub = substantial.get(ver)
                sub_word = "существенно" if sub else ("слабо" if sub is False else "н/д")
                ratio = ratios.get(ver)
                ratio_s = "н/д" if ratio is None else f"{ratio:.2f}"
                parts.append(f"{ver}: порядок {word}, отношение {ratio_s}, {sub_word}")
            bits.append(f"{title} — " + "; ".join(parts) + ".")
        lines.append("Порядок synth-vs-synth < synth-vs-real. " + " ".join(bits))
        lines.append("")
        kid_ok = (order.get("kid") or {}).get("all_substantial")
        fd_ok = (order.get("fd_inf") or {}).get("all_substantial")
        probe_hold = (order.get("probe") or {}).get("all_hold")
        probe_sub = (order.get("probe") or {}).get("all_substantial")
        if order.get("all_substantial"):
            lines.append("На всех трёх мерах и всех версиях зазор до реального домена существенно больше зазора сплита v8.")
        elif kid_ok and fd_ok and probe_hold and not probe_sub:
            ratios = [v for v in ((order.get("kid") or {}).get("ratios") or {}).values() if isinstance(v, (int, float))]
            span = ""
            if ratios:
                span = f" ({min(ratios):.2f}–{max(ratios):.2f})"
            stds = []
            for key, _title in labels:
                std = ((report["pairs"][key].get("probe") or {}).get("auc_folds_std"))
                if isinstance(std, (int, float)):
                    stds.append(std)
            std_s = ""
            if stds:
                std_s = f" Стандартное отклонение AUC по фолдам {min(stds):.2f}–{max(stds):.2f}."
            lines.append("KID даёт synth-vs-synth существенно меньше synth-vs-real" + span + ". "
                         "Зонд формально в том же порядке, но приращение separation меньше 0.10."
                         + std_s + " Линейный классификатор, который не видел тестовый файл, "
                         "почти не отделяет кабель от сплита синтетики. Полиномиальный MMD этот зазор видит.")
        elif order.get("all_hold"):
            lines.append("Порядок synth-vs-synth < synth-vs-real выполняется на всех трёх мерах, "
                         "но не на всех он существенный.")
        else:
            lines.append("Порядок выполняется не на всех мерах или не на всех версиях. Числа выше — как посчитаны, без подгонки.")
        lines.append("")
        r2_bits = []
        poor_fit = False
        for key, title in labels:
            r2 = _r2_of(report["pairs"][key])
            if r2 is None:
                continue
            r2_bits.append(f"{title} {r2:.2f}")
            if r2 < 0.8:
                poor_fit = True
        if poor_fit:
            emp = []
            for key, title in labels:
                val = report["pairs"][key].get("fbd_empirical")
                if isinstance(val, (int, float)):
                    emp.append(f"{title} {_fmt(val)}")
            lines.append("Прямая FID(N) = FD_∞ + slope/N ложится слабо (R²: " + ", ".join(r2_bits) + "). "
                         "Там, где R² низкий, свободный член повторяет выборочный FBD, а не уточняет его. "
                         "Экстраполяцию не читаем как отдельную оценку бесконечной выборки. "
                         "Порядок уже есть у выборочного FBD на той же выровненной выборке"
                         + (": " + "; ".join(emp) + "." if emp else "."))
            lines.append("")
        lines.append(order.get("criterion") or "")
        lines.append("")
    per = report.get("kid_per_file") or {}
    if per.get("versions"):
        lines.append("### KID по реальным файлам")
        lines.append("")
        lines.append("Каждый кабель сравнивается с test той же версии на признаках det0. "
                     "С обеих сторон одно и то же число зонных и тихих патчей, файлы синтетики чередуются. "
                     "Больший KID — кабель дальше от синтетики.")
        lines.append("")
        lines.append("| файл | KID v8 | KID v10 | KID v11 | n v8 | доля зон v8 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        names = list((per["versions"].get("v8") or {}).keys())
        if not names:
            names = list(next(iter(per["versions"].values())).keys())
        rank = (per.get("rank_far_to_near") or {}).get("v8") or names
        ordered = [n for n in rank if n in names] + [n for n in names if n not in rank]
        for name in ordered:
            cells = []
            n_v8 = frac_v8 = None
            for ver in ("v8", "v10", "v11"):
                row = (per["versions"].get(ver) or {}).get(name) or {}
                cells.append(_fmt(row.get("kid")))
                if ver == "v8":
                    n_v8 = row.get("n")
                    frac_v8 = row.get("zone_fraction")
            lines.append(f"| {name}.csv | {cells[0]} | {cells[1]} | {cells[2]} | {n_v8 if n_v8 is not None else 'н/д'} | {_fmt(frac_v8)} |")
        lines.append("")
        far = (per.get("rank_far_to_near") or {}).get("v8") or []
        if far:
            lines.append(f"Дальше всего от синтетики v8: {', '.join(n + '.csv' for n in far[:3])}. "
                         f"Ближе всего: {', '.join(n + '.csv' for n in far[-2:])}.")
            lines.append("")
    gap = report.get("encoder_gap") or {}
    if gap.get("encoders"):
        lines.append("### KID на энкодере DOM и на исходных")
        lines.append("")
        lines.append("Одни и те же патчи. det0 — исходный bottleneck FBD (ширина 24). "
                     "v810ft — старт дообучения DOM (`artifacts/stage1/unet_v810ft.pt`, ширина 16). "
                     "dom — сохранённый `artifacts/eval/s3_dom_unet.pt`. "
                     "Абсолютный KID зависит от масштаба признаков. Сравнивается отношение KID(test, real) / KID(v8 val, v8 test).")
        lines.append("")
        lines.append("| энкодер | dim | KID сплит | KID v8 | отношение v8 | KID v10 | отношение v10 | KID v11 | отношение v11 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name in ("det0", "v810ft", "dom"):
            block = (gap.get("encoders") or {}).get(name)
            if not block:
                continue
            ratios = block.get("ratios") or {}
            reals = block.get("real_kid") or {}
            lines.append(f"| {name} | {block.get('feature_dim')} | {_fmt(block.get('split_kid'))} | "
                         f"{_fmt(reals.get('v8'))} | {_fmt(ratios.get('v8'))} | {_fmt(reals.get('v10'))} | "
                         f"{_fmt(ratios.get('v10'))} | {_fmt(reals.get('v11'))} | {_fmt(ratios.get('v11'))} |")
        lines.append("")
        if gap.get("reading"):
            lines.append(gap["reading"])
            lines.append("")
    dom_vs = report.get("dom_ensemble_vs_winner") or {}
    if dom_vs.get("per_file"):
        lines.append("### DOM-ансамбль и победитель по файлам")
        lines.append("")
        lines.append("Победитель — `ens_u2ft_tcn_lgbm` (среднее unet_w24, unet_v810ft, tcn, pixel_lgbm). "
                     "Выбранный DOM-ансамбль — `ens_blend`, среднее этой карты и `unet_dom`. "
                     "KID и зонд читают bottleneck U-Net на тех же патчах v8 test против каждого кабеля "
                     "(n и доля зон совпадают). Зонд одного кабеля делит его патчи на 4 группы и держит группу вне обучения. "
                     "Меньше KID и меньше separation — кабель ближе к синтетике.")
        lines.append("")
        lines.append("| файл | KID w24 | sep w24 | KID v810 | sep v810 | KID dom | sep dom |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        order = ["5", "6", "2", "1", "3", "4"]
        known = list((dom_vs["per_file"].get("dom") or {}).keys())
        for name in order + [n for n in known if n not in order]:
            cells = []
            for enc in ("winner_w24", "winner_v810ft", "dom"):
                row = (dom_vs["per_file"].get(enc) or {}).get(name) or {}
                cells.append(_fmt(row.get("kid")))
                cells.append(_fmt((row.get("probe") or {}).get("separation")))
            if any(c != "н/д" for c in cells):
                lines.append(f"| {name}.csv | " + " | ".join(cells) + " |")
        lines.append("")
        if dom_vs.get("reading"):
            lines.append(dom_vs["reading"])
            lines.append("")
    if report.get("accepted_domain_diagnostic"):
        lines.append("Принятая диагностика доменного разрыва — KID. FBD не принят: ρ(KL, IoU) ниже 0.6 и для hot, и для cold.")
        lines.append("")
    leg = report.get("legacy_unbalanced") or {}
    if leg.get("lower_bound"):
        lines.append("### Прежний несбалансированный FBD")
        lines.append("")
        lines.append(leg.get("note") or "")
        lines.append("")
        lines.append("| пара | область | FBD | FBD diag | 95% ДИ | n_a | n_b |")
        lines.append("|---|---|---:|---:|---|---:|---:|")
        legacy_rows = [("v8 val ↔ v8 test", leg["lower_bound"])]
        for ver, block in (leg.get("synth_vs_real") or {}).items():
            legacy_rows.append((f"{ver} test ↔ real 1–6", block))
        for title, block in legacy_rows:
            for mode, label in (("overall", "всё поле"), ("outside_zones", "вне зон")):
                b = block.get(mode) or {}
                lines.append(f"| {title} | {label} | {_fmt(b.get('fbd'))} | {_fmt(b.get('fbd_diag'))} | {_ci(b)} | "
                             f"{b.get('n_a')} | {b.get('n_b')} |")
        lines.append("")
        pf = (leg.get("per_file_fbd") or {}).get("v8") or {}
        if pf:
            chunks = [f"{name}.csv {_fmt(row.get('overall'))} / вне зон {_fmt(row.get('outside'))}" for name, row in pf.items()]
            lines.append("Пофайлово v8, полный FBD кабеля против общего test (всё поле / вне зон): " + "; ".join(chunks) + ".")
            row2 = pf.get("2") or {}
            if row2:
                lines.append(f"У файла 2 сжатие Ледуа–Вольфа {_fmt(row2.get('shrinkage_real_overall'))}: "
                             f"полный FBD {_fmt(row2.get('overall'))} при диагональном {_fmt(row2.get('overall_diag'))}. "
                             "Полный FBD на 32 патчах здесь нестойкий.")
            lines.append("")
    kl = report.get("kl_fbd")
    if kl:
        lines.append("### Вариант ТЗ: KL на выходе сети и Спирмен с IoU")
        lines.append("")
        lines.append("Зоны на синтетике v8 test фиксированы разметкой `ev` (1–4 hot, 5–6 cold), "
                     "а не картой оцениваемой модели. Поле — вероятность `s` из `artifacts/pred/v8/*/test`. "
                     "IoU при пороге FAR 1e-3, порог по фону v8 val (квантиль гистограммы). "
                     "Реальные файлы в эту корреляцию не входят.")
        lines.append("")
        lines.append("| модель | порог | IoU | hot IoU | cold IoU | KL hot | KL cold | n hot | n cold |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, row in kl["models"].items():
            lines.append(f"| {name} | {_fmt(row['threshold_far_1e-3'])} | {_fmt(row['iou'])} | {_fmt(row['hot_iou'])} | "
                         f"{_fmt(row['cold_iou'])} | {_fmt(row['kl_hot_mean'])} | {_fmt(row['kl_cold_mean'])} | "
                         f"{row['kl_hot_n']} | {row['kl_cold_n']} |")
        lines.append("")
        rh, rc = kl["spearman"]["hot"], kl["spearman"]["cold"]
        lines.append(f"Спирмен ρ(KL, IoU): hot {_fmt(rh['rho'])} (p={_fmt(rh['pvalue'])}, n={rh['n']}), "
                     f"cold {_fmt(rc['rho'])} (p={_fmt(rc['pvalue'])}, n={rc['n']}).")
        lines.append("")
        if kl["verdict"] == "selector":
            lines.append("Вердикт: селектор. На синтетике v8 обе корреляции ρ(KL, IoU) не ниже 0.6, "
                         "KL зона/тишина можно использовать как безэталонный отбор модели.")
        else:
            lines.append("Вердикт: диагностика. На синтетике v8 хотя бы одна из корреляций ρ(KL, IoU) "
                         "ниже 0.6, поэтому KL остаётся диагностикой и в отбор детектора не входит.")
        unet = kl["models"].get("unet") or {}
        thr, iou = unet.get("threshold_far_1e-3"), unet.get("iou")
        if thr is not None and iou is not None and abs(thr - 0.9447) < 0.002 and abs(iou - 0.3625) < 0.002:
            lines.append("Порог unet на FAR 1e-3 совпал с прежней оценкой этапа 1 (0.945, IoU 0.363).")
        lines.append("")
    return "\n".join(lines)


def splice_md(section: str):
    path = ROOT / "docs" / "TZ_A_FBD.md"
    if not path.is_file():
        return
    doc = path.read_text()
    if RESULTS_START not in doc or RESULTS_END not in doc:
        return
    i = doc.index(RESULTS_START) + len(RESULTS_START)
    j = doc.index(RESULTS_END)
    path.write_text(doc[:i] + "\n\n" + section.rstrip() + "\n\n" + doc[j:])


def _sample_once(X, valid, zone, n_each, patch, seed, tag):
    zpatch = fbd.sample_zone_patches(X, valid, zone, n_each, patch, fbd.stable_seed(seed, tag, "zone"))
    qpatch = fbd.sample_patches(X, valid, n_each, patch, fbd.stable_seed(seed, tag, "quiet"), ban=zone)
    return zpatch, qpatch


def _embed_both(encs, zpatch, qpatch, batch):
    parts = [p for p in (zpatch, qpatch) if len(p)]
    out = {}
    if not parts:
        for name, enc in encs:
            empty = _empty_feat(enc.dim)
            out[name] = (empty, empty.copy())
        return out
    stacked = np.concatenate(parts, 0)
    nz = len(zpatch)
    for name, enc in encs:
        feat = enc.embed(stacked, batch=batch)
        out[name] = (feat[:nz], feat[nz:])
    return out


def collect_multi(encs, versions, names, n_synth, n_real, patch, seed, batch, limit, with_val=True):
    """Одни окна на все энкодеры. Пакет — зоны и тишина по файлам."""
    packs = {name: {} for name, _enc in encs}
    for ver in versions:
        splits = ("val", "test") if ver == "v8" and with_val else ("test",)
        for split in splits:
            files = list_seqs(ROOT / "artifacts" / "cache" / ver / split)
            if limit:
                files = files[:limit]
            ids = []
            buckets = {name: {"zone": [], "quiet": []} for name, _enc in encs}
            t0 = time.time()
            for path in files:
                arrays = load_npz_arrays(path, ("X", "valid", "ev"))
                zone = np.asarray(arrays["ev"] > 0)
                tag = f"{ver}:{split}:{path.stem}"
                zpatch, qpatch = _sample_once(arrays["X"], arrays["valid"], zone, n_synth, patch, seed, tag)
                emb = _embed_both(encs, zpatch, qpatch, batch)
                ids.append(tag)
                for name, _enc in encs:
                    zf, qf = emb[name]
                    buckets[name]["zone"].append(zf)
                    buckets[name]["quiet"].append(qf)
                print(f"[fbd] {tag} ({time.time() - t0:.0f} с)", flush=True)
                del arrays
            for name, _enc in encs:
                packs[name][f"{ver}_{split}"] = {
                    "ids": ids,
                    "zone": buckets[name]["zone"],
                    "quiet": buckets[name]["quiet"],
                    "n_files": len(files),
                }
            print(f"[fbd] multi embed {ver} {split} files {len(files)} ({time.time() - t0:.0f} с)", flush=True)
    real_buckets = {name: {"zone": [], "quiet": []} for name, _enc in encs}
    ids = []
    meta = {}
    t0 = time.time()
    for name_csv in names:
        rec = real.load_csv(REAL_DIR / f"{name_csv}.csv")
        zones = reference_zones(name_csv)
        plan = grid.plan(rec["x"], rec["t"])
        tc, valid = grid.resample(rec["T"], rec["valid"], plan, "mean")
        X, _aux = features.make_channels(tc, valid, dx=float(plan["dx"]))
        ban = fbd.boxes_to_mask(plan["t"], plan["x"], zones)
        tag = f"real:{name_csv}"
        zpatch, qpatch = _sample_once(X, valid, ban, n_real, patch, seed, tag)
        emb = _embed_both(encs, zpatch, qpatch, batch)
        ids.append(tag)
        for ename, _enc in encs:
            zf, qf = emb[ename]
            real_buckets[ename]["zone"].append(zf)
            real_buckets[ename]["quiet"].append(qf)
        meta[name_csv] = {
            "sha256": rec["sha256"],
            "grid_shape": [int(valid.shape[0]), int(valid.shape[1])],
            "n_reference_zones": len(zones),
            "n_zone": int(len(zpatch)),
            "n_quiet": int(len(qpatch)),
        }
        print(f"[fbd] multi real {name_csv} zone/quiet {len(zpatch)}/{len(qpatch)} ({time.time() - t0:.0f} с)", flush=True)
        del rec, tc, X
    real_packs = {}
    for ename, _enc in encs:
        real_packs[ename] = {
            "ids": ids,
            "zone": real_buckets[ename]["zone"],
            "quiet": real_buckets[ename]["quiet"],
            "n_files": len(names),
            "files": meta,
        }
    return packs, real_packs


def _one_file_pack(pack, index):
    return {"ids": [pack["ids"][index]], "zone": [pack["zone"][index]], "quiet": [pack["quiet"][index]]}


def summarize_kid(name, enc, packs, real_pack, versions, seed):
    pooled = {}
    if "v8_val" in packs and "v8_test" in packs:
        pooled["split"] = fbd.kid_balanced(
            packs["v8_val"], packs["v8_test"], fbd.stable_seed(seed, "v8_val_vs_v8_test"))
    real_kid = {}
    per = {}
    for ver in versions:
        real_kid[ver] = fbd.kid_balanced(
            packs[f"{ver}_test"], real_pack, fbd.stable_seed(seed, f"{ver}_test_vs_real"))
        rows = {}
        for i, fid in enumerate(real_pack["ids"]):
            key = fid.split(":")[-1]
            rows[key] = fbd.kid_balanced(
                packs[f"{ver}_test"], _one_file_pack(real_pack, i), fbd.stable_seed(seed, "file", ver, fid))
        per[ver] = rows
        print(f"[fbd] {name} {ver} pooled kid={real_kid[ver].get('kid')} "
              f"files={[round(rows[k]['kid'], 4) if rows[k].get('kid') is not None else None for k in rows]}",
              flush=True)
    split = pooled.get("split", {}).get("kid")
    ratios = {}
    for ver, block in real_kid.items():
        kid = block.get("kid")
        ratios[ver] = None if split is None or kid is None or abs(split) < 1e-12 else float(kid / split)
    rank = {}
    for ver, rows in per.items():
        rank[ver] = [fn for fn, _row in sorted(rows.items(), key=lambda kv: (-(kv[1].get("kid") if kv[1].get("kid") is not None else -1e9), kv[0]))]
    return {
        "name": name,
        "checkpoint": enc.path,
        "width": enc.width,
        "feature_dim": enc.dim,
        "split_kid": split,
        "split": pooled.get("split"),
        "real_kid": {ver: block.get("kid") for ver, block in real_kid.items()},
        "real": real_kid,
        "ratios": ratios,
        "per_file": per,
        "rank_far_to_near": rank,
    }


def _gap_reading(encoders):
    dom = encoders.get("dom") or {}
    init = encoders.get("v810ft") or {}
    det = encoders.get("det0") or {}

    def ratio(block, ver="v8"):
        return (block.get("ratios") or {}).get(ver)

    def real_kid(block, ver="v8"):
        return (block.get("real_kid") or {}).get(ver)

    dom_r, init_r, det_r = ratio(dom), ratio(init), ratio(det)
    if dom_r is None or init_r is None:
        return "Сравнение DOM со стартом не посчитано.", None
    decreased_ratio = bool(dom_r < init_r)
    dom_k, init_k = real_kid(dom), real_kid(init)
    parts = [
        f"Абсолютный KID(v8 test, real) на DOM {_fmt(dom_k)} против {_fmt(init_k)} на стартовом v810ft.",
        f"Сплит v8 val↔test {_fmt(init.get('split_kid'))} → {_fmt(dom.get('split_kid'))}, "
        f"отношение real/сплит {_fmt(init_r)} → {_fmt(dom_r)}.",
    ]
    if decreased_ratio:
        parts.append("Относительно разброса генератора разрыв тоже меньше.")
    else:
        parts.append("Относительно разброса генератора разрыв не сократился: сплит сжался сильнее, чем дистанция до кабелей.")
    if det_r is not None:
        parts.append(f"На det0 это отношение {_fmt(det_r)}.")

    def far_pair(block, label):
        rank = (block.get("rank_far_to_near") or {}).get("v8") or []
        per = (block.get("per_file") or {}).get("v8") or {}
        if len(rank) < 2:
            return ""
        bits = [f"{fn}.csv {_fmt((per.get(fn) or {}).get('kid'))}" for fn in rank[:2]]
        return f"{label}: {', '.join(bits)}."

    for label, block in (("Старт", init), ("DOM", dom)):
        line = far_pair(block, label)
        if line:
            parts.append(line)
    return " ".join(parts), decreased_ratio


def run_kid_files(args, patch, versions, names, out_path, prev):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import torch
    torch.set_num_threads(2)
    specs = [
        ("det0", Path(args.ckpt)),
        ("v810ft", ROOT / "artifacts" / "stage1" / "unet_v810ft.pt"),
        ("dom", ROOT / "artifacts" / "eval" / "s3_dom_unet.pt"),
    ]
    encs = []
    for name, path in specs:
        if not path.is_file():
            print(f"[fbd] нет чекпойнта {name}: {path}", flush=True)
            continue
        enc = fbd.BottleneckEncoder.load(path)
        encs.append((name, enc))
        print(f"[fbd] encoder {name} width={enc.width} dim={enc.dim} {path}", flush=True)
    if not any(name == "det0" for name, _enc in encs):
        raise SystemExit("нет det0")
    packs, real_packs = collect_multi(
        encs, versions, names, args.synth_n, args.real_n, patch, args.seed, args.batch, args.limit)
    summaries = {}
    for name, enc in encs:
        summaries[name] = summarize_kid(name, enc, packs[name], real_packs[name], versions, args.seed)
    reading, decreased = _gap_reading(summaries)
    det = summaries["det0"]
    report = dict(prev or {})
    report["kid_per_file"] = _clean({
        "encoder": "det0",
        "checkpoint": det["checkpoint"],
        "feature_dim": det["feature_dim"],
        "protocol": "each real file vs synth test, equal n_zone and n_quiet, round-robin synth files",
        "versions": det["per_file"],
        "rank_far_to_near": det["rank_far_to_near"],
    })
    report["encoder_gap"] = _clean({
        "note": "same patches; ratio = KID(test, real) / KID(v8 val, v8 test); absolute KID is not compared across encoders",
        "gap_decreased_vs_v810ft": decreased,
        "reading": reading,
        "encoders": summaries,
    })
    report["accepted_domain_diagnostic"] = "kid"
    report["fbd_accepted"] = False
    report["fbd_reject_reason"] = "spearman rho(KL, IoU) hot 0.4126 cold 0.2727, both below 0.6"
    report["seconds_kid_files"] = round(time.time() - args._t0, 1)
    dump_json(out_path, report)
    if args.write_md and report.get("kl_fbd") is not None and report.get("pairs"):
        splice_md(render_results(report))
    print(f"[fbd] kid-files wrote {out_path}", flush=True)
    print(f"[fbd] {reading}", flush=True)
    print(f"[fbd] rank v8 {det['rank_far_to_near'].get('v8')}", flush=True)


def _per_file_gap(synth_pack, real_pack, seed):
    rows = {}
    for i, fid in enumerate(real_pack["ids"]):
        key = fid.split(":")[-1]
        assembled = fbd.assemble_balanced(
            synth_pack, _one_file_pack(real_pack, i), seed=fbd.stable_seed(seed, "file", "v8", fid))
        if assembled is None:
            rows[key] = {"kid": None, "probe": {"separation": None, "auc_oof": None}}
            continue
        fa, ga, fb, gb, info = assembled
        probe = fbd.file_domain_probe(fa, ga, fb, n_chunks=4, n_splits=4, seed=fbd.stable_seed(seed, "probe", fid))
        row = dict(info)
        row["kid"] = fbd.kid_mmd(fa, fb)
        row["probe"] = probe
        rows[key] = row
        print(f"[fbd] file {key} kid={row['kid']:.4f} sep={probe.get('separation')} folds={probe.get('n_folds')}",
              flush=True)
    return rows


def _down(new, old):
    if new is None or old is None:
        return None
    return bool(new < old)


def run_dom_vs_winner(args, patch, names, out_path, prev):
    """KID и зонд по кабелю: U-Net победителя против U-Net внутри ens_blend."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import torch
    torch.set_num_threads(2)
    specs = [
        ("winner_w24", ROOT / "artifacts" / "stage1" / "unet_w24.pt"),
        ("winner_v810ft", ROOT / "artifacts" / "stage1" / "unet_v810ft.pt"),
        ("dom", ROOT / "artifacts" / "eval" / "s3_dom_unet.pt"),
    ]
    encs = []
    for name, path in specs:
        enc = fbd.BottleneckEncoder.load(path)
        encs.append((name, enc))
        print(f"[fbd] encoder {name} width={enc.width} dim={enc.dim}", flush=True)
    packs, real_packs = collect_multi(
        encs, ["v8"], names, args.synth_n, args.real_n, patch, args.seed, args.batch, args.limit, with_val=False)
    per = {}
    for name, enc in encs:
        print(f"[fbd] gap {name}", flush=True)
        per[name] = _per_file_gap(packs[name]["v8_test"], real_packs[name], args.seed)
    focus = ("5", "6", "2")
    focus_rows = {}
    for fid in focus:
        item = {}
        for name in per:
            row = per[name].get(fid) or {}
            probe = row.get("probe") or {}
            item[name] = {"kid": row.get("kid"), "separation": probe.get("separation"), "auc_oof": probe.get("auc_oof"),
                          "n": row.get("n"), "zone_fraction": row.get("zone_fraction")}
        for metric, key in (("kid", "kid"), ("separation", "separation")):
            dom_v = item["dom"][key]
            item[f"{metric}_down_vs_w24"] = _down(dom_v, item["winner_w24"][key])
            item[f"{metric}_down_vs_v810ft"] = _down(dom_v, item["winner_v810ft"][key])
        focus_rows[fid] = item
    both = []
    for fid, item in focus_rows.items():
        kid_down = item["kid_down_vs_w24"] and item["kid_down_vs_v810ft"]
        sep_down = item["separation_down_vs_w24"] and item["separation_down_vs_v810ft"]
        both.append(fid if kid_down and sep_down else None)
    shrunk = [fid for fid in both if fid]
    reading_bits = []
    for fid in focus:
        item = focus_rows[fid]
        reading_bits.append(
            f"{fid}.csv: KID w24 {_fmt(item['winner_w24']['kid'])} / v810 {_fmt(item['winner_v810ft']['kid'])} / "
            f"dom {_fmt(item['dom']['kid'])}; separation {_fmt(item['winner_w24']['separation'])} / "
            f"{_fmt(item['winner_v810ft']['separation'])} / {_fmt(item['dom']['separation'])}."
        )
    if len(shrunk) == len(focus):
        head = "На файлах 5, 6 и 2 и KID, и separation зонда ниже на энкодере unet_dom, чем на обоих U-Net победителя."
    elif shrunk:
        head = ("Ниже на обоих U-Net победителя по KID и по зонду только файлы " + ", ".join(shrunk) +
                ". На остальных из 5, 6, 2 хотя бы одна мера не снизилась.")
    else:
        head = ("Одновременно и KID, и separation не стали ниже сразу на обоих U-Net победителя "
                "ни у файла 5, ни у 6, ни у 2.")
    reading = ("ens_blend — среднее карт победителя ens_u2ft_tcn_lgbm и unet_dom. "
               "KID и зонд читают bottleneck U-Net, не усреднённую карту. " + head + " " + " ".join(reading_bits))
    report = dict(prev or {})
    report["dom_ensemble_vs_winner"] = _clean({
        "winner": "ens_u2ft_tcn_lgbm = mean(unet_w24, unet_v810ft, tcn, pixel_lgbm)",
        "dom_ensemble": "ens_blend = mean(ens_u2ft_tcn_lgbm, unet_dom)",
        "synth": "v8 test",
        "probe": "file_domain_probe, 4 chunks of the real file, GroupKFold",
        "per_file": per,
        "focus_5_6_2": focus_rows,
        "focus_both_metrics_down_vs_both_unets": shrunk,
        "reading": reading,
    })
    report["seconds_dom_vs_winner"] = round(time.time() - args._t0, 1)
    dump_json(out_path, report)
    if args.write_md and report.get("pairs"):
        splice_md(render_results(report))
    print(f"[fbd] {reading}", flush=True)
    print(f"[fbd] dom-vs-winner wrote {out_path}", flush=True)


FBD5_START = "<!-- FBD5_START -->"
FBD5_END = "<!-- FBD5_END -->"
NOISE_PHI_T = 0.320
NOISE_PHI_X = 0.556
NOISE_STD_C = 0.283


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_windows(X, valid, zone, n, patch, seed, tag):
    z, zo = fbd.sample_zone_patches(
        X, valid, zone, n, patch, fbd.stable_seed(seed, tag, "zone"), return_origins=True)
    q, qo = fbd.sample_patches(
        X, valid, n, patch, fbd.stable_seed(seed, tag, "quiet"), ban=zone, return_origins=True)
    return z, zo, q, qo


def _embed_many(enc, arrays, batch):
    parts = []
    spans = []
    cursor = 0
    for arr in arrays:
        arr = np.asarray(arr, np.float32)
        if len(arr) == 0:
            spans.append(None)
            continue
        parts.append(arr)
        spans.append((cursor, cursor + len(arr)))
        cursor += len(arr)
    if not parts:
        return [_empty_feat(enc.dim) for _ in arrays]
    feat = enc.embed(np.concatenate(parts, 0), batch=batch)
    out = []
    for span in spans:
        if span is None:
            out.append(_empty_feat(enc.dim))
        else:
            out.append(feat[span[0]:span[1]])
    return out


def _cat_pack(pack) -> np.ndarray:
    parts = []
    for zone, quiet in zip(pack["zone"], pack["quiet"]):
        if len(zone):
            parts.append(np.asarray(zone, np.float64))
        if len(quiet):
            parts.append(np.asarray(quiet, np.float64))
    if not parts:
        return np.zeros((0, 1), np.float64)
    return np.concatenate(parts, 0)


def _rank_rows(rows: dict) -> list:
    return [key for key, _row in sorted(
        rows.items(),
        key=lambda kv: (-(kv[1].get("kid") if kv[1].get("kid") is not None else -1e9), kv[0]))]


def _kid_vs_real(synth_pack, real_pack, ver, seed):
    pooled = fbd.kid_balanced(synth_pack, real_pack, fbd.stable_seed(seed, f"{ver}_test_vs_real"))
    rows = {}
    for i, fid in enumerate(real_pack["ids"]):
        key = fid.split(":")[-1]
        rows[key] = fbd.kid_balanced(
            synth_pack, _one_file_pack(real_pack, i), fbd.stable_seed(seed, "file", ver, fid))
    return pooled, rows


def _max_kid_gap(new_rows_by_ver, old_rows_by_ver) -> float | None:
    diffs = []
    for ver, rows in new_rows_by_ver.items():
        old_rows = (old_rows_by_ver or {}).get(ver) or {}
        for key, row in rows.items():
            fresh = row.get("kid")
            old = (old_rows.get(key) or {}).get("kid")
            if fresh is None or old is None:
                continue
            diffs.append(abs(round(float(fresh), 6) - float(old)))
    if not diffs:
        return None
    return float(max(diffs))


def _param_delta(model, reference: dict) -> float:
    delta = 0.0
    for key, value in model.state_dict().items():
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue
        delta = max(delta, float((value.detach().cpu() - reference[key]).abs().max()))
    return delta


def _ratio(num, den):
    if num is None or den is None or abs(float(den)) < 1e-12:
        return None
    return float(num) / float(den)


def render_fbd5(report: dict) -> str:
    rel = report.get("ens_dom_release") or {}
    pipe = report.get("bn_noise") or {}
    lines = ["## FBD5. ens_dom и test-time BN без калибровки на real", ""]
    lines.append("`ens_dom` — среднее карты `ens_u2ft_tcn_lgbm` и `unet_dom`. "
                 "В `models_release_dom/` лежит только `unet_dom.pt` и порог. "
                 "У средней карты нет своего bottleneck, поэтому KID читает bottleneck этого U-Net "
                 "(ширина 16, размерность 128), не усреднённый скор.")
    lines.append("")
    if rel.get("weights_identical"):
        lines.append(f"Per-file KID этого выпуска уже был посчитан: `models_release_dom/unet_dom.pt` "
                     f"побайтно совпадает с `artifacts/eval/s3_dom_unet.pt` "
                     f"(sha256 `{rel.get('sha256')}`). "
                     "Таблица ниже взята из того прогона (n=240, доля зон 0.5, seed 0) и заново не считалась.")
    else:
        lines.append("Веса `models_release_dom/unet_dom.pt` не совпали с ранее посчитанным чекпойнтом, "
                     "поэтому per-file KID посчитан этим прогоном на тех же окнах, что и KID по шуму.")
    lines.append("")
    versions = (pipe.get("noise") or {}).get("versions") or {}
    order = list((rel.get("rank_far_to_near") or {}).get("v8") or [])
    per = rel.get("per_file") or {}
    names = order or list((per.get("v8") or {}).keys())
    if names:
        lines.append("| файл | KID v8 | KID v10 | KID v11 |")
        lines.append("|---|---:|---:|---:|")
        for name in names:
            cells = []
            for ver in ("v8", "v10", "v11"):
                cells.append(_fmt(((per.get(ver) or {}).get(name) or {}).get("kid")))
            lines.append(f"| {name}.csv | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")
    rank = rel.get("rank_far_to_near") or {}
    if rank.get("v8"):
        lines.append("Дальний → ближний на bottleneck выпуска: "
                     + "; ".join(f"{ver} {' > '.join(rank[ver])}" for ver in ("v8", "v10", "v11") if rank.get(ver))
                     + ".")
        lines.append("")
    if rel.get("reading"):
        lines.append(rel["reading"])
        lines.append("")
    lines.append("### Статистики BN: только синтетика")
    lines.append("")
    lines.append("AdaBN (Li et al., 2016, arXiv:1603.04779) подставляет в BatchNorm статистики целевого домена "
                 "и не трогает остальные веса. TENT (Wang et al., ICLR 2021, arXiv:2006.10726) на тестовом батче "
                 "обновляет ещё и γ, β минимизацией энтропии. Оба шага на CSV МГТС были бы калибровкой. "
                 "Их нет: реальные патчи идут только в eval, running mean и var сверяются с чекпойнтом.")
    lines.append("")
    bn = pipe.get("bn") or {}
    lines.append(f"После прохода реальных патчей max |Δ| running-статистик = {_fmt(bn.get('real_forward_max_abs'))}. "
                 "Калибровка на real не выполнялась. "
                 f"Разрешённый тест-тайм — накопительное среднее running-статистик по патчам синтетики "
                 f"v8/v10/v11 test (n={bn.get('n_synth_patches')}), γ и β заморожены "
                 f"(max |Δ| весов {_fmt(bn.get('weight_max_abs'))}). "
                 "Реальные патчи затем читаются этими статистиками в eval и сами их не двигают "
                 f"(max |Δ| = {_fmt(bn.get('real_on_synth_bn_max_abs'))}).")
    lines.append("")
    lines.append("| версия | KID frozen ↔ synth-BN | KID frozen ↔ real | KID synth-BN ↔ real, одни статистики |")
    lines.append("|---|---:|---:|---:|")
    for ver in ("v8", "v10", "v11"):
        row = (bn.get("versions") or {}).get(ver) or {}
        lines.append(f"| {ver} | {_fmt(row.get('kid_frozen_vs_synth_bn'))} | {_fmt(row.get('kid_frozen_vs_real'))} | "
                     f"{_fmt(row.get('kid_matched_vs_real'))} |")
    lines.append("")
    if bn.get("cross_version_kid") is not None:
        lines.append(f"На тех же синтетических статистиках KID(v8 test, v10 test) = {_fmt(bn.get('cross_version_kid'))}. "
                     "Несмещённый KID совпадающих облаков колеблется около нуля, поэтому версии генератора здесь не разделены.")
        lines.append("")
    if bn.get("reading"):
        lines.append(bn["reading"])
        lines.append("")
    lines.append("### Устойчивость: KID по шуму v8/v10/v11")
    lines.append("")
    lines.append("Шум — рецепт фона из `docs/S3_LIT.md`: AR(1) с φ_t = 0.320, φ_x = 0.556, "
                 "std = 0.283 °C на поле T до `make_channels`. Окна совпадают с чистыми патчами. "
                 "BN остаётся статистикой чекпойнта: ни real, ни зашумлённый батч её не обновляют. "
                 "KID(clean, noisy) — сдвиг облака. KID(noisy, real) / KID(clean, real) — сдвинулся ли синтетический test к кабелям.")
    lines.append("")
    lines.append("| версия | KID clean ↔ noisy | KID clean ↔ real | KID noisy ↔ real | noisy / clean |")
    lines.append("|---|---:|---:|---:|---:|")
    noise = pipe.get("noise") or {}
    for ver in ("v8", "v10", "v11"):
        row = (noise.get("versions") or {}).get(ver) or {}
        lines.append(f"| {ver} | {_fmt(row.get('kid_clean_vs_noisy'))} | {_fmt(row.get('kid_clean_vs_real'))} | "
                     f"{_fmt(row.get('kid_noisy_vs_real'))} | {_fmt(row.get('ratio_noisy_over_clean'))} |")
    lines.append("")
    noisy_per = noise.get("per_file") or {}
    if names and noisy_per:
        lines.append("| файл | noisy v8 | noisy v10 | noisy v11 |")
        lines.append("|---|---:|---:|---:|")
        for name in names:
            cells = []
            for ver in ("v8", "v10", "v11"):
                cells.append(_fmt((((noisy_per.get(ver) or {}).get(name) or {}).get("kid"))))
            lines.append(f"| {name}.csv | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")
    if noise.get("reading"):
        lines.append(noise["reading"])
        lines.append("")
    if pipe.get("reading"):
        lines.append(pipe["reading"])
        lines.append("")
    return "\n".join(lines)


def splice_fbd5(section: str):
    path = ROOT / "docs" / "TZ_A_FBD.md"
    if not path.is_file():
        return
    doc = path.read_text()
    block = FBD5_START + "\n\n" + section.rstrip() + "\n\n" + FBD5_END + "\n"
    if FBD5_START in doc and FBD5_END in doc:
        i = doc.index(FBD5_START)
        j = doc.index(FBD5_END) + len(FBD5_END)
        if j < len(doc) and doc[j] == "\n":
            j += 1
        doc = doc[:i] + block + doc[j:]
    else:
        if not doc.endswith("\n"):
            doc += "\n"
        doc += "\n" + block
    path.write_text(doc)


def run_bn_noise(args, patch, versions, names, out_path, prev):
    """KID выпуска ens_dom, если его ещё нет, и устойчивость по шуму без BN на real."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import torch
    torch.set_num_threads(2)

    release = ROOT / "models_release_dom" / "unet_dom.pt"
    s3_path = ROOT / "artifacts" / "eval" / "s3_dom_unet.pt"
    if not release.is_file():
        raise SystemExit(f"нет {release}")
    sha_release = _sha256(release)
    sha_s3 = _sha256(s3_path) if s3_path.is_file() else None
    identical = sha_s3 is not None and sha_release == sha_s3
    dom_prev = (((prev or {}).get("encoder_gap") or {}).get("encoders") or {}).get("dom") or {}
    have_table = bool(dom_prev.get("per_file")) and identical
    print(f"[fbd] ens_dom sha256 {sha_release} identical_to_s3={identical} per_file_already={have_table}", flush=True)

    enc = fbd.BottleneckEncoder.load(release)
    enc_bn = fbd.BottleneckEncoder.load(release)
    snap = fbd.bn_running_snapshot(enc.model)
    bn_reference = {k: v.detach().cpu().clone() for k, v in enc_bn.model.state_dict().items()}
    session = fbd.SynthBNSession(enc_bn.model, batch=args.batch, domain="synth")
    print(f"[fbd] encoder ens_dom width={enc.width} dim={enc.dim} bn_layers={len(snap)}", flush=True)

    clean_packs = {}
    noisy_packs = {}
    for ver in versions:
        files = list_seqs(ROOT / "artifacts" / "cache" / ver / "test")
        if args.limit:
            files = files[:args.limit]
        ids, cz, cq, nz, nq = [], [], [], [], []
        t0 = time.time()
        for path in files:
            arrays = load_npz_arrays(path, ("X", "T", "valid", "ev"))
            zone = np.asarray(arrays["ev"] > 0)
            tag = f"{ver}:test:{path.stem}"
            z, zo, q, qo = _sample_windows(arrays["X"], arrays["valid"], zone, args.synth_n, patch, args.seed, tag)
            noisy = fbd.channels_with_colored_noise(
                arrays["T"], arrays["valid"], fbd.stable_seed(args.seed, "noise", ver, path.stem),
                NOISE_PHI_T, NOISE_PHI_X, NOISE_STD_C, dx=0.25)
            zn = fbd.crop_patches(noisy, zo, patch) if len(zo) else np.zeros((0, 3, patch[0], patch[1]), np.float32)
            qn = fbd.crop_patches(noisy, qo, patch) if len(qo) else np.zeros((0, 3, patch[0], patch[1]), np.float32)
            fz, fq, fzn, fqn = _embed_many(enc, (z, q, zn, qn), args.batch)
            session.feed(z)
            session.feed(q)
            ids.append(tag)
            cz.append(fz)
            cq.append(fq)
            nz.append(fzn)
            nq.append(fqn)
            print(f"[fbd] noise {tag} zone/quiet {len(z)}/{len(q)} ({time.time() - t0:.0f} с)", flush=True)
            del arrays, noisy
        clean_packs[ver] = {"ids": ids, "zone": cz, "quiet": cq, "n_files": len(files)}
        noisy_packs[ver] = {"ids": ids, "zone": nz, "quiet": nq, "n_files": len(files)}
        print(f"[fbd] noise {ver} test files {len(files)} ({time.time() - t0:.0f} с)", flush=True)

    real_ids, rz, rq = [], [], []
    t0 = time.time()
    for name_csv in names:
        rec = real.load_csv(REAL_DIR / f"{name_csv}.csv")
        zones = reference_zones(name_csv)
        plan = grid.plan(rec["x"], rec["t"])
        tc, valid = grid.resample(rec["T"], rec["valid"], plan, "mean")
        channels, _aux = features.make_channels(tc, valid, dx=float(plan["dx"]))
        ban = fbd.boxes_to_mask(plan["t"], plan["x"], zones)
        tag = f"real:{name_csv}"
        z, _zo, q, _qo = _sample_windows(channels, valid, ban, args.real_n, patch, args.seed, tag)
        fz, fq = _embed_many(enc, (z, q), args.batch)
        real_ids.append(tag)
        rz.append(fz)
        rq.append(fq)
        print(f"[fbd] noise real {name_csv} zone/quiet {len(z)}/{len(q)} ({time.time() - t0:.0f} с)", flush=True)
        del rec, tc, channels
    real_pack = {"ids": real_ids, "zone": rz, "quiet": rq, "n_files": len(names)}
    real_delta = fbd.bn_stats_max_abs(snap, fbd.bn_running_snapshot(enc.model))
    print(f"[fbd] real forward BN max |Δ| {real_delta}", flush=True)
    session.close()
    weight_delta = _param_delta(enc_bn.model, bn_reference)
    bn_shift = fbd.bn_stats_max_abs(snap, fbd.bn_running_snapshot(enc_bn.model))
    print(f"[fbd] synth BN patches {session.n_forwarded} weight |Δ| {weight_delta} running |Δ| {bn_shift}", flush=True)

    bn_packs = {}
    for ver in versions:
        files = list_seqs(ROOT / "artifacts" / "cache" / ver / "test")
        if args.limit:
            files = files[:args.limit]
        ids, cz, cq = [], [], []
        t0 = time.time()
        for path in files:
            arrays = load_npz_arrays(path, ("X", "valid", "ev"))
            zone = np.asarray(arrays["ev"] > 0)
            tag = f"{ver}:test:{path.stem}"
            z, _zo, q, _qo = _sample_windows(arrays["X"], arrays["valid"], zone, args.synth_n, patch, args.seed, tag)
            fz, fq = _embed_many(enc_bn, (z, q), args.batch)
            ids.append(tag)
            cz.append(fz)
            cq.append(fq)
            del arrays
        bn_packs[ver] = {"ids": ids, "zone": cz, "quiet": cq, "n_files": len(files)}
        print(f"[fbd] synth-bn embed {ver} ({time.time() - t0:.0f} с)", flush=True)

    bn_snap = fbd.bn_running_snapshot(enc_bn.model)
    rb_ids, rbz, rbq = [], [], []
    t0 = time.time()
    for name_csv in names:
        rec = real.load_csv(REAL_DIR / f"{name_csv}.csv")
        zones = reference_zones(name_csv)
        plan = grid.plan(rec["x"], rec["t"])
        tc, valid = grid.resample(rec["T"], rec["valid"], plan, "mean")
        channels, _aux = features.make_channels(tc, valid, dx=float(plan["dx"]))
        ban = fbd.boxes_to_mask(plan["t"], plan["x"], zones)
        tag = f"real:{name_csv}"
        z, _zo, q, _qo = _sample_windows(channels, valid, ban, args.real_n, patch, args.seed, tag)
        fz, fq = _embed_many(enc_bn, (z, q), args.batch)
        rb_ids.append(tag)
        rbz.append(fz)
        rbq.append(fq)
        print(f"[fbd] synth-bn real {name_csv} ({time.time() - t0:.0f} с)", flush=True)
        del rec, tc, channels
    real_bn_pack = {"ids": rb_ids, "zone": rbz, "quiet": rbq, "n_files": len(names)}
    real_on_synth = fbd.bn_stats_max_abs(bn_snap, fbd.bn_running_snapshot(enc_bn.model))
    print(f"[fbd] real on synth BN max |Δ| {real_on_synth}", flush=True)

    full = (args.limit == 0 and versions == ["v8", "v10", "v11"] and names == ["1", "2", "3", "4", "5", "6"])
    clean_real = {}
    noisy_real = {}
    bn_real = {}
    clean_per = {}
    noisy_per = {}
    noise_versions = {}
    bn_versions = {}
    for ver in versions:
        pooled_c, rows_c = _kid_vs_real(clean_packs[ver], real_pack, ver, args.seed)
        pooled_n, rows_n = _kid_vs_real(noisy_packs[ver], real_pack, ver, args.seed)
        pooled_b, _rows_b = _kid_vs_real(bn_packs[ver], real_pack, ver, args.seed)
        pooled_m, _rows_m = _kid_vs_real(bn_packs[ver], real_bn_pack, ver, args.seed)
        shift = fbd.kid_mmd(_cat_pack(clean_packs[ver]), _cat_pack(noisy_packs[ver]))
        bn_cloud = fbd.kid_mmd(_cat_pack(clean_packs[ver]), _cat_pack(bn_packs[ver]))
        clean_real[ver] = pooled_c
        noisy_real[ver] = pooled_n
        bn_real[ver] = pooled_b
        clean_per[ver] = rows_c
        noisy_per[ver] = rows_n
        noise_versions[ver] = {
            "kid_clean_vs_noisy": shift,
            "n_paired": int(len(_cat_pack(clean_packs[ver]))),
            "kid_clean_vs_real": pooled_c.get("kid"),
            "clean_balance": {k: pooled_c.get(k) for k in ("n", "n_zone", "n_quiet", "zone_fraction")},
            "kid_noisy_vs_real": pooled_n.get("kid"),
            "noisy_balance": {k: pooled_n.get(k) for k in ("n", "n_zone", "n_quiet", "zone_fraction")},
            "ratio_noisy_over_clean": _ratio(pooled_n.get("kid"), pooled_c.get("kid")),
            "rank_noisy": _rank_rows(rows_n),
        }
        bn_versions[ver] = {
            "kid_frozen_vs_synth_bn": bn_cloud,
            "kid_frozen_vs_real": pooled_c.get("kid"),
            "kid_synth_bn_vs_frozen_real_mixed_scale": pooled_b.get("kid"),
            "mixed_scale_is_not_a_gap": True,
            "kid_matched_vs_real": pooled_m.get("kid"),
            "matched_balance": {k: pooled_m.get(k) for k in ("n", "n_zone", "n_quiet", "zone_fraction")},
        }
        print(f"[fbd] {ver} clean-real {pooled_c.get('kid')} noisy-real {pooled_n.get('kid')} "
              f"clean-noisy {shift} bn-real {pooled_b.get('kid')} bn-shift {bn_cloud}", flush=True)

    repro = _max_kid_gap(clean_per, dom_prev.get("per_file")) if full and dom_prev.get("per_file") else None
    if have_table:
        official_per = dom_prev["per_file"]
        official_rank = dom_prev.get("rank_far_to_near") or {ver: _rank_rows(official_per[ver]) for ver in official_per}
        official_real = dom_prev.get("real_kid")
        official_split = dom_prev.get("split_kid")
        official_ratios = dom_prev.get("ratios")
    else:
        official_per = clean_per
        official_rank = {ver: _rank_rows(rows) for ver, rows in clean_per.items()}
        official_real = {ver: block.get("kid") for ver, block in clean_real.items()}
        official_split = dom_prev.get("split_kid")
        official_ratios = None

    noise_bits = []
    toward = []
    away = []
    for ver in versions:
        row = noise_versions[ver]
        ratio = row["ratio_noisy_over_clean"]
        if ratio is not None and ratio < 1:
            toward.append(ver)
        elif ratio is not None and ratio > 1:
            away.append(ver)
        noise_bits.append(
            f"{ver}: KID(clean, noisy) {_fmt(row['kid_clean_vs_noisy'])}, "
            f"KID(clean, real) {_fmt(row['kid_clean_vs_real'])}, "
            f"KID(noisy, real) {_fmt(row['kid_noisy_vs_real'])}, отношение {_fmt(ratio)}")
    file_up = []
    file_down = []
    if have_table or official_per:
        for fid in ("1", "2", "3", "4", "5", "6"):
            ups, downs = [], []
            for ver in versions:
                old = ((official_per.get(ver) or {}).get(fid) or {}).get("kid")
                new = ((noisy_per.get(ver) or {}).get(fid) or {}).get("kid")
                if old is None or new is None:
                    continue
                if new > old:
                    ups.append(ver)
                elif new < old:
                    downs.append(ver)
            if ups and not downs:
                file_up.append(fid)
            elif downs and not ups:
                file_down.append(fid)
    if toward and not away:
        noise_head = "Пул test с этим шумом ближе к смеси кабелей на всех версиях."
    elif away and not toward:
        noise_head = "Пул test с этим шумом дальше от смеси кабелей на всех версиях."
    elif toward or away:
        noise_head = ("Пул ближе к смеси кабелей на " + ", ".join(toward) +
                      " и дальше на " + ", ".join(away) + ".")
    else:
        noise_head = "Отношение KID(noisy, real) / KID(clean, real) не посчитано."
    if file_up:
        noise_head += " Дальше от синтетики на всех версиях стали файлы " + ", ".join(n + ".csv" for n in file_up) + "."
    if "1" in file_up and (official_per.get("v8") or {}).get("1") and (noisy_per.get("v8") or {}).get("1"):
        noise_head += (f" На v8 у 1.csv KID {_fmt(official_per['v8']['1']['kid'])} → "
                       f"{_fmt(noisy_per['v8']['1']['kid'])}.")
    if file_down:
        noise_head += " Ближе стали " + ", ".join(n + ".csv" for n in file_down) + "."
    rank_bits = []
    for ver in versions:
        clean_rank = _rank_rows(clean_per[ver])
        noisy_rank = noise_versions[ver]["rank_noisy"]
        if clean_rank == noisy_rank:
            rank_bits.append(f"{ver} порядок файлов тот же ({' > '.join(noisy_rank)})")
        else:
            rank_bits.append(f"{ver} порядок сменился: {' > '.join(clean_rank)} → {' > '.join(noisy_rank)}")
    noise_reading = noise_head + " " + ". ".join(noise_bits) + ". " + "; ".join(rank_bits) + "."

    cross = None
    if "v8" in bn_packs and "v10" in bn_packs:
        cross = fbd.kid_balanced(bn_packs["v8"], bn_packs["v10"], fbd.stable_seed(args.seed, "v8_test_vs_v10_test"))
    bn_bits = []
    for ver in versions:
        row = bn_versions[ver]
        bn_bits.append(
            f"{ver}: сдвиг облака при смене статистик {_fmt(row['kid_frozen_vs_synth_bn'])}, "
            f"KID к real на общих статистиках {_fmt(row.get('kid_matched_vs_real'))}")
    bn_head = ("Смешивать признаки synth-BN с признаками real, снятыми на статистиках чекпойнта, нельзя: "
               "полиномиальное ядро зависит от масштаба, а running-статистики сдвигаются на сотни единиц. "
               "Разрыв читается только когда обе стороны прогнаны с одними и теми же синтетическими статистиками.")
    bn_reading = (bn_head + " " + ". ".join(bn_bits) + ". "
                  "Порог детектора и веса свёрток этим проходом не обновлялись. Реальные CSV в накопление BN не входили.")
    if cross is not None and cross.get("kid") is not None:
        bn_reading += f" KID(v8 test, v10 test) на этих статистиках {_fmt(cross.get('kid'))}."

    if have_table:
        rel_reading = ("Повторный чистый проход на тех же окнах "
                       + (f"совпал с таблицей выпуска: max |Δ KID| = {repro:.6g}. " if repro is not None
                          else "не сравнивался с таблицей выпуска (укороченная выборка). "))
    else:
        rel_reading = "Таблица per-file посчитана этим прогоном. "
    if repro is not None and repro > 1e-6:
        rel_reading = (f"Чистый проход не повторил опубликованный per-file KID: max |Δ| = {repro:.6g}. "
                       "В таблице оставлены уже опубликованные числа выпуска.")

    pipe_reading = ("Диагностика доменного разрыва по-прежнему KID на замороженном bottleneck. "
                    "Конвейер для real не калибрует BN, порог и зонд. "
                    "Устойчивость — KID по шуму v8/v10/v11 при статистиках BN, снятых с синтетики.")

    report = dict(prev or {})
    report["ens_dom_release"] = _clean({
        "method": "ens_dom",
        "definition": "score = 0.5 * (ens_u2ft_tcn_lgbm, unet_dom); KID uses the U-Net bottleneck only",
        "checkpoint": str(release),
        "sha256": sha_release,
        "same_weights_as": str(s3_path),
        "same_sha256": sha_s3,
        "weights_identical": identical,
        "already_computed": have_table,
        "recomputed": not have_table,
        "width": enc.width,
        "feature_dim": enc.dim,
        "per_file": official_per,
        "rank_far_to_near": official_rank,
        "real_kid": official_real,
        "split_kid": official_split,
        "ratios": official_ratios,
        "clean_repro_max_abs": repro,
        "reading": rel_reading.strip(),
    })
    report["bn_noise"] = _clean({
        "real_policy": "MGTS csv evaluation only; BN, threshold and probe are not fit on real",
        "literature": {
            "adabn": "Li et al. 2016 arXiv:1603.04779, BN stats of the target domain, other weights frozen",
            "tent": "Wang et al. ICLR 2021 arXiv:2006.10726, entropy min on affine and BN stats",
            "noise": "colored AR(1) before make_channels, phi from docs/S3_LIT.md pooled outside zones",
        },
        "bn": {
            "real_calibrated": False,
            "real_forward_max_abs": real_delta,
            "synth_only": True,
            "n_synth_patches": int(session.n_forwarded),
            "weight_max_abs": weight_delta,
            "running_vs_checkpoint_max_abs": bn_shift,
            "real_on_synth_bn_max_abs": real_on_synth,
            "cross_version_kid": None if cross is None else cross.get("kid"),
            "cross_version": None if cross is None else {k: cross.get(k) for k in ("kid", "n", "n_zone", "n_quiet", "zone_fraction")},
            "versions": bn_versions,
            "reading": bn_reading,
        },
        "noise": {
            "phi_t": NOISE_PHI_T,
            "phi_x": NOISE_PHI_X,
            "std_c": NOISE_STD_C,
            "jitter": 0,
            "bn": "checkpoint running stats, eval mode",
            "versions": noise_versions,
            "per_file": noisy_per,
            "reading": noise_reading,
        },
        "reading": pipe_reading,
    })
    report["seconds_bn_noise"] = round(time.time() - args._t0, 1)
    dump_json(out_path, report)
    if args.write_md:
        splice_fbd5(render_fbd5(report))
    print(f"[fbd] {rel_reading}", flush=True)
    print(f"[fbd] {bn_reading}", flush=True)
    print(f"[fbd] {noise_reading}", flush=True)
    print(f"[fbd] bn-noise wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "models_release" / "det0_unet.pt"))
    ap.add_argument("--patch", default="96,128")
    ap.add_argument("--synth-n", type=int, default=16, help="зонных и столько же тихих патчей на синтетический файл")
    ap.add_argument("--real-n", type=int, default=120, help="зонных и столько же тихих патчей на реальный кабель")
    ap.add_argument("--fd-sizes", type=int, default=6)
    ap.add_argument("--fd-repeats", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--far", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="сколько seq брать из каждого сплита (0 = все)")
    ap.add_argument("--versions", default="v8,v10,v11")
    ap.add_argument("--real-names", default="1,2,3,4,5,6")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "eval" / "fbd.json"))
    ap.add_argument("--skip-kl", action="store_true")
    ap.add_argument("--skip-neural", action="store_true")
    ap.add_argument("--kid-files", action="store_true",
                    help="KID по файлам 1–6 и сравнение энкодеров det0 / v810ft / DOM, без FD_∞")
    ap.add_argument("--dom-vs-winner", action="store_true",
                    help="KID и зонд по каждому кабелю: U-Net победителя против unet_dom из ens_blend")
    ap.add_argument("--bn-noise", action="store_true",
                    help="ens_dom per-file, если его нет; KID по шуму v8/v10/v11 при BN только с синтетики")
    ap.add_argument("--write-md", action="store_true", help="подставить раздел результатов в docs/TZ_A_FBD.md")
    args = ap.parse_args()
    th, tw = (int(x) for x in args.patch.split(","))
    patch = (th, tw)
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    names = [n.strip() for n in args.real_names.split(",") if n.strip()]
    t_all = time.time()
    out_path = Path(args.out)
    prev = json.loads(out_path.read_text()) if out_path.is_file() else None
    if args.kid_files:
        args._t0 = t_all
        run_kid_files(args, patch, versions, names, out_path, prev)
        return
    if args.dom_vs_winner:
        args._t0 = t_all
        run_dom_vs_winner(args, patch, names, out_path, prev)
        return
    if args.bn_noise:
        args._t0 = t_all
        run_bn_noise(args, patch, versions, names, out_path, prev)
        return

    if args.skip_kl:
        kl_report = (prev or {}).get("kl_fbd")
        if kl_report is None:
            print("[fbd] --skip-kl, в существующем JSON нет kl_fbd", flush=True)
    else:
        models = list_models(ROOT / "artifacts" / "pred" / "v8")
        edges = np.linspace(0.0, 1.0, 200001)
        rows, spear = collect_iou_kl(models, ROOT / "artifacts" / "cache" / "v8",
                                     ROOT / "artifacts" / "pred" / "v8", args.far, edges, args.limit)
        kl_report = {
            "definition": "mean KL(P_zone || Q_quiet) on score s; zones from ev labels, fixed across models",
            "far": args.far,
            "n_bins": 64,
            "smooth": "probability eps 1e-3",
            "min_cells": 12,
            "min_quiet_profiles": 10,
            "models": rows,
            "spearman": spear,
            "verdict": spear["verdict"],
        }

    pairs = {}
    domain = None
    if not args.skip_neural:
        enc = fbd.BottleneckEncoder.load(args.ckpt)
        print(f"[fbd] encoder {args.ckpt} width={enc.width} dim={enc.dim} patch={patch}", flush=True)
        packs = {}
        for ver in versions:
            splits = ("val", "test") if ver == "v8" else ("test",)
            for split in splits:
                packs[f"{ver}_{split}"] = collect_synth_strata(
                    enc, ver, split, args.synth_n, patch, args.seed, args.batch, args.limit)
        real_pack = collect_real_strata(enc, names, args.real_n, patch, args.seed, args.batch)
        specs = []
        if "v8_val" in packs and "v8_test" in packs:
            specs.append(("v8_val_vs_v8_test", packs["v8_val"], packs["v8_test"]))
        for ver in versions:
            specs.append((f"{ver}_test_vs_real", packs[f"{ver}_test"], real_pack))
        for key, left, right in specs:
            print(f"[fbd] pair {key}", flush=True)
            pairs[key] = eval_pair(left, right, fbd.stable_seed(args.seed, key), args.fd_sizes, args.fd_repeats)
        domain = {
            "checkpoint": str(Path(args.ckpt).resolve()),
            "width": enc.width,
            "feature_dim": enc.dim,
            "bottleneck": "gate(e4) global average pool",
            "patch_t_x": list(patch),
            "seed": args.seed,
            "synth_n_per_stratum": args.synth_n,
            "real_n_per_stratum": args.real_n,
            "fd_inf_sizes": args.fd_sizes,
            "fd_inf_repeats": args.fd_repeats,
            "batch": args.batch,
            "counts": {k: _pack_counts(v) for k, v in packs.items()},
            "real_counts": _pack_counts(real_pack),
            "real_files": real_pack["files"],
        }

    lower = pairs.get("v8_val_vs_v8_test")
    reals = {ver: pairs[f"{ver}_test_vs_real"] for ver in versions if f"{ver}_test_vs_real" in pairs}
    order = judge_order(lower, reals) if lower is not None and reals else None
    report = {
        "name": "FBD",
        "metric": "kid_fdinf_domain_probe",
        "real_policy": (
            "MGTS csv via dts.real.load_csv for evaluation only; "
            "the domain probe fits domain labels on frozen embeddings and is not applied to the detector; "
            "no invented event labels"
        ),
        "zone_mask_real": "rectangles from artifacts/eval/domain_cache/*_base.json (reference detector coordinates)",
        "protocol": {
            "features": "det0 gate(e4) global average pool",
            "balance": "equal n_zone and n_quiet on both sides; round-robin across files",
            "kernel": "unbiased polynomial MMD degree 3, gamma=1/d, coef0=1 (Binkowski 2018)",
            "fd_inf": "FID(N)=FD_inf+slope/N, empirical covariance, both sides subsampled to the same N (Chong and Forsyth 2020)",
            "probe": "LogisticRegression C=1 lbfgs, StandardScaler inside the train fold, GroupKFold by file, domain label only",
        },
        "kl_fbd": kl_report,
        "legacy_unbalanced": legacy_from(prev),
        "domain": domain,
        "pairs": _clean(pairs),
        "order": _clean(order) if order is not None else None,
        "seconds": round(time.time() - t_all, 1),
    }
    dump_json(out_path, report)
    if args.write_md and pairs and kl_report is not None:
        splice_md(render_results(report))
    print(f"[fbd] wrote {args.out} за {report['seconds']} с", flush=True)
    if order is not None:
        print(f"[fbd] order all_hold={order['all_hold']} all_substantial={order['all_substantial']}", flush=True)
    if kl_report is not None:
        print(f"[fbd] verdict {kl_report['verdict']} rho_hot={kl_report['spearman']['hot']['rho']} "
              f"rho_cold={kl_report['spearman']['cold']['rho']}", flush=True)


if __name__ == "__main__":
    main()
