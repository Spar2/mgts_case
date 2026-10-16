"""Model soup (Wortsman et al., ICML 2022) без градиентного обучения.

Берутся U-Net ширины 16 из artifacts/stage1/unet*.pt с одинаковыми ключами.
Жадный суп: модели по убыванию своего val IoU; кандидат добавляется, если
равномерное среднее весов не снижает IoU@FAR 1e-3 на всём v8 val.
У смеси из двух и более running-статистика BatchNorm пересчитывается на тайлах
synth v8 train. Affine BN остаётся усреднённым. Градиентов нет.

В ансамбле ens_u2ft_tcn_lgbm участник unet_v810ft заменяется супом:
new = ens + (soup - unet_v810ft) / 4.
Порог new и победителя — только фон v8 val. v10/v11 его не двигают.
Вложенные k=2 и k=5 — та же процедура, что scripts/s3_sel2_cv.py.
Промоушн: прирост min(k) ≥ 0.003 и оба внешних test не хуже победителя.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

import s3_sel2_cv as cv
import s3_sel_core as sel
import s3_sel_eval as sev
from dts import data, metrics as M, models

WINNER_VAL = 0.38671235821914546
WINNER_V10 = 0.4038977883892501
WINNER_V11 = 0.40390028295846503
BASE = "unet_v810ft"
MEMBERS = ("unet_w24", "unet_v810ft", "tcn", "pixel_lgbm")
STAGE = Path("artifacts/stage1")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _predict():
    spec = importlib.util.spec_from_file_location("s3_dom_eval", ROOT / "scripts" / "s3_dom_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict_batched


def load_sd(path: Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    keys = list(sd)
    if keys and all(k.startswith("net.") for k in keys):
        sd = {k[4:]: v for k, v in sd.items()}
    return sd


def average(sds: list[dict]) -> dict:
    keys = list(sds[0])
    out = {}
    n = float(len(sds))
    for k in keys:
        vs = [s[k] for s in sds]
        if vs[0].is_floating_point():
            acc = vs[0].float().clone()
            for v in vs[1:]:
                acc += v.float()
            out[k] = (acc / n).to(dtype=vs[0].dtype)
        else:
            out[k] = torch.zeros_like(vs[0])
    return out


def build(sd: dict):
    model = models.build("unet", 16)
    model.load_state_dict(sd)
    model.eval()
    return model


_SAMPLER = None


def train_sampler(train_items, seed: int = 0):
    """Один TileSampler на все смеси: встроенные зеркало по x и gain — как при обучении stage-1."""
    global _SAMPLER
    if _SAMPLER is None:
        _SAMPLER = data.TileSampler(train_items, tile=(128, 256), seed=seed, p_weak=0.0, aug=None)
    return _SAMPLER


def refresh_bn(model, train_items, n_batches: int = 32, batch: int = 8, seed: int = 0) -> None:
    """Running-статистика BN с нуля по тайлам v8 train. Affine и свёртки не меняются."""
    model.eval()
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None
            mod.train()
    sampler = train_sampler(train_items, seed=seed)
    with torch.no_grad():
        for _ in range(n_batches):
            x, _y = sampler.sample(batch)
            model(torch.from_numpy(np.ascontiguousarray(x)))
    model.eval()
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.momentum = 0.1


def score_stored(rows):
    parts = []
    for _stem, s, ev, valid, nu in rows:
        parts.append(np.asarray(s[valid & (ev == 0) & (nu == 0)], np.float32))
    bg = np.concatenate(parts)
    thr = float(M.far_threshold(bg, 1e-3))
    tot = np.zeros(3, np.int64)
    far_n = 0
    for _stem, s, ev, valid, nu in rows:
        pred = (s >= thr) & valid
        tot += np.asarray(M.confusion(pred, ev > 0, valid), np.int64)
        far_n += int((pred & valid & (ev == 0) & (nu == 0)).sum())
    row = dict(threshold=thr, far=float(far_n / max(bg.size, 1)), n_bg=int(bg.size), **M.prf(*tot))
    return row


def predict_val(model, predict, files, tag: str):
    t0 = time.time()
    rows = []
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"], bool)
            nu = np.asarray(z["nu"])
        prob = predict(model, X)
        s = (1.0 - prob[0]).astype(np.float32)
        rows.append((p.stem, s, ev, valid, nu))
        del X, prob
        if i == 1 or i % 15 == 0 or i == len(files):
            log(f"{tag} {i}/{len(files)} {time.time() - t0:.0f}s")
    return rows


def rows_from_dir(files, pred_dir: Path):
    if not pred_dir.is_dir():
        return None
    rows = []
    for p in files:
        q = pred_dir / p.name
        if not q.is_file():
            return None
        with np.load(p) as z:
            ev = np.asarray(z["ev"])
            valid = np.asarray(z["valid"], bool)
            nu = np.asarray(z["nu"])
        with np.load(q) as w:
            s = np.asarray(w["s"], np.float32)
        rows.append((p.stem, s, ev, valid, nu))
    return rows


def iou_from_dir(files, pred_dir: Path):
    rows = rows_from_dir(files, pred_dir)
    if rows is None:
        return None
    return score_stored(rows)


def check_predict_matches_disk(model, predict, files) -> float:
    """Один кадр: predict_batched против сохранённой карты unet_v810ft."""
    p = files[0]
    with np.load(p) as z:
        X = np.asarray(z["X"], np.float32)
    pred = (1.0 - predict(model, X)[0]).astype(np.float32)
    disk = load_s("v8", BASE, "val", p.stem)
    err = float(np.max(np.abs(pred - disk)))
    log(f"сверка predict vs {BASE} {p.stem} max|Δ| {err:.3e}")
    if err > 1e-3:
        raise SystemExit(f"predict не совпал с картой {BASE}: {err}")
    return err


def load_s(ver, name, split, stem):
    path = Path(f"artifacts/pred/{ver}/{name}/{split}/{stem}.npz")
    with np.load(path) as z:
        return np.asarray(z["s"], np.float32)


def check_ensemble(files) -> float:
    p = files[len(files) // 2]
    ss = [load_s("v8", m, "val", p.stem) for m in MEMBERS]
    ens = load_s("v8", "ens_u2ft_tcn_lgbm", "val", p.stem)
    err = float(np.max(np.abs(sum(ss) / 4.0 - ens)))
    if err > 1e-5:
        raise SystemExit(f"ансамбль не среднее четырёх участников, max|Δ|={err}")
    log(f"сверка ансамбля {p.stem} max|Δ| {err:.3e}")
    return err


def replace_scores(ver, split, stem, soup_s):
    ens = load_s(ver, "ens_u2ft_tcn_lgbm", split, stem)
    base = load_s(ver, BASE, split, stem)
    if ens.shape != soup_s.shape or base.shape != soup_s.shape:
        raise SystemExit(f"{ver}/{split}/{stem} shape soup {soup_s.shape} ens {ens.shape} base {base.shape}")
    return (ens + (soup_s - base) / 4.0).astype(np.float32), ens


def as_frames(rows_ens, rows_new):
    out = []
    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    cache = {p.stem: p for p in data.list_cache("artifacts/cache/v8", "val")}
    by_new = {r[0]: r for r in rows_new}
    for stem, s_ens, ev, valid, nu in rows_ens:
        s_new = by_new[stem][1]
        with np.load(cache[stem]) as z:
            x = np.asarray(z["x"], np.float64)
            t = np.asarray(z["t"], np.float64)
        ev_idx, nu_idx, bg_idx, n_bg = sev._indices(ev, valid, nu, stem)
        ee, en, eb = sev._take(s_ens, ev_idx, nu_idx, bg_idx)
        ne, nn, nb = sev._take(s_new, ev_idx, nu_idx, bg_idx)
        cooling, _ = sev._events_for(stem, ev, x, t, labels, seqs, ev_idx)
        out.append(dict(
            stem=stem, n_bg=n_bg, cooling=cooling,
            ens_ev=ee.astype(np.float16), ens_nu=en.astype(np.float16), ens_bg=eb.astype(np.float16),
            new_ev=ne.astype(np.float16), new_nu=nn.astype(np.float16), new_bg=nb.astype(np.float16),
        ))
    out.sort(key=lambda fr: fr["stem"])
    return out


def frames_of(rows, key):
    made = []
    for fr in rows:
        made.append(dict(
            ev_m=np.asarray(fr[key + "_ev"], np.float32)[None],
            nu_m=np.asarray(fr[key + "_nu"], np.float32)[None],
            bg_m=np.asarray(fr[key + "_bg"], np.float32)[None],
            n_bg=fr["n_bg"], cooling=fr["cooling"],
        ))
    return made


def nested(rows, key):
    made = frames_of(rows, key)
    out = {}
    for k in (2, 5):
        parts, details = [], []
        for r, hold_idx in enumerate(sel.sequence_folds(len(made), k)):
            hold = sel.evaluate_frames(cv.take(made, hold_idx), np.array([1.0]))
            parts.append(hold)
            details.append(dict(fold=r, n_seq=len(hold_idx), hold=cv.pack_metrics(hold)))
            log(f"nested {key} k={k} fold {r} IoU {hold['iou']:.4f}")
        micro = sel.micro_iou(parts)
        out[str(k)] = dict(micro_iou=micro, folds=details)
        log(f"nested {key} k={k} micro {micro:.4f}")
    return dict(by_k=out, verdict=sel.nested_verdict({k: v["micro_iou"] for k, v in out.items()}))


def eval_transfer(model, predict, thr_new, thr_ens):
    report = {}
    for ver in ("v10", "v11"):
        files = data.list_cache(f"artifacts/cache/{ver}", "test")
        acc = {k: np.zeros(3, np.int64) for k in ("new", "ens")}
        far = {k: [0, 0] for k in acc}
        t0 = time.time()
        for i, p in enumerate(files, 1):
            with np.load(p) as z:
                X = np.asarray(z["X"], np.float32)
                ev = np.asarray(z["ev"])
                valid = np.asarray(z["valid"], bool)
                nu = np.asarray(z["nu"])
            prob = predict(model, X)
            soup = (1.0 - prob[0]).astype(np.float32)
            new_s, ens = replace_scores(ver, "test", p.stem, soup)
            bg = valid & (ev == 0) & (nu == 0)
            truth = ev > 0
            for key, s, thr in (("new", new_s, thr_new), ("ens", ens, thr_ens)):
                pred = (s >= thr) & valid
                acc[key] += np.asarray(M.confusion(pred, truth, valid), np.int64)
                far[key][0] += int((pred & bg).sum())
                far[key][1] += int(bg.sum())
            del X, prob
            if i == 1 or i % 15 == 0 or i == len(files):
                log(f"{ver} {i}/{len(files)} {time.time() - t0:.0f}s")
        report[ver] = {}
        for key, thr in (("ens_soup", "new"), ("ens_u2ft_tcn_lgbm", "ens")):
            src = "new" if key == "ens_soup" else "ens"
            t = thr_new if src == "new" else thr_ens
            report[ver][key] = dict(
                threshold=float(t),
                far=float(far[src][0] / max(far[src][1], 1)),
                n_bg=int(far[src][1]),
                **M.prf(*acc[src]),
            )
        log(f"{ver} soup {report[ver]['ens_soup']['iou']:.4f} winner {report[ver]['ens_u2ft_tcn_lgbm']['iou']:.4f}")
    return report


def main() -> None:
    torch.set_num_threads(2)
    predict = _predict()
    files = data.list_cache("artifacts/cache/v8", "val")
    ens_err = check_ensemble(files)
    ref_keys = None
    ref_sd = None
    bank = {}
    skipped = []
    for path in sorted(STAGE.glob("unet*.pt")):
        sd = load_sd(path)
        w = int(sd["e1.0.weight"].shape[0])
        if w != 16:
            skipped.append(dict(name=path.name, why=f"width {w}"))
            continue
        if ref_keys is None:
            ref_keys = set(sd)
            ref_sd = sd
        elif set(sd) != ref_keys or any(tuple(sd[k].shape) != tuple(ref_sd[k].shape) for k in sd):
            skipped.append(dict(name=path.name, why="keys or shapes differ"))
            continue
        bank[path.stem] = sd
    log(f"кандидаты {sorted(bank)} пропущены {skipped}")
    anchor = build(bank[BASE])
    pred_err = check_predict_matches_disk(anchor, predict, files)
    del anchor

    solo_path = Path("artifacts/eval/s3_soup_solo.json")
    cached = {}
    if solo_path.is_file():
        cached = {r["name"]: r for r in json.loads(solo_path.read_text())}
    solo = []
    for name, sd in bank.items():
        if name in cached and set(cached) >= set(bank):
            disk = cached[name]
            log(f"solo {name} IoU {disk['iou']:.4f} из кэша")
        else:
            disk = iou_from_dir(files, Path("artifacts/pred/v8") / name / "val")
            if disk is None:
                log(f"нет карт {name}, считаю")
                model = build(sd)
                disk = score_stored(predict_val(model, predict, files, name))
                del model
            disk = dict(name=name, iou=disk["iou"], threshold=disk["threshold"], far=disk["far"],
                        source="disk" if (Path("artifacts/pred/v8") / name / "val").is_dir() else "predict")
            log(f"solo {name} IoU {disk['iou']:.4f}")
        solo.append(disk if "name" in disk else disk)
    if not (cached and set(cached) >= set(bank)):
        solo_path.write_text(json.dumps(plain(solo), indent=1))
        log(f"solo записан {solo_path}")
    solo.sort(key=lambda r: -r["iou"])

    # train tiles only if a mix is accepted
    train_items = None

    def materialize(names):
        nonlocal train_items
        sd = average([bank[n] for n in names]) if len(names) > 1 else bank[names[0]]
        model = build(sd)
        if len(names) > 1:
            if train_items is None:
                mm = Path("artifacts/mm/v8/train")
                missing = [p.name for p in Path("artifacts/cache/v8/train").glob("seq*.npz") if not (mm / f"{p.stem}_y.npy").exists()]
                if missing:
                    raise SystemExit(f"нет mmap y, новый кэш не пишем: {len(missing)}")
                log("mmap train для BN")
                train_items = data.mmap_items("artifacts/cache/v8", "train", Path("artifacts/mm/v8"))
                log(f"mmap train {len(train_items)}")
            t0 = time.time()
            log(f"BN старт {len(names)} моделей")
            refresh_bn(model, train_items)
            log(f"BN {len(names)} моделей за {time.time() - t0:.0f}s")
        return model

    chosen = [solo[0]["name"]]
    disk_rows = rows_from_dir(files, Path("artifacts/pred/v8") / chosen[0] / "val")
    if disk_rows is not None:
        model = build(bank[chosen[0]])
        rows = disk_rows
        log(f"старт супа {chosen[0]} с дисковых карт")
    else:
        model = materialize(chosen)
        rows = predict_val(model, predict, files, "soup:" + "+".join(chosen))
    best = score_stored(rows)
    log(f"старт супа {chosen[0]} IoU {best['iou']:.4f}")
    greedy = [dict(added=chosen[0], kept=True, ingredients=list(chosen), iou=best["iou"], threshold=best["threshold"])]
    for cand in solo[1:]:
        trial_names = chosen + [cand["name"]]
        trial = materialize(trial_names)
        trows = predict_val(trial, predict, files, "try:" + cand["name"])
        got = score_stored(trows)
        keep = got["iou"] >= best["iou"]
        log(f"try {cand['name']} IoU {got['iou']:.4f} prev {best['iou']:.4f} keep {keep}")
        greedy.append(dict(added=cand["name"], kept=bool(keep), ingredients=list(trial_names), iou=got["iou"], threshold=got["threshold"]))
        if keep:
            chosen = trial_names
            best = got
            rows = trows
            model = trial
        else:
            del trial, trows
    log(f"суп {chosen} IoU {best['iou']:.4f}")
    out_pt = Path("artifacts/eval/s3_soup_unet.pt")
    tmp = out_pt.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, out_pt)
    log(f"веса {out_pt}")

    # ансамбль на тех же картах val
    ens_rows = []
    new_rows = []
    for stem, s, ev, valid, nu in rows:
        new_s, ens = replace_scores("v8", "val", stem, s)
        ens_rows.append((stem, ens, ev, valid, nu))
        new_rows.append((stem, new_s, ev, valid, nu))
    ens_val = score_stored(ens_rows)
    new_val = score_stored(new_rows)
    log(f"val ensemble soup {new_val['iou']:.4f} winner {ens_val['iou']:.4f}")
    compact = as_frames(ens_rows, new_rows)
    nest_w = nested(compact, "ens")
    nest_n = nested(compact, "new")
    gain = float(nest_n["verdict"]["worst"] - nest_w["verdict"]["worst"])
    log(f"nested gain {gain:.4f}")
    transfer = eval_transfer(model, predict, new_val["threshold"], ens_val["threshold"])
    v10_ok = transfer["v10"]["ens_soup"]["iou"] + 1e-6 >= transfer["v10"]["ens_u2ft_tcn_lgbm"]["iou"]
    v11_ok = transfer["v11"]["ens_soup"]["iou"] + 1e-6 >= transfer["v11"]["ens_u2ft_tcn_lgbm"]["iou"]
    sanity = abs(ens_val["iou"] - WINNER_VAL) < 0.005
    promote = bool(sanity and gain >= 0.003 and v10_ok and v11_ok)
    report = dict(
        name="SOUP",
        paper="Wortsman et al., Model soups, ICML 2022",
        width=16,
        skipped=skipped,
        replaced_member=BASE,
        ensemble="(ens_u2ft_tcn_lgbm * 4 - unet_v810ft + soup) / 4",
        ensemble_check_max_abs=ens_err,
        predict_vs_disk_max_abs=pred_err,
        ingredients=chosen,
        solo=solo,
        greedy=greedy,
        soup_val=best,
        bn="running stats recomputed on 32x8 v8 train tiles when the soup has 2+ models; TileSampler keeps its built-in x-flip and gain; affine is the weight average; no gradients; single-model soup keeps the checkpoint BN",
        val=dict(ens_soup=new_val, ens_u2ft_tcn_lgbm=ens_val),
        nested_winner=nest_w,
        nested_soup=nest_n,
        nested_gain=gain,
        nested_protocol="sequence_folds, k=2 and k=5, holdout FAR threshold, micro IoU, same as scripts/s3_sel2_cv.py. Ingredient set was chosen on all v8 val, not re-fit inside the fold.",
        v10_test=transfer["v10"],
        v11_test=transfer["v11"],
        published={"val": WINNER_VAL, "v10": WINNER_V10, "v11": WINNER_V11},
        sanity_ok=bool(sanity),
        v10_not_worse=bool(v10_ok),
        v11_not_worse=bool(v11_ok),
        promote=promote,
        real_used=False,
        ckpt=str(out_pt),
    )
    out = Path("artifacts/eval/s3_soup_v8.json")
    out.write_text(json.dumps(plain(report), indent=1, ensure_ascii=False))
    log(f"promote {promote} записан {out}")


if __name__ == "__main__":
    main()
