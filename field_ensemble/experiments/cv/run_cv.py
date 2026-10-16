"""Предсказания cv_ref / cv_pipe / unet_cv_post и оценка карточек оператора.

Скор пишется в artifacts/pred/<ver>/<name>/<split>/<seq>.npz (s float32, h float16).
Карточки и реальные 1–6.csv только оцениваются: порог берётся из уже посчитанного
val FAR 1e-3, на реальных файлах ничего не подбирается.

Потоки: OMP и cv2 ограничены --threads (по умолчанию 2).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dts import data, events as EV, grid
from experiments.cv import cv_pipe, cv_ref


def _threads(n: int) -> None:
    n = str(int(n))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["CV_NUM_THREADS"] = n
    try:
        import cv2
        cv2.setNumThreads(int(n))
    except ImportError:
        pass


def _quantize(score: np.ndarray, step: float) -> np.ndarray:
    """Округление перед записью: zip почти не жмёт сырой float32.

    0.05 °C и 0.001 вероятности на контрольных записях сдвигают пофайловый
    IoU меньше чем на 0.004, а файл меньше в десятки раз. Ключ s остаётся float32.
    """
    step = np.float32(step)
    values = np.asarray(score, np.float32)
    return (np.round(values / step) * step).astype(np.float32)


def predict_split(cache: Path, split: str, methods: list[str], limit: int) -> None:
    ver = cache.name
    files = data.list_cache(cache, split)
    if limit:
        files = files[:limit]
    for method in methods:
        out_dir = Path("artifacts/pred") / ver / method / split
        out_dir.mkdir(parents=True, exist_ok=True)
        done = 0
        t0 = time.time()
        for path in files:
            dst = out_dir / path.name
            if dst.exists():
                done += 1
                continue
            z = np.load(path)
            T = z["T"]
            valid = z["valid"]
            x, t, sigma = z["x"], z["t"], z["sigma"]
            X = z["X"]
            if method == "cv_ref":
                s, h = cv_ref.score_sequence(T, valid, x, t, sigma)
            elif method == "cv_pipe":
                s, h = cv_pipe.score_sequence(T, valid, x, t, sigma, X)
            elif method == "unet_cv_post":
                src = Path("artifacts/pred") / ver / "unet" / split / path.name
                if not src.exists():
                    raise FileNotFoundError(src)
                q = np.load(src)
                s, h = cv_pipe.postprocess_unet(q["s"], q["h"], T, valid, x, t, sigma, X)
            else:
                raise SystemExit(f"unknown method {method}")
            step = 0.001 if method == "unet_cv_post" else 0.05
            np.savez_compressed(dst, s=_quantize(s, step), h=h.astype(np.float16))
            done += 1
            print(method, split, path.stem, done, len(files), "s", round(time.time() - t0), flush=True)
        print(method, split, "done", done, round(time.time() - t0), flush=True)


def _threshold(eval_json: Path, method: str) -> float:
    report = json.loads(eval_json.read_text())
    return float(report["thresholds"][method]["far_1e-3"])


def _score_real(method: str, field: dict):
    T, valid, x, t = field["T"], field["valid"], field["x"], field["t"]
    p = grid.plan(x, t)
    Tg, vg = grid.resample(T, valid, p, "mean")
    from dts import features
    X, aux = features.make_channels(Tg, vg)
    sigma = aux["sigma_x"]
    if method == "cv_ref":
        s, h = cv_ref.score_sequence(Tg, vg, p["x"], p["t"], sigma)
    elif method == "cv_pipe":
        s, h = cv_pipe.score_sequence(Tg, vg, p["x"], p["t"], sigma, X)
    elif method == "unet_cv_post":
        raise SystemExit("карточки U-Net+CV считают по уже ресемплированному скору Analyzer; здесь только cv_ref и cv_pipe")
    else:
        raise SystemExit(method)
    return s, h, vg, p, X


def eval_cards(methods: list[str], eval_json: Path, real_root: Path, card_paths: list[str], out: Path) -> None:
    import glob
    import pandas as pd
    from dts import baseline, real
    cards = []
    for c in card_paths:
        cards += [json.loads(line) for line in open(c) if line.strip()]
    files = sorted(glob.glob(str(real_root / "*.csv")))
    report = {"op": "far_1e-3", "threshold_source": str(eval_json), "note": "порог с синтетического val, карточки в порог не входят", "methods": {}}
    for method in methods:
        if method == "unet_cv_post":
            continue
        thr = _threshold(eval_json, method)
        block = {"threshold": thr, "files": {}, "cards": []}
        for f in files:
            raw = real.load_csv(f)
            name = raw["name"]
            t0 = time.time()
            s, h, vg, p, _X = _score_real(method, raw)
            zones = EV.zones_from_prob(s, h, vg, p["x"], p["t"], thr)
            b = baseline.run(raw["T"], raw["x"], raw["t"], valid=raw["valid"])
            bz, _bw = grid.resample(b["zone"], np.ones_like(raw["valid"]), p, "nearest")
            cnn = (s >= thr) & vg
            base_mask = (bz > 0) & vg
            inter = int((cnn & base_mask).sum())
            uni = int((cnn | base_mask).sum())
            block["files"][name] = dict(
                n_zones=len(zones),
                n_zones_base=len(b["zones"]),
                alarm_fraction=float(cnn[vg].mean()) if vg.any() else 0.0,
                iou_vs_base=float(inter / max(uni, 1)),
                seconds=round(time.time() - t0, 1),
            )
            print(method, name, block["files"][name], flush=True)
            E0 = pd.Timestamp("1970-01-01", tz="UTC")
            for c in [c for c in cards if str(c.get("record")) == name]:
                ts = [(pd.Timestamp(v).tz_convert("UTC") if pd.Timestamp(v).tzinfo else pd.Timestamp(v, tz="UTC")) for v in c["time"]]
                ta, tb = [(v - E0).total_seconds() for v in ts]
                xa, xb = c["meters"]
                ix = (p["x"] >= xa) & (p["x"] <= xb)
                it = (p["t"] >= ta) & (p["t"] <= tb)
                if not ix.any() or not it.any():
                    block["cards"].append(dict(id=c["id"], record=name, status="box_outside_record", presence=c["presence"]))
                    continue
                box = np.ix_(it, ix)
                vb = vg[box]
                overl = [z for z in zones if z["end_m"] >= xa and z["start_m"] <= xb and z["t_end_s"] >= ta and z["t_start_s"] <= tb]
                block["cards"].append(dict(
                    id=c["id"], record=name, presence=c["presence"], shape=c.get("shape", ""),
                    max_score=float(s[box][vb].max()) if vb.any() else None,
                    frac_flagged=float((s[box] >= thr)[vb].mean()) if vb.any() else None,
                    n_zones=len(overl),
                    hotness_mean=float(h[box][vb].mean()) if vb.any() else None,
                ))
        yes = [c for c in block["cards"] if c.get("presence") == "yes" and "frac_flagged" in c]
        no = [c for c in block["cards"] if c.get("presence") == "no" and "frac_flagged" in c]
        block["summary"] = dict(
            n_cards=len(block["cards"]),
            yes_hit=float(np.mean([c["frac_flagged"] >= 0.05 or c["n_zones"] > 0 for c in yes])) if yes else None,
            no_quiet=float(np.mean([c["frac_flagged"] < 0.05 and c["n_zones"] == 0 for c in no])) if no else None,
            n_yes=len(yes), n_no=len(no),
        )
        report["methods"][method] = block
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("wrote", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="artifacts/cache/v8")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--methods", default="cv_ref,cv_pipe,unet_cv_post")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cards", action="store_true")
    ap.add_argument("--eval-json", default="artifacts/eval/cv_v8.json")
    ap.add_argument("--real-root", default="/workspace/mgts/MGTS_data")
    ap.add_argument("--cards-out", default="artifacts/eval/cv_cards.json")
    ap.add_argument("--card-paths", nargs="*", default=["data/operator_cards/cards_batch1.jsonl", "data/operator_cards/cards_batch2.jsonl"])
    a = ap.parse_args()
    _threads(a.threads)
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    if a.cards:
        methods = [m for m in a.methods.split(",") if m and m != "unet_cv_post"]
        eval_cards(methods, Path(a.eval_json), Path(a.real_root), a.card_paths, Path(a.cards_out))
        return
    cache = Path(a.cache)
    methods = [m for m in a.methods.split(",") if m]
    for split in a.splits.split(","):
        if not split:
            continue
        predict_split(cache, split, methods, a.limit)


if __name__ == "__main__":
    main()
