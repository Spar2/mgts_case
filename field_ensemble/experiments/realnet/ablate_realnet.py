"""Абляция блоков realnet на v8 val. Порог каждого скора — FAR 1e-3 по фону этого же val.

Блоки:
  (а) unet_v810ft на каналах кэша, без доработки;
  (б) те же каналы после AdaBN по фону train;
  (в) на вход поданы каналы после вычитания своей базовой линии;
  (г) маскированная голова, остаток смешан со скором сегментации;
  (д) сочетания, включая базовую линию как дополнительный скор, а не замену входа.

Выбор конфигурации — по pixel IoU. Test здесь не читается.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dts import data, metrics, models
from experiments.realnet.realnet import (
    WIDE_SIGMA,
    MASK_SIGMA,
    adapt_batchnorm,
    combine_scores,
    component_score,
    load_pretrained,
    make_channels,
    predict_mask_residual,
    sample_tile,
    segmentation_head_tensors,
    train_mask_head,
    weight_grid,
)

import torch


def event(msg: str) -> None:
    line = time.strftime("%H:%M", time.gmtime()) + " realnet: " + msg
    print(line, flush=True)
    with open("/workspace/run_logs/_events.log", "a") as f:
        f.write(line + "\n")


def seg_map(model, X, valid):
    prob = models.predict_full(model.unet, np.ascontiguousarray(X, np.float32))
    return np.where(valid, 1.0 - prob[0], 0.0).astype(np.float32)


def wide_map(residual, sigma, valid):
    sig = np.maximum(np.asarray(sigma, np.float32), 0.03)
    rs = np.where(valid, np.asarray(residual, np.float32) / sig[None, :], 0.0)
    return component_score(rs, valid, WIDE_SIGMA)


def load_frame(path):
    z = np.load(path)
    T = np.asarray(z["T"], np.float32)
    valid = np.asarray(z["valid"], bool) & np.isfinite(T)
    x = np.asarray(z["x"], np.float64)
    t = np.asarray(z["t"], np.float64)
    dx = float(np.median(np.diff(x))) if x.size > 1 else 0.25
    Xb, info = make_channels(T, valid, t, x, dx=dx)
    return dict(name=path.stem, X=np.asarray(z["X"], np.float32), Xb=Xb, residual=info["residual"],
                sigma=info["sigma_x"], valid=valid, ev=np.asarray(z["ev"]), nu=np.asarray(z["nu"]))


def collect_train_tiles(n_files=12, n_tiles=4):
    rng = np.random.default_rng(0)
    paths = list(data.list_cache("artifacts/cache/v8", "train"))
    rng.shuffle(paths)
    v10 = list(data.list_cache("artifacts/cache/v10", "train"))
    rng10 = np.random.default_rng(1)
    rng10.shuffle(v10)
    n8 = min(8, n_files, len(paths))
    n10 = min(4, max(n_files - n8, 0), len(v10))
    tiles, weights = [], []
    for p in list(paths[:n8]) + list(v10[:n10]):
        z = np.load(p)
        X = np.asarray(z["X"], np.float32)
        valid = np.asarray(z["valid"], bool)
        bg = (valid & (np.asarray(z["ev"]) == 0) & (np.asarray(z["nu"]) == 0)).astype(np.float32)
        for _ in range(n_tiles):
            tile, w = sample_tile(X, rng, bg)
            tiles.append(tile)
            weights.append(w if w is not None else np.ones(tile.shape[-2:], np.float32))
    return tiles, weights


def save_maps(dirpath, name, **arrays):
    path = Path(dirpath) / f"{name}.npz"
    old = {}
    if path.exists():
        z = np.load(path)
        old = {k: np.asarray(z[k]) for k in z.files}
        z.close()
    for k, v in arrays.items():
        old[k] = np.asarray(v, np.float16)
    np.savez_compressed(path, **old)


def load_bank(dirpath):
    """Все карты val в памяти один раз: дальше смеси не перечитывают диск."""
    bank = []
    for p in sorted(Path(dirpath).glob("*.npz")):
        comp = np.load(p)
        z = np.load(Path("artifacts/cache/v8/val") / p.name)
        valid = np.asarray(z["valid"], bool)
        ev = np.asarray(z["ev"])
        nu = np.asarray(z["nu"])
        maps = {k: np.asarray(comp[k]) for k in comp.files}
        bank.append(dict(valid=valid, ev=ev, nu=nu, bg=valid & (ev == 0) & (nu == 0), maps=maps))
        comp.close()
        z.close()
    return bank


def frame_score(frame, seg_k, mask_k, wide_k, weights):
    shape = frame["valid"].shape
    zeros = None

    def take(key, w):
        nonlocal zeros
        if key is None or w == 0.0:
            if zeros is None:
                zeros = np.zeros(shape, np.float32)
            return zeros
        return frame["maps"][key]

    return combine_scores(take(seg_k, weights[0]), take(mask_k, weights[1]), take(wide_k, weights[2]), weights)


def metrics_at(frames, scores, thr):
    tot = np.zeros(3)
    far_n = far_d = 0
    sum_bg = sum_ev = n_bg = n_ev = 0.0
    for frame, score in zip(frames, scores):
        valid, ev, bg = frame["valid"], frame["ev"], frame["bg"]
        pred = (score >= thr) & valid
        tot += metrics.confusion(pred, ev > 0, valid)
        far_n += int((pred & bg).sum())
        far_d += int(bg.sum())
        sum_bg += float(score[bg].sum())
        n_bg += int(bg.sum())
        evm = valid & (ev > 0)
        sum_ev += float(score[evm].sum())
        n_ev += int(evm.sum())
    pr = metrics.prf(*tot)
    pr.update(threshold=float(thr), far=float(far_n / max(far_d, 1)),
              mean_bg=float(sum_bg / max(n_bg, 1)), mean_event=float(sum_ev / max(n_ev, 1)))
    return pr


def best_mixture(bank, seg_k, mask_k, wide_k, allow_mask, allow_wide):
    """Лучший вес по IoU. Ничья остаётся за более ранней точкой сетки (больший вес головы)."""
    cands = [w for w in weight_grid() if (allow_mask or w[1] == 0.0) and (allow_wide or w[2] == 0.0)]
    packed = []
    for w in cands:
        scores = [frame_score(fr, seg_k, mask_k, wide_k, w) for fr in bank]
        bg = np.concatenate([np.asarray(s[fr["bg"]], np.float32).ravel() for s, fr in zip(scores, bank)])
        thr = float(metrics.far_threshold(bg, 1e-3))
        del bg
        rec = metrics_at(bank, scores, thr)
        rec["weights"] = list(w)
        packed.append(rec)
        del scores
    best = packed[0]
    for rec in packed[1:]:
        if rec["iou"] > best["iou"] + 1e-6:
            best = rec
    return best


def main():
    torch.set_num_threads(2)
    out_dir = Path("artifacts/realnet/ablation_v8val")
    if out_dir.exists():
        for p in out_dir.glob("*.npz"):
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = data.list_cache("artifacts/cache/v8", "val")
    event(f"абляция v8 val, {len(paths)} файлов")
    ckpt = "artifacts/stage1/unet_v810ft.pt"

    model = load_pretrained(ckpt)
    head = segmentation_head_tensors(model.unet)
    t0 = time.time()
    for i, p in enumerate(paths):
        fr = load_frame(p)
        save_maps(out_dir, fr["name"],
                  seg_raw=seg_map(model, fr["X"], fr["valid"]),
                  seg_base=seg_map(model, fr["Xb"], fr["valid"]),
                  wide=wide_map(fr["residual"], fr["sigma"], fr["valid"]))
        if (i + 1) % 5 == 0 or i + 1 == len(paths):
            print(f"pass A {i+1}/{len(paths)} {time.time()-t0:.0f}s", flush=True)
        del fr
    w2, b2 = segmentation_head_tensors(model.unet)
    if not torch.equal(head[0], w2) or not torch.equal(head[1], b2):
        raise RuntimeError("голова сегментации сдвинулась на проходе A")

    tiles, tw = collect_train_tiles()
    adapt_batchnorm(model.unet, tiles, batch=4)
    w2, b2 = segmentation_head_tensors(model.unet)
    if not torch.equal(head[0], w2) or not torch.equal(head[1], b2):
        raise RuntimeError("AdaBN сдвинул голову сегментации")
    t1 = time.time()
    for i, p in enumerate(paths):
        fr = load_frame(p)
        save_maps(out_dir, fr["name"],
                  seg_adabn=seg_map(model, fr["X"], fr["valid"]),
                  seg_base_adabn=seg_map(model, fr["Xb"], fr["valid"]))
        if (i + 1) % 5 == 0 or i + 1 == len(paths):
            print(f"pass B {i+1}/{len(paths)} {time.time()-t1:.0f}s", flush=True)
        del fr
    del model

    model = load_pretrained(ckpt)
    hist = train_mask_head(model, tiles, tw, minutes=2.0, seed=0, batch=4)
    w2, b2 = segmentation_head_tensors(model.unet)
    if not torch.equal(head[0], w2) or not torch.equal(head[1], b2):
        raise RuntimeError("маска сдвинула голову сегментации")
    t2 = time.time()
    for i, p in enumerate(paths):
        z = np.load(p)
        X = np.asarray(z["X"], np.float32)
        valid = np.asarray(z["valid"], bool)
        # остаток маски — на тех же каналах, на которых голова училась
        resid = predict_mask_residual(model, X)
        save_maps(out_dir, p.stem, mask=component_score(resid, valid, MASK_SIGMA))
        if (i + 1) % 5 == 0 or i + 1 == len(paths):
            print(f"pass C {i+1}/{len(paths)} {time.time()-t2:.0f}s", flush=True)
        del z, resid
    del model, tiles, tw

    # Каждая строка — один вопрос абляции. Смеси перебирают симплекс только по включённым блокам.
    questions = [
        ("a_unet", "seg_raw", None, None, False, False, "исходный U-Net, каналы кэша"),
        ("b_adabn", "seg_adabn", None, None, False, False, "AdaBN на тех же каналах"),
        ("c_baseline_input", "seg_base", None, None, False, False, "вход заменён остатком базовой линии"),
        ("d_mask", "seg_raw", "mask", None, True, False, "маска добавлена к исходному U-Net"),
        ("e_adabn_baseline", "seg_base_adabn", None, None, False, False, "AdaBN и замена входа"),
        ("e_adabn_mask", "seg_adabn", "mask", None, True, False, "AdaBN и маска, вход исходный"),
        ("e_baseline_wide", "seg_base", None, "wide", False, True, "замена входа и широкий скор"),
        ("e_wide_extra", "seg_raw", None, "wide", False, True, "базовая линия только доп. скором"),
        ("e_adabn_wide_extra", "seg_adabn", None, "wide", False, True, "AdaBN и базовая линия как доп. скор"),
        ("e_all_original_input", "seg_raw", "mask", "wide", True, True, "маска и доп. скор, вход исходный"),
        ("e_shipped_style", "seg_base_adabn", "mask", "wide", True, True, "как прогон: AdaBN, замена входа, маска, широкий скор"),
    ]
    bank = load_bank(out_dir)
    table = []
    for name, seg_k, mask_k, wide_k, allow_mask, allow_wide, note in questions:
        tq = time.time()
        best = best_mixture(bank, seg_k, mask_k, wide_k, allow_mask, allow_wide)
        best = dict(name=name, note=note, **{k: best[k] for k in
                    ("weights", "threshold", "iou", "f1", "precision", "recall", "far", "mean_bg", "mean_event")})
        table.append(best)
        print(f"ROW {name} iou {best['iou']:.4f} w {best['weights']} thr {best['threshold']:.4f} "
              f"P {best['precision']:.3f} R {best['recall']:.3f} ({time.time()-tq:.0f}s) {note}", flush=True)

    # Конфигурация: максимальный IoU. Замена входа допускается, только если она строго лучше
    # лучшего варианта без замены. При ничьей — больший вес головы, затем вариант без AdaBN и без маски.
    def rank(row):
        replaces = row["name"] in {"c_baseline_input", "e_adabn_baseline", "e_baseline_wide", "e_shipped_style"}
        w = row["weights"]
        return (row["iou"], 0 if replaces else 1, w[0], 0 if w[1] else 1, 0 if row["name"].startswith("b_") or "adabn" in row["name"] else 1)

    chosen = max(table, key=rank)
    # Отдельное сравнение замены входа с исходным U-Net, чтобы вывод не зависел от ранжирования.
    by = {r["name"]: r for r in table}
    baseline_hurts = bool(by["c_baseline_input"]["iou"] + 1e-4 < by["a_unet"]["iou"])
    adabn_hurts = bool(by["b_adabn"]["iou"] + 1e-4 < by["a_unet"]["iou"])
    blob = dict(split="v8_val", n_files=len(paths), far_target=1e-3, ckpt=ckpt,
                mask_steps=hist[-1].get("step") if hist else 0,
                baseline_input_hurts=baseline_hurts, adabn_hurts=adabn_hurts,
                chosen=chosen, rows=table)
    dst = Path("artifacts/eval/realnet_ablation.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob, ensure_ascii=False, indent=2))
    tmp.replace(dst)
    removed = 0
    for p in out_dir.glob("*.npz"):
        p.unlink()
        removed += 1
    event(f"абляция готова: chosen {chosen['name']} IoU {chosen['iou']:.4f} веса {chosen['weights']}; "
          f"замена входа вредит={baseline_hurts} AdaBN вредит={adabn_hurts}; карты удалены ({removed})")
    print("ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
