"""Дообучение U-Net синтетическим шумовым априором (конвейер P).

Реальные CSV, real_zones и real_noise_psd.npz не читаются.
Старт — artifacts/stage1/unet_v810ft.pt (тот же слот, что у DOM, не det1).
Чекпойнт — максимум proxy IoU@FAR 1e-3 на первых 8 последовательностях v8 val.
Официальные 45 и вложенные фолды считает scripts/s3_dg_eval.py.

  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \\
  /workspace/repo/run/.venv/bin/python -u scripts/s3_dg_train.py
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import aug_dg, data, losses, metrics as M, models


def _eval_mod():
    spec = importlib.util.spec_from_file_location("s3_dg_eval", ROOT / "scripts" / "s3_dg_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def assert_mmap(ver: str, split: str) -> None:
    cache = Path("artifacts/cache") / ver / split
    mm = Path("artifacts/mm") / ver / split
    missing = [p.name for p in sorted(cache.glob("seq*.npz")) if not (mm / f"{p.stem}_y.npy").exists()]
    if missing:
        raise SystemExit(f"нет mmap {ver}/{split}: {len(missing)} файлов, новый кэш не пишу")


def assert_no_real_stats() -> None:
    """Имена запрещённых статистик не должны всплыть как код или путь. Текст пояснения не проверяется."""
    import ast
    src = Path(aug_dg.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned_names = {"PHI_T_REAL", "PHI_X_REAL", "SIGMA_REAL_C", "featurize_noisy"}
    banned_bits = ("real_noise_psd", "1-6.csv", "real_zones", "aug_real")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in banned_names:
            raise SystemExit(f"aug_dg.py ссылается на {node.id}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[-1] in ("aug_real", "real"):
                    raise SystemExit(f"aug_dg.py импортирует {alias.name}")
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] in ("aug_real", "real"):
            raise SystemExit(f"aug_dg.py импортирует {node.module}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(bit in node.value for bit in banned_bits):
                raise SystemExit(f"aug_dg.py содержит путь к реальной статистике: {node.value[:80]}")


def proxy_iou(model, files, predict_batched) -> tuple[float, float]:
    """IoU@FAR 1e-3 на нескольких val. Порог только по их фону. Для выбора эпохи."""
    model.eval()
    maps = []
    parts = []
    for p in files:
        with np.load(p) as z:
            prob = predict_batched(model, np.asarray(z["X"], np.float32))
            valid = np.asarray(z["valid"], bool)
            ev = np.asarray(z["ev"])
            nu = np.asarray(z["nu"])
        s = 1.0 - prob[0]
        parts.append(s[valid & (ev == 0) & (nu == 0)])
        maps.append((s, valid, ev > 0))
        del prob
    bg = np.concatenate(parts)
    thr = M.far_threshold(bg, 1e-3)
    tot = np.zeros(3, np.int64)
    for s, valid, truth in maps:
        tot += np.asarray(M.confusion((s >= thr) & valid, truth, valid), np.int64)
    return float(M.prf(*tot)["iou"]), float(thr)


def main() -> None:
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    assert_no_real_stats()
    aug_dg.self_check()

    epochs = int(os.environ.get("DG_EPOCHS", "8"))
    steps = int(os.environ.get("DG_STEPS", "80"))
    batch = int(os.environ.get("DG_BATCH", "8"))
    lr = float(os.environ.get("DG_LR", "1e-4"))
    max_minutes = min(float(os.environ.get("DG_MAX_MINUTES", "40")), 45.0)
    seed = int(os.environ.get("DG_SEED", "0"))
    patience = int(os.environ.get("DG_PATIENCE", "3"))
    tile = (128, 256)
    init = Path("artifacts/stage1/unet_v810ft.pt")
    out_json = Path("artifacts/eval/s3_dg_train.json")
    out_pt = Path("artifacts/eval/s3_dg_unet.pt")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    log(
        f"приор phi_t~U(0,{aug_dg.PHI_T_HI}) phi_x~U(0,{aug_dg.PHI_X_HI}) "
        f"sigma={aug_dg.SIGMA_C}*U({aug_dg.SIGMA_LO},{aug_dg.SIGMA_HI}) p={aug_dg.P_APPLY}"
    )

    assert_mmap("v8", "train")
    assert_mmap("v8", "val")
    tr = data.mmap_items("artifacts/cache/v8", "train", Path("artifacts/mm/v8"))
    n_val = len(data.list_cache("artifacts/cache/v8", "val"))
    for it in tr:
        with np.load(it["src"]) as z:
            it["sigma"] = np.asarray(z["sigma"], np.float32)
    log(f"train {len(tr)} val {n_val} sigma записан только как шкала канала 0")

    aug_rng = np.random.default_rng(seed + 23)

    def _aug(xt, yt, it, t0, w0, rng=aug_rng):
        sig = it["sigma"]
        tw = xt.shape[-1]
        col = np.full(tw, 0.03, np.float32)
        n = min(tw, max(int(sig.shape[0]) - int(w0), 0))
        if n > 0:
            col[:n] = np.asarray(sig[w0:w0 + n], np.float32)
        return aug_dg.apply_tile(xt, yt, col, rng, p_apply=aug_dg.P_APPLY)

    sampler = data.TileSampler(tr, tile=tile, seed=seed, p_weak=0.0, aug=_aug)
    t_aug = time.time()
    sampler.sample(batch)
    log(f"aug batch {batch} за {time.time() - t_aug:.2f} с")

    model = models.build("unet", 16)
    model.load_state_dict(torch.load(init, map_location="cpu", weights_only=True))
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    cw = torch.tensor([1.0, 3.0, 3.0])

    def bn_eval():
        model.train()
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()

    predict_batched = _eval_mod().predict_batched
    err = _eval_mod().check_predict(model)
    log(f"predict max|Δ| {err:.3e}")
    proxy_files = data.list_cache("artifacts/cache/v8", "val")[:8]

    def save_ckpt():
        tmp = out_pt.with_suffix(".pt.tmp")
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, out_pt)

    def payload(hist, best_proxy, best_epoch, stopped, seconds):
        return dict(
            task="DG",
            init=str(init),
            init_why="unet_v810ft — тот же старт, что у заменяемого DOM; det1 другой чекпойнт и в смесь ens_blend не подставлялся",
            arch="unet",
            width=16,
            loss="focal+tversky_ce",
            lr=lr,
            weight_decay=0.0,
            epochs=epochs,
            steps=steps,
            batch=batch,
            tile=[128, 256],
            max_minutes=max_minutes,
            seed=seed,
            bn="frozen_running_stats_and_affine",
            prior=dict(
                phi_t="U(0, 0.34)",
                phi_x="U(0, 0.69)",
                sigma_c=aug_dg.SIGMA_C,
                sigma_factor="U(0.5, 2)",
                center_note="0.297 ровно, как в задании; в S3_LIT_DG.md медиана 0.2971",
                p_apply=aug_dg.P_APPLY,
                sign_p=0.0,
            ),
            freesdg=(
                "FFT только шума. Фаза первой AR(1)-реализации сохраняется. "
                "2–4 прямоугольника амплитуды берутся у второй независимой реализации того же синтетического априора. "
                "Остальные бины, кроме DC, умножаются на один U(0.5, 2). DC не масштабируется. "
                "После обратного БПФ СКО возвращается к sigma первой реализации: общий множитель этой нормировкой почти снимается, "
                "форму спектра меняют прямоугольники. Шум добавляется к T до make_channels. Радиусы (27, 9) не используются."
            ),
            ckpt_rule="max proxy IoU@FAR1e-3 on the first 8 sorted v8 val sequences; full 45-seq val and nested folds decide",
            train_cache="artifacts/cache/v8/train",
            real_traces_used=False,
            replaced_release=False,
            history=hist,
            best_proxy_iou=float(best_proxy),
            best_epoch=int(best_epoch),
            stopped=stopped,
            seconds=round(seconds, 1),
            ckpt=str(out_pt),
            predict_max_abs_err=err,
        )

    best_proxy, best_thr = proxy_iou(model, proxy_files, predict_batched)
    best_epoch = -1
    save_ckpt()
    t0 = time.time()
    hist = [dict(epoch=-1, train=None, proxy_iou=best_proxy, proxy_thr=best_thr, sec=0.0)]
    log(f"init proxy_iou {best_proxy:.4f} thr {best_thr:.6f}")
    out_json.write_text(json.dumps(payload(hist, best_proxy, best_epoch, "init", 0.0), indent=1, ensure_ascii=False))
    bad = 0
    stopped = "epochs"
    for ep in range(epochs):
        if (time.time() - t0) / 60.0 > max_minutes:
            stopped = "time"
            break
        bn_eval()
        tl = []
        cut = False
        for step in range(steps):
            if step % 10 == 0 and (time.time() - t0) / 60.0 > max_minutes:
                cut = True
                stopped = "time"
                break
            x, y = sampler.sample(batch)
            logit = model(torch.from_numpy(x))
            loss = losses.compute("focal+tversky_ce", logit, torch.from_numpy(y), cw)
            if not torch.isfinite(loss):
                raise SystemExit(f"loss не конечен на эпохе {ep} шаге {step}")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(float(loss.item()))
        if not tl:
            break
        pi, pt = proxy_iou(model, proxy_files, predict_batched)
        rec = dict(
            epoch=ep,
            train=float(np.mean(tl)),
            steps_done=len(tl),
            proxy_iou=pi,
            proxy_thr=pt,
            sec=round(time.time() - t0, 1),
        )
        hist.append(rec)
        log(str(rec))
        if pi > best_proxy + 1e-4:
            best_proxy = pi
            best_epoch = ep
            bad = 0
            save_ckpt()
        else:
            bad += 1
            if bad >= patience and ep >= 2:
                stopped = "patience"
                out_json.write_text(json.dumps(payload(hist, best_proxy, best_epoch, stopped, time.time() - t0), indent=1, ensure_ascii=False))
                break
        if cut:
            stopped = "time"
            out_json.write_text(json.dumps(payload(hist, best_proxy, best_epoch, stopped, time.time() - t0), indent=1, ensure_ascii=False))
            break
        out_json.write_text(json.dumps(payload(hist, best_proxy, best_epoch, stopped, time.time() - t0), indent=1, ensure_ascii=False))

    # вернуть веса лучшей эпохи: последние шаги могли уйти ниже proxy
    model.load_state_dict(torch.load(out_pt, map_location="cpu", weights_only=True))
    final = payload(hist, best_proxy, best_epoch, stopped, time.time() - t0)
    out_json.write_text(json.dumps(final, indent=1, ensure_ascii=False))
    log(f"готово best_proxy {best_proxy:.4f} epoch {best_epoch} stop {stopped} sec {final['seconds']}")


if __name__ == "__main__":
    main()
