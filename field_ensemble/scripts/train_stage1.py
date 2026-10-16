"""Обучение CNN-сегментатора (stage 1) на train-сплите синтетики; ранняя остановка по val."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import data, models, losses, noise_aug

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="artifacts/cache/v8"); ap.add_argument("--out", default="artifacts/stage1")
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--batch", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--width", type=int, default=None); ap.add_argument("--arch", default="unet"); ap.add_argument("--threads", type=int, default=7)
ap.add_argument("--max-minutes", type=float, default=80)
ap.add_argument("--weak-weight", type=float, default=0.0, help="доп. вес пикселей слабых событий (≤2 °C)")
ap.add_argument("--p-weak", type=float, default=0.0); ap.add_argument("--tile", default="128,256")
ap.add_argument("--labels", default="data/synth_meta/v8/labels.csv"); ap.add_argument("--init", default=None); ap.add_argument("--tag", default="")
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--loss", default=None, help="ce, focal, tversky_ce или сумма через '+'; без флага — прежний CE+dice")
ap.add_argument("--sets", default=None, help="версии кэша через запятую (v8,v9); без флага — один --cache")
ap.add_argument("--noise-std", type=float, default=0.0, help="std AR(1) на каналах 0 и 1 нормированного тайла; 0 — без добавки")
ap.add_argument("--noise-phi-t", type=float, default=0.0)
ap.add_argument("--noise-phi-x", type=float, default=0.0)
ap.add_argument("--temp-noise", type=float, default=0.0, help="СКО цветного шума в °C до make_channels; 0 выкл; <0 — σ real из EXPERIMENTS_LATE")
ap.add_argument("--temp-phi-t", type=float, default=0.357)
ap.add_argument("--temp-phi-x", type=float, default=0.736)
ap.add_argument("--temp-jitter", type=float, default=0.25, help="относительный разброс амплитуды вокруг σ")
ap.add_argument("--sign-flip", type=float, default=0.0, help="вероятность T→−T вокруг фона и hot↔cold")
a = ap.parse_args()
if a.loss:
    losses.parse_spec(a.loss)
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

def items_one(cache, split):
    return data.mmap_items(cache, split, Path("artifacts/mm") / Path(cache).name)

if a.sets:
    tr, va = [], []
    for ver in [v.strip() for v in a.sets.split(",") if v.strip()]:
        cache, mm = Path("artifacts/cache") / ver, Path("artifacts/mm") / ver
        tr_v, va_v = data.mmap_items(cache, "train", mm), data.mmap_items(cache, "val", mm)
        if not tr_v or not va_v:
            raise SystemExit(f"{ver}: train {len(tr_v)} val {len(va_v)} — нет mmap-кэша artifacts/cache/{ver}/{{train,val}}")
        if a.weak_weight > 0 or a.p_weak > 0:
            import pandas as pd
            data.add_weak_weights(tr_v, pd.read_csv(Path("data/synth_meta") / ver / "labels.csv"), mm, "train")
        print("set", ver, "train", len(tr_v), "val", len(va_v), flush=True)
        tr.extend(tr_v); va.extend(va_v)
else:
    tr, va = items_one(a.cache, "train"), items_one(a.cache, "val")
    if a.weak_weight > 0 or a.p_weak > 0:
        import pandas as pd
        labels = pd.read_csv(a.labels); mm = Path("artifacts/mm") / Path(a.cache).name
        data.add_weak_weights(tr, labels, mm, "train")
print("train", len(tr), "val", len(va), flush=True)
tile = tuple(int(v) for v in a.tile.split(","))
S = data.TileSampler(tr, tile=tile, seed=a.seed, p_weak=a.p_weak); Sv = data.TileSampler(va, seed=1234)
if a.temp_noise != 0 or a.sign_flip > 0:
    from dts import augment
    if a.temp_noise < 0:
        rec = augment.lit_temperature_recipe(Path("docs/EXPERIMENTS_LATE.md").read_text(encoding="utf-8")) or {}
        sigma_c = float(rec.get("sigma_c", augment.SIGMA_REAL_C))
        phi_t = float(rec.get("phi_t", a.temp_phi_t))
        phi_x = float(rec.get("phi_x", a.temp_phi_x))
    else:
        sigma_c, phi_t, phi_x = a.temp_noise, a.temp_phi_t, a.temp_phi_x
    spec = augment.Spec(sigma_c, phi_t, phi_x, a.temp_jitter, a.sign_flip)
    aug_rng = np.random.default_rng(a.seed + 23)
    n_sig = 0
    for it in tr:
        if "sigma" not in it:
            with np.load(it["src"]) as z:
                it["sigma"] = np.array(z["sigma"], dtype=np.float32)
            n_sig += 1
    def _aug(xt, yt, it, t0, w0, spec=spec, rng=aug_rng):
        sig = it["sigma"]
        tw = xt.shape[-1]
        col = np.full(tw, 0.03, np.float32)
        n = min(tw, max(int(sig.shape[0]) - int(w0), 0))
        if n > 0:
            col[:n] = sig[w0:w0 + n]
        return augment.apply_tile(xt, yt, col, rng, spec)
    S.aug = _aug
    print("temp-aug", spec.as_dict(), "sigma_loaded", n_sig, flush=True)
vx, vy = Sv.sample(96)
model = models.build(a.arch, a.width)
if a.init: model.load_state_dict(torch.load(a.init))
opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * a.steps)
cw = torch.tensor([1.0, 3.0, 3.0])

def loss_fn(logit, y, wk=None):
    if wk is not None and a.weak_weight > 0:
        pix = F.cross_entropy(logit, y, weight=cw, ignore_index=-1, reduction="none")
        w = (1 + a.weak_weight * wk) * (y >= 0)
        ce = (pix * w).sum() / w.sum().clamp(min=1)
    else:
        ce = F.cross_entropy(logit, y, weight=cw, ignore_index=-1)
    p = torch.softmax(logit, 1); m = (y >= 0).float()
    fg = 1 - p[:, 0]; t = (y > 0).float()
    dice = 1 - (2 * (fg * t * m).sum() + 1) / ((fg * m).sum() + (t * m).sum() + 1)
    return ce + dice

noise_rng = np.random.default_rng(a.seed + 17)

def criterion(logit, y, wk=None):
    if not a.loss:
        return loss_fn(logit, y, wk)
    return losses.compute(a.loss, logit, y, cw, wk, a.weak_weight)

best, hist, t0 = 1e9, [], time.time()
for ep in range(a.epochs):
    model.train(); tl = []
    for s in range(a.steps):
        x, y = S.sample(a.batch)
        if a.noise_std > 0:
            x = noise_aug.add_channel_ar1(x, a.noise_std, a.noise_phi_t, a.noise_phi_x, noise_rng)
        loss = criterion(model(torch.from_numpy(x)), torch.from_numpy(y), torch.from_numpy(S.last_weak))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl.append(loss.item())
    model.eval()
    with torch.no_grad():
        vl = np.mean([criterion(model(torch.from_numpy(vx[i:i + 16])), torch.from_numpy(vy[i:i + 16])).item() for i in range(0, len(vx), 16)])
    hist.append(dict(epoch=ep, train=float(np.mean(tl)), val=float(vl), sec=round(time.time() - t0)))
    print(hist[-1], flush=True)
    if vl < best:
        best = vl; torch.save(model.state_dict(), out / f"{a.arch}{a.tag}.pt")
    if (time.time() - t0) / 60 > a.max_minutes:
        print("time budget reached"); break
json.dump(dict(args=vars(a), history=hist, best_val=float(best)), open(out / f"train_log_{a.arch}{a.tag}.json", "w"), indent=1)
