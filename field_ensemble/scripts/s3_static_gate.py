"""Обучаемый гейт слияния и вторая сеть StaticNet.

Гейт — маленькая 1D-сеть на [сводка s победителя, две карты ветви, локальная
форма горба]. Веса гейта считаются только по кропам synth train (v8 и v10).
Val и test в градиент не входят. Вторая сеть — тот же StaticNet с другим сидом,
оба ствола заморожены.

Слияние max(s, g * p) карту победителя не опускает. Выбор — по v8 val:
максимум static recall на synth_static_val среди карт с dval >= -0.003,
при равенстве recall берётся больший val IoU. Test в выборе не участвует.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts.domain import ensemble_maps
from dts.infer import Analyzer
from dts.static_net import (
    GATE_CIN, StaticGate, StaticNet, fuse, fuse_gated, gate_channels, gate_loss, predict_gate,
    residual_from_cache, temporal_features,
)
from dts.static_synth import DX, apply_profiles, sample_train_specs

LOG = Path("/workspace/run_logs/_events.log")
WINNER_PUB_VAL = 0.386712
DVAL_MIN = -0.003
CROP_W = 768
CROP_T = 256


def log(msg: str):
    line = time.strftime("%H:%M") + " " + msg
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def load_eval():
    path = ROOT / "scripts" / "s3_static_eval.py"
    spec = importlib.util.spec_from_file_location("s3_static_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_static(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = StaticNet(
        cin=int(ckpt.get("cin", 9)), width=int(ckpt.get("width", 16)), pool=int(ckpt.get("pool", 80)),
        max_rows=int(ckpt.get("max_rows", 64)), n_bins=int(ckpt.get("n_bins", 0)),
    )
    net.load_state_dict(ckpt["state"])
    net.eval()
    for par in net.parameters():
        par.requires_grad_(False)
    return net


def event_columns(ev: np.ndarray) -> np.ndarray:
    return (ev > 0).mean(axis=0) > 0.02


def column_pack(nets, R, valid, s):
    feats, good = temporal_features(
        R, valid, max_rows=int(nets[0].max_rows), n_bins=int(nets[0].n_bins))
    x = torch.from_numpy(np.ascontiguousarray(feats[None]))
    probs = []
    with torch.no_grad():
        for net in nets:
            p = torch.sigmoid(net(x))[0].detach().numpy().astype(np.float32)
            p = p.copy()
            p[~good] = 0.0
            probs.append(p)
    ch, mean = gate_channels(s, valid, probs[0], probs[1], feats)
    return feats, probs[0], probs[1], ch, mean, good


def targets(pos, hard, avoid, good, feats):
    y = np.asarray(pos, np.float32)
    w = np.ones(y.shape, np.float32)
    w[~good] = 0.0
    w[avoid & (y < 0.5)] = 0.0
    if hard is not None:
        w[np.asarray(hard, bool)] = 4.0
    peak = good & (y < 0.5) & (feats[0] > 0.25)
    w[peak] = np.maximum(w[peak], 3.0)
    w[y > 0.5] = 1.0
    return y, w


def build_crops(nets, analyzer, n_files: int, seed: int):
    rng = np.random.default_rng(seed)
    paths = sorted((ROOT / "artifacts/cache/v8/train").glob("seq*.npz"))
    paths += sorted((ROOT / "artifacts/cache/v10/train").glob("seq*.npz"))
    order = rng.permutation(len(paths))
    items = []
    t0 = time.time()
    for k in order:
        if len(items) >= n_files * 2:
            break
        path = paths[int(k)]
        z = np.load(path)
        R, valid = residual_from_cache(z)
        X = np.asarray(z["X"], np.float32)
        ev = np.asarray(z["ev"])
        rows, width = R.shape
        if rows < 32 or width < 128:
            continue
        t0i, t1i = 0, rows
        if rows > CROP_T:
            t0i = int(rng.integers(0, rows - CROP_T + 1))
            t1i = t0i + CROP_T
        w0, w1 = 0, width
        if width > CROP_W:
            w0 = int(rng.integers(0, width - CROP_W + 1))
            w1 = w0 + CROP_W
        sl_t = slice(t0i, t1i)
        sl_x = slice(w0, w1)
        Rc, vc, evc = R[sl_t, sl_x], valid[sl_t, sl_x], ev[sl_t, sl_x]
        Xc = np.ascontiguousarray(X[:, sl_t, sl_x])
        s, _ = ensemble_maps(analyzer, Xc)
        avoid = event_columns(evc)
        feats, p1, p2, ch, mean, good = column_pack(nets, Rc, vc, s)
        y, w = targets(np.zeros(Rc.shape[1], bool), None, avoid, good, feats)
        items.append(dict(x=ch, y=y, w=w, p=mean, name=path.stem, view="clean"))
        specs = sample_train_specs(rng, Rc.shape[1], DX, avoid)
        Rp, pos, hard = apply_profiles(Rc, specs)
        feats_p, p1p, p2p, chp, meanp, goodp = column_pack(nets, Rp, vc, s)
        yp, wp = targets(pos, hard, avoid, goodp, feats_p)
        items.append(dict(x=chp, y=yp, w=wp, p=meanp, name=path.stem, view="paste"))
        if len(items) % 8 == 0:
            print(f"crops {len(items)} {time.time() - t0:.0f}s", flush=True)
    if len(items) < 8:
        raise SystemExit(f"мало кропов для гейта: {len(items)}")
    return items


def _pad(a, w):
    if a.shape[-1] == w:
        return a
    out = np.zeros(a.shape[:-1] + (w,), a.dtype)
    out[..., :a.shape[-1]] = a
    return out


def run_gate_epoch_proxy(gate, items):
    hits, n_pos, fired, n_neg = 0.0, 0, 0.0, 0
    for it in items:
        g = predict_gate(gate, it["x"])
        opened = g * it["p"]
        pos = it["y"] > 0.5
        neg = (it["y"] < 0.5) & (it["w"] > 0) & (it["p"] > 0.40)
        if pos.any():
            # попадание столбца: g*p не ниже 0.90 там, где ветвь сама уже высока
            strong = pos & (it["p"] >= 0.85)
            if strong.any():
                hits += float((opened[strong] >= 0.90).mean())
                n_pos += 1
        if neg.any():
            fired += float((opened[neg] >= 0.80).mean())
            n_neg += 1
    return dict(pos_open=hits / max(n_pos, 1), n_pos=n_pos,
                bg_hi=fired / max(n_neg, 1), n_neg=n_neg)


def train_gate(items, steps: int, seed: int, lr: float):
    rng = np.random.default_rng(seed)
    hold = items[-8:]
    pool = items[:-8] or items
    gate = StaticGate(GATE_CIN)
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state = None, None
    history = []
    for step in range(1, steps + 1):
        frac = step / steps
        lr_now = lr * (0.2 + 0.8 * 0.5 * (1 + np.cos(np.pi * frac)))
        for gopt in opt.param_groups:
            gopt["lr"] = float(lr_now)
        pick = [pool[int(rng.integers(0, len(pool)))] for _ in range(8)]
        # отражение по x — та же физика, другие координаты
        xs, ys, ws, ps = [], [], [], []
        for it in pick:
            x, y, w, p = it["x"], it["y"], it["w"], it["p"]
            if rng.random() < 0.5:
                x, y, w, p = x[:, ::-1], y[::-1], w[::-1], p[::-1]
            xs.append(np.ascontiguousarray(x))
            ys.append(np.ascontiguousarray(y))
            ws.append(np.ascontiguousarray(w))
            ps.append(np.ascontiguousarray(p))
        wmax = max(a.shape[-1] for a in xs)
        xb = torch.from_numpy(np.stack([_pad(a, wmax) for a in xs]))
        yb = torch.from_numpy(np.stack([_pad(a, wmax) for a in ys]))
        wb = torch.from_numpy(np.stack([_pad(a, wmax) for a in ws]))
        pb = torch.from_numpy(np.stack([_pad(a, wmax) for a in ps]))
        opt.zero_grad(set_to_none=True)
        g = gate(xb)
        loss, parts = gate_loss(g, yb, wb, pb)
        if not torch.isfinite(loss):
            log(f"STATIC3_GATE nonfinite step={step}")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 2.0)
        opt.step()
        if step % 50 == 0 or step == steps:
            gate.eval()
            pr = run_gate_epoch_proxy(gate, hold)
            gate.train()
            score = pr["pos_open"] - 8.0 * pr["bg_hi"]
            rec = dict(step=step, loss=float(loss.detach()), **pr, score=score)
            history.append(rec)
            log(f"STATIC3_GATE step={step} loss={rec['loss']:.4f} pos_open={pr['pos_open']:.3f} "
                f"bg_hi={pr['bg_hi']:.4f} n_pos={pr['n_pos']}")
            if best is None or score >= best["score"]:
                best = rec
                best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
    if best_state is not None:
        gate.load_state_dict(best_state)
    gate.eval()
    return gate, best, history


def variant_scores(s, p1, p2, g):
    mean = (0.5 * (p1 + p2)).astype(np.float32)
    both = np.maximum(p1, p2)
    agree = np.minimum(p1, p2)
    return {
        "winner": s,
        "max_s1": fuse(s, p1, "max"),
        "max_s2": fuse(s, p2, "max"),
        "max_mean": fuse(s, mean, "max"),
        "max_min": fuse(s, agree, "max"),
        "gate_mean": fuse_gated(s, mean, g),
        "gate_max": fuse_gated(s, both, g),
    }


def _split_pass(nets, gate, E, split: str, seed_base: int, thr: dict | None, save_specs: bool):
    """Один проход по split. thr is None — только гистограмма фона val, без хранения карт."""
    names = ("winner", "max_s1", "max_s2", "max_mean", "max_min", "gate_mean", "gate_max")
    hists = {n: np.zeros(E.NBIN, np.int64) for n in names}
    counts = {n: dict(tot=np.zeros(3, np.int64), far_n=0, far_d=0) for n in names}
    events = {n: [] for n in names}
    spec_dump = []
    n = 0
    for cache_p, pred_p in E.pairs(split):
        d = E.load_seq(cache_p, pred_p)
        _f, p1, p2, ch, _mean, _good = column_pack(nets, d["R"], d["valid"], d["s"])
        g = predict_gate(gate, ch)
        clean = variant_scores(d["s"], p1, p2, g)
        bg = d["valid"] & (d["ev"] == 0) & (d["nu"] == 0)
        if thr is None:
            for name in names:
                E.add_score_hist(hists[name], clean[name][bg])
            n += 1
            del d
            continue
        specs = E.static_specs_for(d, seed_base)
        if save_specs:
            spec_dump.append(dict(seq=d["name"], specs=specs))
        Rp, _, _ = apply_profiles(d["R"], specs)
        _fp, p1p, p2p, chp, _mp, _gp = column_pack(nets, Rp, d["valid"], d["s"])
        gp = predict_gate(gate, chp)
        pasted = variant_scores(d["s"], p1p, p2p, gp)
        truth = d["ev"] > 0
        for name in names:
            pred = (clean[name] >= thr[name]) & d["valid"]
            counts[name]["tot"] += E.M.confusion(pred, truth, d["valid"])
            counts[name]["far_n"] += int((pred & bg).sum())
            counts[name]["far_d"] += int(bg.sum())
            events[name].extend(E.event_rows(specs, pasted[name], d["valid"], thr[name]))
        n += 1
        if n % 10 == 0:
            print(f"  {split} {n}", flush=True)
        del d, Rp
    return hists, counts, events, spec_dump


def evaluate(nets, gate, E):
    names = ("winner", "max_s1", "max_s2", "max_mean", "max_min", "gate_mean", "gate_max")
    hists, _, _, _ = _split_pass(nets, gate, E, "val", 20261015, None, False)
    thr = {name: E.threshold_from_hist(hists[name], E.FAR) for name in names}
    for name in names:
        print(f"thr {name} {thr[name]:.6f}", flush=True)
    _, counts, events, specs = _split_pass(nets, gate, E, "val", 20261015, thr, True)
    val_rows = {}
    for name in names:
        row = E.pack_counts(counts[name], thr[name])
        row["static"] = E.summarize_events(events[name])
        val_rows[name] = row
        st = row["static"]
        print(name, "val", "iou", round(row["iou"], 4), "static",
              None if st.get("recall") is None else round(st["recall"], 4), flush=True)
    _, counts_te, events_te, _ = _split_pass(nets, gate, E, "test", 20261015 + 17, thr, False)
    test_rows = {}
    for name in names:
        row = E.pack_counts(counts_te[name], thr[name])
        row["static"] = E.summarize_events(events_te[name])
        test_rows[name] = row
        st = row["static"]
        print(name, "test", "iou", round(row["iou"], 4), "static",
              None if st.get("recall") is None else round(st["recall"], 4), flush=True)
    return dict(val=dict(rows=val_rows, thresholds=thr, specs=specs),
                test=dict(rows=test_rows, thresholds=thr, specs=None))


def choose(val_rows):
    winner_iou = val_rows["winner"]["iou"]
    ranked = [(name, row) for name, row in val_rows.items() if name != "winner"]
    ok = [r for r in ranked if r[1]["iou"] >= winner_iou + DVAL_MIN - 1e-12]

    def key(r):
        return (r[1]["static"]["recall"] or 0.0, r[1]["iou"])

    pool = ok or ranked
    pool.sort(key=key, reverse=True)
    return pool[0][0], pool[0][1], bool(ok), winner_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-a", default="artifacts/static/static_net2_s800.pt")
    ap.add_argument("--net-b", default="artifacts/static/static_net3_s2.pt")
    ap.add_argument("--out-gate", default="artifacts/static/static_gate.pt")
    ap.add_argument("--out", default="artifacts/static/s3_static3.json")
    ap.add_argument("--train-n", type=int, default=48)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()
    torch.set_num_threads(3)
    t0 = time.time()
    E = load_eval()
    net_a = load_static(ROOT / a.net_a)
    net_b = load_static(ROOT / a.net_b)
    if int(net_a.cin) != int(net_b.cin) or int(net_a.n_bins) != int(net_b.n_bins):
        raise SystemExit("сети разной ширины входа, гейт их не смешивает")
    nets = (net_a, net_b)
    log(f"STATIC3_START net_a={a.net_a} net_b={a.net_b} train_n={a.train_n} steps={a.steps}")
    analyzer = Analyzer()
    items = build_crops(nets, analyzer, a.train_n, a.seed)
    del analyzer
    log(f"STATIC3_CROPS n={len(items)} sec={time.time() - t0:.0f}")
    gate, best, history = train_gate(items, a.steps, a.seed, a.lr)
    out_gate = ROOT / a.out_gate
    out_gate.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=gate.state_dict(), cin=GATE_CIN, step=None if best is None else best["step"],
                    proxy=best, seed=a.seed), out_gate)
    log("STATIC3_EVAL val+test")
    reports = evaluate(nets, gate, E)
    name, row, ok, winner_iou = choose(reports["val"]["rows"])
    te = reports["test"]["rows"][name]
    beats = bool(row["iou"] > WINNER_PUB_VAL + 1e-4)
    art = ROOT / "artifacts/static"
    if reports["val"]["specs"] is not None:
        (art / "synth_static3_val.json").write_text(json.dumps(reports["val"]["specs"]))
    out = dict(
        name="STATIC3",
        far=1e-3,
        dval_min=DVAL_MIN,
        net_a=a.net_a,
        net_b=a.net_b,
        gate=a.out_gate,
        gate_step=None if best is None else best.get("step"),
        train_crops=len(items),
        train_only=True,
        selected=name,
        constraint_met=ok,
        beats_winner=beats,
        published_winner=dict(val_iou=WINNER_PUB_VAL, test_iou=0.405276),
        val={k: v for k, v in reports["val"]["rows"].items()},
        test={k: v for k, v in reports["test"]["rows"].items()},
        gate_history=[{k: v for k, v in h.items() if k != "score"} for h in history],
        seconds=round(time.time() - t0, 1),
        note="Гейт обучен на кропах v8/v10 train. Порог — квантиль фона v8 val без вклейки. "
             "Выбор — max static recall при dval >= -0.003, затем val IoU. Test не выбирает.",
    )
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    wte = reports["test"]["rows"]["winner"]
    log(f"STATIC3_DONE selected={name} val_iou={row['iou']:.6f} test_iou={te['iou']:.6f} "
        f"dval={row['iou'] - winner_iou:+.6f} static_val={row['static']['recall']} "
        f"static_test={te['static']['recall']} thr={row['threshold']:.6f} "
        f"winner_val={winner_iou:.6f} winner_test={wte['iou']:.6f} "
        f"winner_static_val={reports['val']['rows']['winner']['static']['recall']} "
        f"winner_static_test={wte['static']['recall']} "
        f"max_s1_static_val={reports['val']['rows']['max_s1']['static']['recall']} "
        f"gate_step={None if best is None else best.get('step')} "
        f"constraint={'ok' if ok else 'best_effort'} beats={int(beats)}")


if __name__ == "__main__":
    main()
