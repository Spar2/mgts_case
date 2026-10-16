"""Синтетика -> каноническая сетка + каналы CNN + скор базы. Кэш: artifacts/cache/<version>/<split>/<seq>.npz"""
import argparse, sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import grid, features, synth, baseline


def one(args):
    item, out = args
    dst = Path(out) / item["split"] / f"{item['seq_id']}.npz"
    if dst.exists():
        return str(dst)
    d = synth.read_seq(item["path"])
    p = grid.plan(d["x"], d["t"])
    T, v = grid.resample(d["noisy"], d["valid"], p, "mean")
    ev, _ = grid.resample(d["mask_event"], np.ones_like(d["valid"]), p, "nearest")
    nu, _ = grid.resample(d["mask_nuisance"], np.ones_like(d["valid"]), p, "nearest")
    ch, aux = features.make_channels(T, v)
    b = baseline.run(d["noisy"], d["x"], d["t"], valid=d["valid"])
    bz, _ = grid.resample(b["z"], b["available"], p, "nearest")
    bzone, _ = grid.resample(b["zone"], np.ones_like(d["valid"]), p, "nearest")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, X=ch.astype(np.float16), T=np.nan_to_num(T).astype(np.float32), valid=v, ev=ev.astype(np.uint8), nu=nu.astype(np.uint8),
             base_z=np.nan_to_num(bz).astype(np.float16), base_zone=bzone.astype(np.uint8), x=p["x"], t=p["t"],
             sigma=aux["sigma_x"])
    return str(dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--synth-root", required=True); ap.add_argument("--out", default="artifacts/cache")
    ap.add_argument("--workers", type=int, default=5); ap.add_argument("--splits", default="train,val,test")
    a = ap.parse_args()
    info = synth.discover(a.synth_root)
    out = Path(a.out) / info["version"]
    items = [s for s in info["seqs"] if s["split"] in a.splits.split(",")]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, [(s, str(out)) for s in items])):
            if i % 20 == 0: print(i, len(items), round(time.time() - t0), r, flush=True)
    print("DONE", info["version"], len(items), round(time.time() - t0))
