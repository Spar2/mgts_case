"""Bench PDRS vs threshold+CC on ensemble probability maps (synthetic, labels from generator)."""
import sys, json, time, glob, itertools, os
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dts.pdrs import extract_zones_pdrs, extract_zones_cc

HOT, COLD = (1, 2, 3, 4), (5, 6)


def score_seq(lab, zones, d):
    ev, valid = d["ev"], d["valid"]
    hot_gt, cold_gt = np.isin(ev, HOT) & valid, np.isin(ev, COLD) & valid
    pol = {z["zone"]: z["polarity"] for z in zones}
    lut = np.zeros(lab.max() + 1, np.int8)
    for k, p in pol.items(): lut[k] = 1 if p == "hot" else 2
    pm = lut[lab]
    r = {}
    for name, g, p in (("hot", hot_gt, pm == 1), ("cold", cold_gt, pm == 2), ("all", hot_gt | cold_gt, pm > 0)):
        r[name] = (int((g & p).sum()), int((~g & p & valid).sum()), int((g & ~p).sum()))
    gtl, ng = ndi.label(hot_gt | cold_gt, structure=np.ones((3, 3)))
    hit = np.unique(gtl[lab > 0]); det = len(set(hit) - {0})
    zt = sum(1 for z in zones if (gtl[z["t0"]:z["t1"], z["x0"]:z["x1"]][lab[z["t0"]:z["t1"], z["x0"]:z["x1"]] == z["zone"]] > 0).any())
    # boundary errors for matched zones (bbox vs best gt bbox)
    be = []
    objs = ndi.find_objects(gtl)
    for z in zones:
        sub = gtl[z["t0"]:z["t1"], z["x0"]:z["x1"]][lab[z["t0"]:z["t1"], z["x0"]:z["x1"]] == z["zone"]]
        sub = sub[sub > 0]
        if len(sub):
            g = objs[np.bincount(sub).argmax() - 1]
            be.append(((abs(g[0].start - z["t0"]) + abs(g[0].stop - z["t1"])) / 2 * 120,
                      (abs(g[1].start - z["x0"]) + abs(g[1].stop - z["x1"])) / 2 * 0.25))
    days = (d["t"][-1] - d["t"][0]) / 86400
    return r, ng, det, len(zones), zt, be, days


def run(method, params, files, pred_dir):
    acc = {k: np.zeros(3) for k in ("hot", "cold", "all")}; NG = DET = NZ = ZT = 0; BE = []; D = 0; T = 0
    for f in files:
        d = np.load(f); p = np.load(os.path.join(pred_dir, os.path.basename(f)))
        t0 = time.time()
        if method == "cc":
            lab, zones = extract_zones_cc(p["s"], d["valid"], p["h"], **params)
        else:
            lab, zones = extract_zones_pdrs(p["s"], d["valid"], p["h"], d["t"], d["x"], **params)
        T += time.time() - t0
        r, ng, det, nz, zt, be, days = score_seq(lab, zones, d)
        for k in acc: acc[k] += r[k]
        NG += ng; DET += det; NZ += nz; ZT += zt; BE += be; D += days
    out = {"method": method, "params": params}
    for k, (tp, fp, fn) in acc.items(): out[f"iou_{k}"] = round(tp / max(tp + fp + fn, 1), 4)
    out.update(event_recall=round(DET / max(NG, 1), 4), zone_precision=round(ZT / max(NZ, 1), 4),
               zones_per_day=round(NZ / max(D, 1e-9), 2), bnd_err_t_s=round(float(np.median([b[0] for b in BE])) if BE else -1, 1),
               bnd_err_x_m=round(float(np.median([b[1] for b in BE])) if BE else -1, 2), sec_per_seq=round(T / len(files), 3))
    return out


if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "v8"
    model = "ens_u2ft_tcn_lgbm"
    res = {}
    grid = [dict(seed=se, floor=fl, r_saddle=rs, region_median=rm)
            for se, fl, rs, rm in itertools.product([0.86, 0.95], [0.3, 0.5, 0.7], [0.3, 0.6, 0.9], [0.5, 0.7])]
    if ver == "v8":  # grid selection on VAL only
        vf = sorted(glob.glob("artifacts/cache/v8/val/*.npz"))
        pv = f"artifacts/pred/v8/{model}/val"
        rows = [run("pdrs", g, vf, pv) for g in grid]
        rows.append(run("cc", dict(thr=0.8608087301264272), vf, pv))
        best = max([r for r in rows if r["method"] == "pdrs"], key=lambda r: r["iou_all"])
        res["val_grid"] = rows; res["best_params"] = best["params"]
        json.dump(res["best_params"], open("artifacts/eval/pdrs_best_params.json", "w"))
    bp = json.load(open("artifacts/eval/pdrs_best_params.json"))
    tf = sorted(glob.glob(f"artifacts/cache/{ver}/test/*.npz"))
    pt = f"artifacts/pred/{ver}/{model}/test"
    tf = [f for f in tf if os.path.exists(os.path.join(pt, os.path.basename(f)))]
    res["test"] = [run("cc", dict(thr=0.8608087301264272), tf, pt), run("pdrs", bp, tf, pt)]
    json.dump(res, open(f"artifacts/eval/pdrs_{ver}.json", "w"), indent=1)
    for r in res["test"]: print(ver, r)
