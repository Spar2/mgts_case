"""Обзор карточек оператора №2 (только оценка, ничего не обучается и пороги не двигаются)."""
import json, sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, "."); from dts import real, baseline
A = "/home/box/agent-data/agents/7620c11d-77b1-4929-aaa8-c80e192177e7/attachments/"
new = [json.loads(l) for l in open(A + "66eacdf5a4b0ab8649ecf57a4b35554ad67790401bcf97c035e546eff4f764ff.jsonl") if l.strip()]
old = [json.loads(l) for f in ["68e46832ee8ed2e697aa47e5593c1b48b577679a566ed48d7b529169d0e80a83.jsonl", "185b92fa42f2df65ec13d44eff7f42052367fdc85bda095b7e1484b062e5ae48.jsonl"] for l in open(A + f) if l.strip()]
oldk = {(o["record"], tuple(o["meters"]), tuple(o["time"])) for o in old}
cards = []; seen = set()
for c in new:
    k = (c["record"], tuple(c["meters"]), tuple(c["time"]))
    c["dup_old"] = k in oldk; c["dup_new"] = k in seen; seen.add(k); cards.append(c)
E0 = pd.Timestamp("1970-01-01", tz="UTC")
ts = lambda s: (pd.Timestamp(s).tz_convert("UTC") - E0).total_seconds()
M = {"winner": "artifacts/eval/real_zones.csv", "ens_dom": "artifacts/eval/real_dom.csv",
     "ens_cnsym": "artifacts/release_tmp_ens_cnsym/zones.csv", "pdrs_hybrid": "artifacts/eval/real_pdrs_hybrid.csv"}
Z = {}
for m, f in M.items():
    d = pd.read_csv(f); d["ta"] = d.t_start_utc.map(ts); d["tb"] = d.t_end_utc.map(ts); d["rec"] = d.source.str.replace(".csv", "", regex=False); Z[m] = d
rows = []
for rec in sorted({c["record"] for c in cards}):
    r = real.load_csv(f"/workspace/mgts/MGTS_data/{rec}.csv"); T = r["T"].astype(float); x = r["x"]; t = r["t"]
    T = np.where(r["valid"], T, np.nan)
    b = baseline.run(r["T"], x, t, valid=r["valid"])
    # ΔT: вычитаем статический профиль (медиана по времени) и синфазную составляющую (медиана по x)
    d = T - np.nanmedian(T, 0, keepdims=True); d = d - np.nanmedian(d, 1, keepdims=True)
    for c in [c for c in cards if c["record"] == rec]:
        xa, xb = c["meters"]; ta, tb = map(ts, c["time"])
        ix = (x >= xa) & (x <= xb); it = (t >= ta) & (t <= tb)
        out = dict(id=c["id"], rec=rec, x0=xa, x1=xb, hours=round((tb - ta) / 3600, 2), user=c["presence"], shape=c["shape"], notes=c["notes"], dup=c["dup_old"] or c["dup_new"])
        for m, dd in Z.items():
            o = dd[(dd.rec == rec) & (dd.x_end_m >= xa) & (dd.x_start_m <= xb) & (dd.tb >= ta) & (dd.ta <= tb) & (dd.is_alarm == 1)]
            out[m] = int(len(o) > 0); out[m + "_cls"] = ",".join(sorted(set(o["class"]))); out[m + "_n"] = len(o)
        bo = [z for z in b["zones"] if z["end_m"] >= xa and z["start_m"] <= xb and z["t_end_s"] >= ta and z["t_start_s"] <= tb]
        out["baseline"] = int(len(bo) > 0); out["baseline_n"] = len(bo)
        w = d[np.ix_(it, ix)]
        if w.size and np.isfinite(w).any():
            noise = float(np.nanmedian(np.abs(np.diff(w, axis=0))) / 0.954)  # σ по разностям соседних кадров
            hi, lo = float(np.nanpercentile(w, 99.5)), float(np.nanpercentile(w, 0.5))
            colmax = np.nanmax(np.abs(w), 0); ext = float((colmax > max(4 * noise, 0.5)).sum() * np.nanmedian(np.diff(x)))
            out.update(noise=round(noise, 2), dT_hi=round(hi, 2), dT_lo=round(lo, 2), snr=round(max(hi, -lo) / max(noise, 1e-3), 1), extent_m=round(ext, 1),
                       com_range=round((lambda q: float(np.nanmax(q)-np.nanmin(q)))(np.nanmedian(T[np.ix_(it, ix)], 1)), 2))
        rows.append(out)
        # рисунок
        pad = max(20, (xb - xa) * .3); jx = (x >= xa - pad) & (x <= xb + pad); jt = (t >= ta - 3600) & (t <= tb + 3600)
        if jx.any() and jt.any():
            fig, ax = plt.subplots(figsize=(7, 4)); s = d[np.ix_(jt, jx)]; v = max(np.nanpercentile(np.abs(s), 99.5), .5) if np.isfinite(s).any() else 1
            t0 = t[jt][0]; ax.imshow(s, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, extent=[x[jx][0], x[jx][-1], (t[jt][-1] - t0) / 3600, 0], interpolation="nearest")
            ax.add_patch(Rectangle((xa, (ta - t0) / 3600), xb - xa, (tb - ta) / 3600, fill=False, ec="k", lw=2, ls="--"))
            for m, col in zip(Z, ["lime", "orange", "magenta", "cyan"]):
                dd = Z[m]; o = dd[(dd.rec == rec) & (dd.x_end_m >= xa - pad) & (dd.x_start_m <= xb + pad) & (dd.tb >= ta - 3600) & (dd.ta <= tb + 3600) & (dd.is_alarm == 1)]
                for _, z in o.iterrows(): ax.add_patch(Rectangle((z.x_start_m, (z.ta - t0) / 3600), max(z.x_end_m - z.x_start_m, .5), (z.tb - z.ta) / 3600 + .03, fill=False, ec=col, lw=1))
                ax.plot([], [], color=col, label=m)
            for z in bo: ax.add_patch(Rectangle((z["start_m"], (z["t_start_s"] - t0) / 3600), z["end_m"] - z["start_m"], (z["t_end_s"] - z["t_start_s"]) / 3600, fill=False, ec="gray", lw=.8, ls=":"))
            ax.plot([], [], color="gray", ls=":", label="baseline"); ax.legend(fontsize=6, loc="lower right")
            ax.set_title(f"{c['id']}\nuser={c['presence']} {c['shape']}", fontsize=7); ax.set_xlabel("x, м"); ax.set_ylabel("ч")
            fig.tight_layout(); fig.savefig(f"docs/figures/cards2/{c['id']}.png", dpi=80); plt.close(fig)
df = pd.DataFrame(rows); df.to_csv("artifacts/eval/cards2_review.csv", index=False)
print(df.drop(columns=["notes"]).to_string())
