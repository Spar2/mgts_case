"""PDRS zones on REAL MGTS_data ensemble probability maps — descriptive only (no training/calibration).
Params frozen from synthetic val (artifacts/eval/pdrs_best_params.json); cards used only for reporting."""
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import real, infer
from dts.pdrs import extract_zones_pdrs, extract_zones_cc
bp = json.load(open("artifacts/eval/pdrs_best_params.json"))
A = infer.Analyzer(op="far_1e-3")
cards = []
for c in glob.glob("data/operator_cards/*.jsonl"):
    cards += [json.loads(l) for l in open(c) if l.strip()]
E0 = pd.Timestamp("1970-01-01", tz="UTC")
res = {"params": bp, "threshold": A.thr, "files": {}, "cards": []}
for f in sorted(glob.glob("/workspace/mgts/MGTS_data/*.csv")):
    r = real.load_csv(f); name = r["name"]
    o = A.run(r["T"], r["valid"], r["x"], r["t"]); p = o["grid"]
    s, v, h = o["score"], o["valid"], np.nan_to_num(o["dT"])
    Z = {"cc": extract_zones_cc(s, v, h, thr=A.thr), "pdrs": extract_zones_pdrs(s, v, h, p["t"], p["x"], **bp)}
    hrs = (p["t"][-1] - p["t"][0]) / 3600
    res["files"][name] = {k: dict(n_zones=len(z), zones_per_day=round(len(z) / hrs * 24, 1),
                                  n_hot=sum(q["polarity"] == "hot" for q in z), alarm_frac=float((lab > 0)[v].mean()))
                          for k, (lab, z) in Z.items()}
    for c in [c for c in cards if str(c.get("record")) == name]:
        ts = [(pd.Timestamp(q).tz_convert("UTC") if pd.Timestamp(q).tzinfo else pd.Timestamp(q, tz="UTC")) for q in c["time"]]
        ta, tb = [(q - E0).total_seconds() for q in ts]; xa, xb = c["meters"]
        ix = (p["x"] >= xa) & (p["x"] <= xb); it = (p["t"] >= ta) & (p["t"] <= tb)
        row = dict(id=c["id"], record=name, presence=c["presence"])
        for k, (lab, z) in Z.items():
            row[k] = int(len(set(np.unique(lab[np.ix_(it, ix)])) - {0})) if ix.any() and it.any() else None
        res["cards"].append(row)
    print(name, res["files"][name], flush=True)
for k in ("cc", "pdrs"):
    pos = [c for c in res["cards"] if c["presence"] in ("yes", "present", True, "anomaly")]
    res[f"summary_{k}"] = dict(cards_total=len(res["cards"]),
                               by_presence={str(pr): f"{sum(1 for c in res['cards'] if c['presence']==pr and (c[k] or 0)>0)}/{sum(1 for c in res['cards'] if c['presence']==pr)}"
                                            for pr in sorted({str(c['presence']) for c in res['cards']})})
print(res["summary_cc"], res["summary_pdrs"])
json.dump(res, open("artifacts/eval/pdrs_real.json", "w"), indent=1, default=float)
