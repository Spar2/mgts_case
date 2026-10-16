"""Краткая таблица: val/test IoU,F1 @op по json-ам eval_stage1."""
import json, sys, glob
op = "far_1e-3"; files = sys.argv[1:] or sorted(glob.glob("artifacts/eval/*.json"))
for f in files:
    try: d = json.load(open(f))
    except Exception: continue
    if not isinstance(d, dict) or "val" not in d or not isinstance(d["val"], dict): continue
    for k, v in d["val"].items():
        if not k.endswith("@" + op) or k.startswith("base_"): continue
        t = d.get("test", {}).get(k, {}); w = t.get("weak") if isinstance(t, dict) else None
        g = lambda x, n: round(float(x.get(n, float("nan"))), 3) if isinstance(x, dict) else None
        print(f"{f.split('/')[-1]:30s} {k:32s} val_iou {g(v,'iou')} test_iou {g(t,'iou')} f1 {g(t,'f1')} P {g(t,'precision')} R {g(t,'recall')}")
