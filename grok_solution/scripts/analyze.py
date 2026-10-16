"""CLI: python scripts/analyze.py path/to/file.csv [--out out_dir] — зоны аномалий + типы + рисунок (CPU)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import real, infer
from dts.render import render

ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--out", default="out")
a = ap.parse_args()
r = real.load_csv(a.csv); o = infer.Analyzer().run(r["T"], r["valid"], r["x"], r["t"])
Path(a.out).mkdir(parents=True, exist_ok=True)
render(o, f"{a.out}/{r['name']}.png", r["name"])
zones = [{k: v for k, v in z.items() if k not in ("t0", "t1", "x0", "x1")} for z in o["zones"]]
json.dump(dict(file=a.csv, sha256=r["sha256"], zones=zones), open(f"{a.out}/{r['name']}.json", "w"), indent=1, ensure_ascii=False)
print(f"{len(zones)} зон, из них тревог: {sum(z['alarm'] for z in zones)} -> {a.out}")
