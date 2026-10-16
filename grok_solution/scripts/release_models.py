"""Копирует детектор (одна сеть или ансамбль), этап 2 и пороги в models_release/ (малые файлы, коммитятся).
Пример: python scripts/release_models.py --eval artifacts/eval/stage1_v8_final.json --method ens_final --members unet:artifacts/stage1/unet.pt:16,tcn:artifacts/stage1/tcn.pt"""
import argparse, json, shutil
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--eval", required=True); ap.add_argument("--method", required=True)
ap.add_argument("--members", required=True, help="kind:path[:width],...  (kind: unet|unet_ms|tcn|dilated|pixel_lgbm)")
a = ap.parse_args()
d = Path("models_release"); d.mkdir(exist_ok=True)
mem = []
for i, spec in enumerate(a.members.split(",")):
    parts = spec.split(":"); kind, path = parts[0], parts[1]; width = int(parts[2]) if len(parts) > 2 else None
    fn = f"det{i}_{kind}{Path(path).suffix}"; shutil.copy(path, d / fn); mem.append(dict(kind=kind, file=fn, width=width))
shutil.copy("artifacts/stage2/eventnet.pt", d / "eventnet.pt"); shutil.copy("artifacts/stage2/lgbm.joblib", d / "lgbm.joblib")
ev = json.load(open(a.eval))
json.dump(dict(arch=mem[0]["kind"], members=mem, far_ref=ev["far_ref"], thresholds={"cnn_unet": ev["thresholds"][a.method]},
               method=a.method, source_eval=a.eval, synth_version=ev.get("version")), open(d / "stage1_thresholds.json", "w"), indent=1)
print("released", a.method, mem)
