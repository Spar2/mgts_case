import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Create a compact, read-only snapshot of reviewed ML experiments"
)
parser.add_argument("--ml-checkout", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--baseline-cpu-reproduced",
    action="store_true",
    help="Explicit operator assertion after independent verification",
)
args = parser.parse_args()
root = args.ml_checkout.resolve()
if subprocess.check_output(
    [
        "git",
        "status",
        "--porcelain",
        "--",
        "fiber",
        "docs/evidence/threshold_cal.json",
        "docs/evidence/hybrid_refine.json",
    ],
    cwd=root,
    text=True,
).strip():
    parser.error("Research inputs must be committed")
if args.output.exists():
    parser.error("Choose a new snapshot file; existing evidence is not overwritten")
commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=root, text=True
).strip()
paths = ["docs/evidence/threshold_cal.json", "docs/evidence/hybrid_refine.json"]
sources = [
    {"path": p, "sha256": hashlib.sha256((root / p).read_bytes()).hexdigest()}
    for p in paths
]
cal = json.loads((root / paths[0]).read_text(encoding="utf-8"))
hybrid = json.loads((root / paths[1]).read_text(encoding="utf-8"))
rows = []
for row in cal["rows"]:
    rows.append({k: v for k, v in row.items() if k not in ("witnesses",)})
modes = []
for name, row in hybrid["modes"].items():
    modes.append(
        {
            "mode": name,
            **{k: v for k, v in row.items() if k != "real"},
            "files": [
                {k: v for k, v in f.items() if k != "refined_zone_count"}
                for f in row["real"]["files"]
            ],
        }
    )
pack = {
    "schema_version": "dts-ml-experiments-v1",
    "source_commit": commit,
    "sources": sources,
    "scope": "local",
    "operational_pair": {
        "denoiser": "median_x3",
        "detector": "spatial_z_halo",
        "threshold": 3.5,
        "unchanged": True,
    },
    "calibration": {
        "manifest": cal["manifest"],
        "device_declared": cal["device"],
        "threshold_rule": cal["threshold_rule"],
        "target_far": cal["target_far"],
        "rows": rows,
    },
    "hybrid": {
        "manifest": hybrid["manifest"],
        "device_declared": hybrid["device"],
        "modes": modes,
    },
    "audit": {
        "baseline_cpu_reproduced": bool(args.baseline_cpu_reproduced),
        "neural_cuda_reproduced": False,
        "refinement_operational": False,
        "witness_origin": "agent_development_regression_not_human_labels",
        "limitations": [
            "Основные IoU и FAR рассчитаны на синтетических planted events. Реальные записи не имеют полного экспертного эталона.",
            "FAR — доля отмеченных фоновых ячеек, не ложные события в час.",
            "Восемь witnesses — агентские регрессионные наблюдения.",
            "Пять bootstrap-повторов недостаточны для уверенной оценки неопределённости.",
            "Протокол нейросетевых весов не полностью воспроизводим по manifest.",
            "Refinement может пересекать разрывы времени; восстановление округлённых границ и подсчёт зон требуют исправления.",
            "У hybrid своя baseline-маска из bounding boxes; её IoU нельзя напрямую сравнивать с threshold support.",
        ],
    },
}
out = args.output.parent
out.mkdir(parents=True, exist_ok=True)
data = json.dumps(
    pack, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
).encode()
args.output.write_bytes(data)
print(
    json.dumps(
        {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "rows": len(rows),
            "modes": len(modes),
        }
    )
)
