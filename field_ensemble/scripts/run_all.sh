#!/usr/bin/env bash
# Полное воспроизведение (CPU). SYNTH=<корень dts_synth_vN> REAL=<папка с 1–6.csv> bash scripts/run_all.sh
set -euo pipefail
SYNTH=${SYNTH:-/workspace/synth_v8/d}; REAL=${REAL:-/workspace/mgts/MGTS_data}; SEED=${SEED:-0}
python scripts/build_cache.py --synth-root "$SYNTH"
VER=$(python -c "from dts import synth; print(synth.discover('$SYNTH')['version'])")
for A in unet tcn dilated; do python scripts/train_stage1.py --arch $A --cache artifacts/cache/$VER --seed $SEED; python scripts/predict.py --method $A --cache artifacts/cache/$VER; done
python scripts/predict.py --method matched_filter --cache artifacts/cache/$VER
python scripts/predict.py --method pixel_lgbm --cache artifacts/cache/$VER
python scripts/eval_stage1.py --cache artifacts/cache/$VER --labels "$SYNTH/labels.csv" --sequences "$SYNTH/sequences.csv"
python scripts/train_stage2.py --cache artifacts/cache/$VER --labels "$SYNTH/labels.csv" --seed $SEED
python scripts/release_models.py --ver $VER
python scripts/run_real.py --real-root "$REAL" --cards ${CARDS:-data/operator_cards/*.jsonl}
python scripts/make_report.py
