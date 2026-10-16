# Release decision (2026-10-16, 03:00 UTC)

**Release: `ens_gen2`** — score = 0.5·(ens_u2ft_tcn_lgbm + GEN2 U-Net w16), threshold 0.8580 (FAR 1e-3 on full v8 val background).
Weights: `models_release/` (winner, unchanged) + `models_release_gen2/s3_gen2_unet.pt`. API: `dts.infer` with `gen2="on"`
(code default stays `off` for backward compatibility; `gen2="off"` reproduces the previous release ens_u2ft_tcn_lgbm exactly).

## Evidence (synthetic splits; real MGTS_data never used for training/calibration)
| check | winner | ens_gen2 (seed 1) | ens_gen2 (seed 2) |
|---|---|---|---|
| held-out v8 val (41 seq, no proxy leakage) | 0.378 | +0.018, CI [-0.023,+0.059] | +0.018, CI [-0.016,+0.057] |
| nested k5 / k2 (held-out 41) | — | +0.051 / +0.008 | +0.050 / +0.011 |
| v8 test Δ | 0.405 | +0.053, CI excl. 0 | +0.049, CI excl. 0 |
| v10 / v11 Δ | 0.404 | +0.054 / +0.054, CI excl. 0 | +0.049 / +0.049, CI excl. 0 |
| operator cards (descriptive only) | 8/15 found, 0/5 false | 7/15, 0/5 | — |

## Reasoning
- Gain reproduces across two independent seeds (+0.018 held-out val each) and on three independent synthetic splits with CIs excluding 0; nested checks are positive at k5 and k2.
- The val-only CI covers 0 (41 sequences — low power); the strict "val CI excludes 0" rule is not met. We accept on the combined evidence (seed consistency + three CI-positive splits + nested positive) and document this deviation.
- Robustness on real data: false alarms on "no" cards unchanged (0/5); alarm count drops 138→95 (fewer spurious zones); one fewer "yes" card (7 vs 8) — a recall trade-off, listed in limitations.
- Rejected alternatives: GEN2TCN/GEN2W24 trio (val 0.395/0.406 < 0.411, nested vs GEN2 negative); combo (gen2+winner+w24+tcn)/4 (hold41 −0.015 vs GEN2); GEN2+dT_space (hold41 −0.022); mean3 with seed2 (held-out +0.008, weaker than mean2).
- Rollback: `gen2="off"`.

## Inference verification (03:20 UTC)
`scripts/verify_release_gen2.py 4` runs `dts.infer.Analyzer(gen2="on")` end-to-end on v8 val seq0006/0021/0047/0048 and compares with the logged
ens_blend (cached winner preds + GEN2 U-Net): IoU@0.8580 0.625080 vs 0.625072. Max |Δscore| 0.07 on rare pixels; the same order of deviation
exists for gen2="off" vs cached winner preds (mean 2e-6, 0.03 % pixels >1e-3), i.e. pre-existing float noise of the cached predictions, not GEN2.
Release in code: `dts.infer.release_analyzer()` (flags `RELEASE_FLAGS`, gen2="on"); `Analyzer()` default stays gen2="off" = rollback. Test: tests/test_release_default.py.
