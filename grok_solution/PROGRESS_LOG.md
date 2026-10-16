# PROGRESS_LOG

## 2026-10-15 01:30 UTC
- Смена курса: новый автономный проект /workspace/dts_solution (haklolarp — только эталон для сравнения).
- dts_synth_v8: sha256 всех 245 артефактов MANIFEST.csv совпали (EXIT 0). Используется распакованная копия /workspace/synth_v8/d.
- haklolarp: git init + тег base-2026-10-15, baseline pytest зелёный (в venv /workspace/repo/run/.venv).
- Каноническая сетка 0.25 м × 120 с, 3 канала CNN (ΔT_time/σ, ΔT_space/σ, valid); кэш синтетики строится.

## 2026-10-15 02:30 UTC
- Этап 2 (oracle-боксы, GroupKFold(5) по seq_id): EventNet macro-F1 0.904 [0.877–0.924], LightGBM 0.939, rules haklolarp 0.328 (coarse 0.985/0.988/0.685).
- Этап 1, первая U-Net (24 эп.): при FAR 1e-3 (порог с val) IoU 0.36, F1 0.53; эталон median_x3+spatial_z при том же FAR — IoU 0.013; matched filter 0.06. При FAR_ref (=5.8 % фона, FAR эталона |Z|≥3.5) U-Net recall 0.87 vs 0.51, но precision ниже.
- Слабые события (≤2 °C): U-Net recall 0.42 при FAR_ref, ~0 при 1e-3 — главная цель улучшения (обучение с весом слабых событий запущено).
- /workspace/synth_v8/d удалён рутиной синтетики; кэш был построен раньше (sha256 MANIFEST сверены), labels/sequences v8 перескачаны с Drive, sha256 совпали.
- Параллельные агенты grok (tmux `grok`): lit (литература — готово), report, stage2, ae.

## 2026-10-15 03:35 UTC
- Раунд сравнения этапа 1 (test v8, порог с val при FAR=1e-3): U-Net IoU 0.363 / F1 0.532 / AP 0.515; pixel-LGBM 0.349; TCN 0.322; dilated FCN 0.326; matched filter 0.06; SSL-автоэнкодер 0.002 (не годится: восстанавливает и события); эталон median_x3+spatial_z 0.013.
- Ансамбль U-Net+TCN+pixel-LGBM: IoU 0.403, F1 0.574. Победитель по семейству — U-Net (CV-сегментация t×x); доводка: unet_ms (интегрирующие каналы), unet_w24 (deep ensemble).
- Этап 2 (агент stage2): контекстные признаки + смесь 0.5 EventNet + 0.5 LGBM: GroupKFold macro-F1 0.955 (было 0.904/0.939; rules 0.328). End-to-end на FAR=1e-3: event recall 0.54–0.60, точность класса на совпавших зонах 0.84; ложные зоны (в основном из неразмеченного «активного» фона) — главный источник ошибок.
- Реальные данные (предв., ансамбль @1e-3): карточки оператора presence=yes — найдено 7/15 (эталон 11/15), presence=no — 0/5 ложных (эталон 1/5).

## 07:35
Финальный выбор по val: ens_u2ft_tcn_lgbm (val 0.387, test 0.405, v10 0.404). Real-карточки 8/15, 0/5. CV/unsup интегрируются. Тесты 80 passed.
