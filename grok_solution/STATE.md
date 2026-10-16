# STATE (обновлять каждые ~20 мин) — 2026-10-15 04:20 UTC
Дедлайн 09:15 UTC, сворачивание с 08:45. Ветка: ml_grok_bot (grok_solution/), синк: /workspace/sync_push.sh "<msg>".
Python: /workspace/repo/run/.venv/bin/python. Логи: /workspace/grok_logs/. tmux `grok`: logs, lit, report, stage2, ae.

## Запущено
- train unet_w24 (artifacts/train_unet_w24.log), ~05:00 -> predict --method unet --width 24 --ckpt artifacts/stage1/unet_w24.pt --name unet_w24
- train unet_s2 (artifacts/train_unet_s2.log), ~05:35 -> predict --ckpt artifacts/stage1/unet_s2.pt --name unet_s2
- агент ae: v9 predict+eval -> artifacts/eval/stage1_v9_test.json

## Лучшее (val IoU@FAR1e-3 / test IoU, F1)
- ens_cnn3_lgbm (unet+unet_ms+tcn+pixelLGBM): 0.379 / 0.401, 0.573; ens_unet_tcn_lgbm 0.378 / 0.403, 0.574; unet 0.368 / 0.363, 0.532; эталон 0.034 / 0.013.
- unet_ms (интегр. каналы) 0.359 / 0.345 — не помог; гистерезис хуже; SSL-AE провал.
- Этап 2: смесь EventNet+LGBM GroupKFold 0.955 (rules 0.328). models_release = ens_unet_tcn_lgbm.

## Поздние задачи (04:25)
- chain artifacts/chain_preds.sh: ждёт w24/s2 и делает predict (unet_w24, unet_s2).
- cache v10 строится (artifacts/cache_v10.log; /workspace/synth_v10/d, accepted).
- агенты: stage2=CFAR/временная интеграция/TTA (task_stage2b), lit=доменный сдвиг+пофайловая калибровка (task_lit2), ae=focal/tversky v8+v9 (task_ae2), unsup=LOFO blind-spot на real (task_unsup), report=NOTE (task_report2). Задачи: /workspace/grok_logs/task_*.md
- План v10: v10 test как колонка eval; дообучение лидера на v8+v10 (выбор по val).

## 05:35
- round5 (val/test IoU@1e-3): ens_u2_tcn_lgbm 0.381/0.402; unet_w24 0.352/0.384; ens_u2 0.365/0.359. round6 (artifacts/round6.sh) — с unet_s2, r6_*.json.
- v10 test preds готовы (unet, unet_w24, tcn, pixel_lgbm) -> нужен eval с --cache artifacts/cache/v10 --thresholds-from v8.
- Диск тесный (synth_v11 30G не наш): удалены pred/v9, cache/v9.

## 06:03
- round6: ens_u3_tcn_lgbm_w val 0.380/test 0.398; ens_all6 0.380/0.394; unet_s2 0.368/0.388. ВЫБОР по val: ens_u2_tcn_lgbm (unet+unet_w24+tcn+pixel_lgbm) 0.381/0.402 -> released (artifacts/release_real.sh), run_real -> real_v8.json.
- v10 test (пороги v8 val): ens_u2 0.400, ens_unet 0.397, unet 0.330, baseline 0.011.
- unsup LOFO: cards 3/15, 0/5; synth IoU 0.212. cv: идёт. ae: v810ft идёт. postproc box15 выключен (прирост шумовой).
- Диск: train-кэши ужаты (scripts/slim_train_cache.py, без T/base_z).

## 06:45
- unet_v810ft (дообучение focal+tversky на v8+v10): v8 val 0.375/test 0.372; v10 test 0.337.
- round7: ens_u2ft_tcn_lgbm (unet_w24+unet_v810ft+tcn+pixel_lgbm) val 0.387 / test 0.405 -> НОВЫЙ ЛИДЕР. v10 preds идут (artifacts/chain_v10b.sh; eval в нём упадёт без --splits test — перезапустить eval вручную с --splits test).
- Released пока ens_u2_tcn_lgbm (real: cards 8/15, 0/5). После nm (ae, tag _nm) — финальный выбор по val, release, run_real.
- 08:00 сходимость: release -> run_real (карточки) -> make_report -> pytest -> push.

## 07:35 ФИНАЛ
- Выпущен ens_u2ft_tcn_lgbm (unet_w24+unet_v810ft+tcn+pixel_lgbm): v8 val 0.387, test IoU 0.405 F1 0.577; v10 test IoU 0.404 F1 0.575 (пороги v8 val). nm хуже (0.371 val).
- real: карточки yes 8/15 (эталон 10/15), no 0/5 (1/5). cluster_real обновлён (138 зон, k=2).
- cv: cv_ref/cv_pipe v8 test 0.203, v10 0.238, cards 3/15 0/5; unet_cv_post = unet. unsup: 3/15, 0/5, synth IoU 0.212.
- pytest 80 passed. Осталось: report-агент дописывает NOTE (task_final.md), финальный push.

## 08:08
- NOTE.md/RESULTS.md обновлены под победителя; тесты зелёные; финальный push.

## Дальше
1. unet_w24, unet_s2 -> ансамбли (выбор по val) -> eval final (stage1_v8_final.json) -> release -> run_real -> cluster_real -> e2e.
2. make_figures/make_report -> RESULTS.md, NOTE.md; PROGRESS_LOG; push. Wrap-up 08:45.

# СЕССИЯ 2 (11:15–17:00 UTC; wrap-up 16:30)
- Задачи агентов: /workspace/grok_logs/task_{ae3,cv3,lit4,realnet}.md (ae: цветной шум/симметрия/длинный контекст; cv: dts/zone_cv.py; lit: литература+PSD real; realnet: новый конвейер AdaBN+masked SSL+базовая линия, LOFO, docs/REAL_PIPELINE.md).
- v11 cache test/val строится (artifacts/cache_v11.log) -> eval победителя на v11 test (пороги v8 val).
- Отчёты: docs/EXPERIMENTS_S2.md, docs/REAL_PIPELINE.md; в конце NOTE.md.
- 11:55 v11 test (пороги v8 val): победитель IoU 0.404, F1 0.575 (≈v10: v11 меняет только шаблон F5); эталон 0.011. artifacts/eval/v11_ens_u2ft_tcn_lgbm.json
- lit: PSD фона real -> artifacts/real_noise_psd.npz (φ_t 0.32, φ_x 0.56 вне зон). ae: обучаются _cn/_cnsym/_cnsymlong. cv: dts/zone_cv.py готов (real_cv_*.png). realnet: стартовал 11:51.
- 12:30 идут: ae _cn* обучение; realnet (val веса); stage2 multiscale (task_stage2c.md).
- 12:55 realnet: первая версия synth IoU 0.138 (v8 test) — идёт абляция (вычитание базовой линии/AdaBN подозреваются). ae/stage2 работают.
- 13:45 multiscale: native выбран (без прироста). realnet: абляция — AdaBN 0.304, замена входа 0.091 < U-Net 0.375; варианты A(synth-выбор=U-Net)/B(фикс. физ. веса)/C(B+порог по файлу).
- 14:22 ждём: ae eval _cn/_cnsym (+_cnsymlong обучение), realnet B/C.
- 14:36 ae: ens_cn val 0.3847/test 0.408 (< победителя по val 0.387 -> победитель не меняется); unet_cnsym val 0.358. ae делает ens_cnsym + real описательно. realnet B/C идёт.
- 15:02 realnet B/C done: cards B 5/15, C 3/15 (0/5). Отчёт report по task_report_s2.md с 15:45.
- 15:15 REPORT_S2_DONE; pytest 111 passed; ae оценивает _cnsymlong до 16:10.

## 15:30 WRAP-UP prepared
- Победитель не изменён: ens_u2ft_tcn_lgbm val 0.387 / test 0.405 / v10 0.404 / v11 0.404.
- S2 улучшения (val ниже победителя): ens_cn 0.385, ens_cnsym 0.384, ens_cnsymlong 0.379; cooling test 0.636–0.727 у cn/cnsym.
- Multiscale: native.
- realnet A (U-Net) cards 7/15; B 5/15; C 3/15. AdaBN/замена входа отклонены по val.
- pytest: rerun сейчас. Commit 319ac60 + финальный push.

## S3 17:50 (ТЗ A–E + нейросетевые улучшения; работа продолжается без дедлайна)
- Сводка docs/INTERIM_1800.md, docs/TZ_AE.md; события /workspace/grok_logs/_events.log; супервизор: /workspace/grok_logs/supervise_snapshot.sh (окна tmux grok: task_A/B/D/E + lit/report/stage2/ae/unsup/cv/realnet).
- Кандидат ens_dom val 0.3898 (models_release_dom), релиз по умолчанию ens_u2ft_tcn_lgbm. Идут: DOM seed2 (unsup), domsweep (ae), DOM w24 (realnet), DOM lgbm (task_D), DOM TCN (task_E), SEL2 nested (stage2).
## 18:15 — ens_dom не продвинут (вложенная дельта +0.0002/+0.0013); DOM исключён (спектр шума с real). Идут: STATIC (task_B), DG синтетический приор (task_D), DOM w24/TCN/sweep доигрывают (только для сравнения). Релиз = ens_u2ft_tcn_lgbm.
## 18:55 — DOM-линия закрыта; DG(FreeSDG) отклонён; RANDCONV на проверке (test −0.016); STATIC ветвь val −0.0003, static recall 0.62/0.68, real narrow 10/16 (описательно), ждём базу победителя; идут MixStyle, ConvGRU, Boundary, Lovász, MTL. Релиз = ens_u2ft_tcn_lgbm.
## 19:25 — STATIC2 принят как opt-in канал (static recall 0/87→62/87 val, 0/88→60/88 test; val IoU −0.0007). Отклонены: RandConv, Lovász, Boundary, ConvGRU, MTL, DG (плато семейства дообучений U-Net). Идут: STATIC интеграция (report), STATIC3 гейт (task_B), soup (unsup), MixStyle (task_E). Релиз = ens_u2ft_tcn_lgbm.
## 20:45 СТАТУС (S3, непрерывный режим)
- Релиз не менялся: ens_u2ft_tcn_lgbm val 0.3867 / test 0.4053 / v10-v11 0.4039; карточки batch1+2 8/15, 0/5.
- Вложенная проверка (порог на трен. фолдах val): ens_dom +0.0002/+0.0013, asc 0.382 — не продвинуты; DOM исключён (статистики шума из real).
- Плато семейства дообучений U-Net (≥12 попыток, всё ≤ победителя по вложенной оценке): weak 0.371, axial 0.361, TTA 0.383, lovasz 0.383, boundary 0.385, DG/FreeSDG 0.383, RandConv (v10/v11 0.383), MixStyle 0.383, ConvGRU nested −0.0001, MTL +0.0004, soup −0.0001, polthr/accum шум.
- STATIC (узкие статичные горячие точки, только synth): opt-in флаг static=on в infer/run_real (по умолчанию off). Synth: static recall победителя 0/87 → 62/87 val, 60/88 test при val IoU −0.0007. Тайловый инференс исправлен (паритет на synth). Real описательно: узкие статичные из CARDS_REVIEW_2 — 0/16 (макс. 0.73–0.89 < порога); вклеенная 10°C точка на real → 0.93 ⇒ разрыв synth→real в форме реальных точек. STATIC5 (расширенный генератор) хуже; STATIC6 — последняя попытка.
- Этап 2: класс Static_Hotspot на synth F1 0.998, macro 0.957 (CV) / 0.946 test (opt-in, не в релизе).
- B PDRS гибрид за флагом zones=pdrs_hybrid; A: KID — диагностика, FBD — нет; D/E выключены.

### AUTO 21:00 interim
- последние события:
  - 20:44 SUPERVISOR: plateau declared for U-Net finetune family; lit -> survey of fundamentally different synth-only pipelines (LIT6); task_B STATIC6 last attempt
  - 20:49 LIT6 F/flow: one-step rectified flow on synth background only, score=abs residual + sign (Beizaee arXiv:2508.02889). No REFLECT VAE, no 1000-step DDPM. Fallback latent NLL 2-5 steps, v1 AUC 0.991 (Sakai arXiv:2504.
  - 20:49 LIT6 D/boxes: YOLOv10-N 2.3M 6.7G AP 38.5, NMS-free, no COCO weights (Wang arXiv:2405.14458). RT-DETR-R50 AP 53.1 AP-s 34.8, do not train (Zhao arXiv:2304.08069). Stride 1-2 on cable; raster confidence; zero-mass F
  - 20:49 LIT6 C/contour: polygon <=8 vertices, 1 refine, <=16 queries (Yao arXiv:2501.17688, table V 67.6/75.5/63.6 at 24.6 FPS, 2716 MB — their GPU). Not the boundary U-Net (val 0.385). Adopt only if val IoU@FAR1e-3 > 0.
  - 20:49 LIT6_DONE papers 4 advice 3 winner_val 0.387 winner_test 0.405 beats no pytest 2 passed
  - 20:53 SUPERVISOR: new pipeline FLOW (rectified-flow background prior, residual score, synth-only) -> task_E
- активные прогоны:
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 21:01 interim
- последние события:
  - 20:49 LIT6 F/flow: one-step rectified flow on synth background only, score=abs residual + sign (Beizaee arXiv:2508.02889). No REFLECT VAE, no 1000-step DDPM. Fallback latent NLL 2-5 steps, v1 AUC 0.991 (Sakai arXiv:2504.
  - 20:49 LIT6 D/boxes: YOLOv10-N 2.3M 6.7G AP 38.5, NMS-free, no COCO weights (Wang arXiv:2405.14458). RT-DETR-R50 AP 53.1 AP-s 34.8, do not train (Zhao arXiv:2304.08069). Stride 1-2 on cable; raster confidence; zero-mass F
  - 20:49 LIT6 C/contour: polygon <=8 vertices, 1 refine, <=16 queries (Yao arXiv:2501.17688, table V 67.6/75.5/63.6 at 24.6 FPS, 2716 MB — their GPU). Not the boundary U-Net (val 0.385). Adopt only if val IoU@FAR1e-3 > 0.
  - 20:49 LIT6_DONE papers 4 advice 3 winner_val 0.387 winner_test 0.405 beats no pytest 2 passed
  - 20:53 SUPERVISOR: new pipeline FLOW (rectified-flow background prior, residual score, synth-only) -> task_E
  - 21:00 SUPERVISOR: interruption-proofing: scripts/run_resumable.sh, QUEUE.md, autostate window (STATE block /15min, push /30min); agents' jobs run under tmux grok (detached from supervisor)
- активные прогоны:
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 21:16 interim
- последние события:
  - 21:13 STATIC6_EVAL ckpt=artifacts/static/static_net6_s400.pt pick=max_g98 val_iou=0.3859 dval=-0.0008 static_val=0.3333333333333333 winner_static=0.0 branch_static=0.5632183908045977 constraint=ok
  - 21:13 STATIC6_EVAL ckpt=artifacts/static/static_net6.pt step=800 val
  - 21:13 STATIC6_EVAL ckpt=artifacts/static/static_net6.pt pick=max_g95 val_iou=0.3859 dval=-0.0008 static_val=0.42528735632183906 winner_static=0.0 branch_static=0.6436781609195402 constraint=ok
  - 21:13 STATIC6_EVAL test ckpt=artifacts/static/static_net6.pt
  - 21:14 STATIC6_EVAL_DONE selected=max_g95 ckpt=artifacts/static/static_net6.pt step=800 val_iou=0.3859 test_iou=0.4037030828521059 dval=-0.0008 static_val=0.42528735632183906 static_test=0.42045454545454547 thr=0.8672 win
  - 21:14 STATIC6_REAL ckpt=artifacts/static/static_net6.pt thr=0.867185 selected=max_g95 descriptive_only
- активные прогоны:
  -       07:43 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_flow_train.py --resume auto >> artifacts/ckpt/s3_fl
  -       07:43 /workspace/repo/run/.venv/bin/python -u scripts/s3_flow_train.py --resume auto
- дальше: см. /workspace/grok_logs/QUEUE.md

## 21:22 СТАТУС (для возобновления: QUEUE.md)
- Релиз без изменений: ens_u2ft_tcn_lgbm val 0.3867 / test 0.4053.
- STATIC: плато (STATIC2 opt-in: synth static recall 0→62/87, val −0.0007; real узкие 0/16; STATIC5/6 хуже).
- Идут: FLOW (task_E, rectified-flow прайор фона), CENTERNET (task_B). Автоблоки STATE каждые 15 мин, автопуш каждые 30 мин (окно grok:autostate). Перезапуск обучений: scripts/run_resumable.sh.

### AUTO 21:31 interim
- последние события:
  - 21:14 STATIC6_EVAL_DONE selected=max_g95 ckpt=artifacts/static/static_net6.pt step=800 val_iou=0.3859 test_iou=0.4037030828521059 dval=-0.0008 static_val=0.42528735632183906 static_test=0.42045454545454547 thr=0.8672 win
  - 21:14 STATIC6_REAL ckpt=artifacts/static/static_net6.pt thr=0.867185 selected=max_g95 descriptive_only
  - 21:16 AUTOSTATE pushed 4f9968b
  - 21:16 STATIC6_DONE selected=max_g95 step=800 seed=20261015 val_iou=0.385866 dval=-0.000845 static_val=37/87 static_test=37/88 thr=0.867185 real_s_ge_thr=0/16 real_s_max=0.373 plateau=1 vs_static2=0 beats=0
  - 21:19 SUPERVISOR: STATIC PLATEAU (STATIC6 static 37/87, real 0/16, max 0.37) -> STATIC2 stays opt-in; task_B -> CenterNet head (LIT6 D lite)
  - 21:20 SUPERVISOR: FLOW alive (train step 120, resumable ckpt artifacts/ckpt/s3_flow/last.pt OK, silent in events only); static6 ckpt OK. YOLO-style deferred until CENTERNET result (same family), contour model low priorit
- активные прогоны:
  -       05:59 /workspace/repo/run/.venv/bin/python -u scripts/s3_flow_eval.py
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 21:46 interim
- последние события:
  - 21:41 CENTERNET_TRAIN step=800 loss=3.6777 center_hit=0.066 bg=0.036 n=1 sec=83
  - 21:41 CENTERNET_TRAIN step=900 loss=3.1123 center_hit=0.060 bg=0.034 n=1 sec=93
  - 21:41 CENTERNET_TRAIN step=1000 loss=3.3186 center_hit=0.100 bg=0.028 n=2 sec=104
  - 21:41 CENTERNET_TRAIN step=1100 loss=2.6489 center_hit=0.069 bg=0.043 n=2 sec=114
  - 21:41 CENTERNET_TRAIN step=1200 loss=2.8976 center_hit=0.078 bg=0.036 n=1 sec=125
  - 21:41 CENTERNET_TRAIN_DONE step=1200 sec=124.5 stopped=steps loss=2.8976 proxy_best_step=1000 center_hit=0.09954122081398964
- активные прогоны:
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 22:01 interim
- последние события:
  - 21:49 RESUMABLE launch s3_contour pid 1337822 : /workspace/repo/run/.venv/bin/python -u scripts/s3_contour_train.py
  - 21:52 SUPERVISOR: resume OK — contour alive (s3_contour epoch~8); CenterNet1 DONE underfit → task_B CENTERNET2; FLOW/STATIC6 closed. Spawn lit=LIT7 survey, task_D=MAE, stage2=DISTILL. YOLO deferred. load~1.4.
  - 21:54 SUPERVISOR: contour epoch~24 proxy_best~0.30; lit/MAE/DISTILL busy; task_B compacting (CENTERNET2 pending); autostate OK; no relaunch needed.
  - 21:56 SUPERVISOR: CONTOUR train DONE (best_epoch=26 proxy=0.3313); task_E starting eval. CENTERNET2 still coding on task_B. lit/MAE/DISTILL coding. No dead trains.
  - 21:58 SUPERVISOR: contour_eval running (pid alive); task_E watching; CENTERNET2 coding; lit/MAE/DISTILL coding; hold idle agents.
  - CONTOUR_DONE selected=ens_blend val=0.3855 test=0.3793 contour_val=0.1590 winner_val=0.3867 nested5_blend=0.3514 nested5_winner=0.3714 delta5=-0.0199 v10_blend=0.3738 v10_winner=0.4039 v11_blend=0.3738 v11_winner=0.4039 
- активные прогоны:
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 22:16 interim
- последние события:
  - 22:14 CENTERNET2_TRAIN step=3200 loss=2.4884 pos_p=0.227 center_hit=0.000 bg=0.006 n=6 ev=0.50 rad=9.0/10.2 sec=639
  - 22:14 SUPERVISOR: CENTERNET2 alive step 3240; ckpt ok. Watching MAE/DISTILL/MEMBANK.
  - 22:15 CENTERNET2_TRAIN step=3400 loss=2.6152 pos_p=0.134 center_hit=0.000 bg=0.009 n=5 ev=0.50 rad=12.4/7.4 sec=678
  - 22:15 RESUMABLE launch s3_distill pid 1353610 : /workspace/repo/run/.venv/bin/python -u scripts/s3_distill_train.py --max-minutes 42
  - 22:15 DISTILL_START width=16 max_min=42.0 elapsed_prev=0 remain=2520 threads=2 real_train=0
  - 22:15 CENTERNET2_TRAIN step=3600 loss=2.6295 pos_p=0.440 center_hit=0.781 bg=0.013 n=5 ev=0.50 rad=7.6/16.0 sec=718
- активные прогоны:
  -       12:19 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_centernet_train.py --steps 20000 --seed 20261015 --
  -       12:19 /workspace/repo/run/.venv/bin/python -u scripts/s3_centernet_train.py --steps 20000 --seed 20261015 --max-minutes 45 --run cente
  -       00:47 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_distill_train.py --max-minutes 42 --resume auto >> 
  -       00:47 /workspace/repo/run/.venv/bin/python -u scripts/s3_distill_train.py --max-minutes 42 --resume auto
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 22:31 interim
- последние события:
  - 22:29 CENTERNET2_TRAIN step=7800 loss=1.6824 pos_p=0.276 center_hit=0.868 bg=0.008 n=6 ev=0.62 rad=22.3/10.2 sec=1556
  - 22:29 MAE_TRAIN epoch=5 loss=1.8588 score=0.0538 sep=5.272 bg=1.653 restore=0.426 sec=222
  - 22:30 CENTERNET2_TRAIN step=8000 loss=1.1639 pos_p=0.416 center_hit=0.698 bg=0.008 n=6 ev=0.50 rad=22.0/11.8 sec=1600
  - 22:30 DISTILL_TRAIN step=1000 loss=0.2850 proxy=0.27025793492794037 sec=913
  - 22:30 MAE_TRAIN epoch=6 loss=1.8230 score=0.0000 sep=5.305 bg=1.481 restore=0.571 sec=262
  - 22:30 SUPERVISOR: alive CN=step 8120 DISTILL=5 MAE=5 MEM=3 load=9.46
- активные прогоны:
  -       27:21 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_centernet_train.py --steps 20000 --seed 20261015 --
  -       27:21 /workspace/repo/run/.venv/bin/python -u scripts/s3_centernet_train.py --steps 20000 --seed 20261015 --max-minutes 45 --run cente
  -       15:49 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_distill_train.py --max-minutes 42 --resume auto >> 
  -       15:49 /workspace/repo/run/.venv/bin/python -u scripts/s3_distill_train.py --max-minutes 42 --resume auto
  -       05:04 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_mae_train.py --resume auto >> artifacts/ckpt/s3_mae
  -       05:04 /workspace/repo/run/.venv/bin/python -u scripts/s3_mae_train.py --resume auto
  -       02:45 /workspace/repo/run/.venv/bin/python -u scripts/s3_membank_eval.py
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 22:46 interim
- последние события:
  - 22:44 DISTILL_TRAIN step=2800 loss=0.2236 proxy=0.29558633267879486 sec=1726
  - 22:44 CENTERNET2_TRAIN step=11800 loss=1.7141 pos_p=0.477 center_hit=0.210 bg=0.004 n=8 ev=0.62 rad=7.9/8.2 sec=2433
  - 22:45 CENTERNET2_TRAIN step=12000 loss=2.1738 pos_p=0.350 center_hit=0.763 bg=0.007 n=7 ev=0.62 rad=23.9/10.4 sec=2471
  - 22:45 SUPERVISOR: CN=step 12080 D=5 load=7.51
  - 22:45 DISTILL_TRAIN step=3000 loss=0.3408 proxy=0.1921234428882599 sec=1808
  - 22:45 CENTERNET2_TRAIN step=12200 loss=1.1133 pos_p=0.302 center_hit=0.824 bg=0.007 n=7 ev=0.50 rad=7.0/21.0 sec=2511
- активные прогоны:
  -       42:21 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_centernet_train.py --steps 20000 --seed 20261015 --
  -       42:21 /workspace/repo/run/.venv/bin/python -u scripts/s3_centernet_train.py --steps 20000 --seed 20261015 --max-minutes 45 --run cente
  -       30:49 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_distill_train.py --max-minutes 42 --resume auto >> 
  -       30:49 /workspace/repo/run/.venv/bin/python -u scripts/s3_distill_train.py --max-minutes 42 --resume auto
  -       12:44 /workspace/repo/run/.venv/bin/python -u scripts/s3_mae_eval.py
- дальше: см. /workspace/grok_logs/QUEUE.md

- 2026-10-15T23:01:19Z CLEANUP (user-approved): deleted pred v8/{unet,unet_cn,unet_cnsymlong,ens_cnsymlong,ens_cn,ens_tta} v10/{ens_u2_tcn_lgbm,unet_cn} (~6.1G) + /workspace/synth_v13 (30G). Kept synth_v8/10/11/12/14, winner+members, cnsym, cv_ref.

### AUTO 23:06 interim
- последние события:
  - 22:55 DISTILL_TRAIN step=3800 loss=0.3235 proxy=0.29687850177288055 sec=2400
  - 22:58 DISTILL_TRAIN_DONE step=3950 stopped=time best_step=900 best_proxy=0.19092248380184174 sec=2594 n_teacher=32 width=16 real_train=0
  - 22:59 SUPERVISOR: MAE re-verdict: rule bug (abs nested_min vs bar). Leakage check clean: MAE trained v8+v10 TRAIN only, real_train=0, real_noise_used=False, ens_mean4 fixed (4w+mae)/5 chosen among 3 fixed blends on val; 
  - 22:59 CENTERNET2_DONE selected=ens_blend step=12607 val_iou=0.349139 dval=-0.037574 test_iou=0.345218 winner_val=0.386712 nested5_delta=-0.0224 v10=0.384683 v11=0.384730 event_recall=0.490 event_precision=0.687 accept=0 
  - 2026-10-15T23:01:19Z CLEANUP (user-approved): deleted pred v8/{unet,unet_cn,unet_cnsymlong,ens_cnsymlong,ens_cn,ens_tta} v10/{ens_u2_tcn_lgbm,unet_cn} (~6.1G) + /workspace/synth_v13 (30G). Kept synth_v8/10/11/12/14, winn
  - 23:06 DISTILL_REJECT student_val=0.366683 winner_val=0.386713 dval=-0.020029 test=0.3792 winner_test=0.4053 v10=0.3484 v10_winner=0.4039 v11=0.3483 v11_winner=0.4039 nested5_delta=-0.0109 nested2_delta=-0.0199 close=0 be
- активные прогоны:
- дальше: см. /workspace/grok_logs/QUEUE.md

### MAE 23:12
- ACCEPTED (rule fix): ens_mean4 val 0.3995 / test 0.4336, nested k5 delta +0.0068, k2 +0.0100. Абсолютный nested_min 0.3782 не планка (у победителя k5 0.3714). Флаг infer mae=off по умолчанию. Релиз models_release_mae/, models_release не менялся. Карточки batch1+2: 8/15 и 1/5 (победитель 8/15 и 0/5). real_train=0. Строка MAE_PROMOTE_DONE.
## 23:15 — MAE ens_mean4 принят кандидатом релиза (models_release_mae, флаг mae=on): val 0.3995 / test 0.4336 / v10 0.4299 / v11 0.4298, вложенные +0.0068/+0.0100. Отклонены: CENTERNET2 (val 0.349), DISTILL (0.367), FLOW. Диск: см. отчёт.

### AUTO 23:21 interim
- последние события:
  - 23:06 DISTILL_REJECT student_val=0.366683 winner_val=0.386713 dval=-0.020029 test=0.3792 winner_test=0.4053 v10=0.3484 v10_winner=0.4039 v11=0.3483 v11_winner=0.4039 nested5_delta=-0.0109 nested2_delta=-0.0199 close=0 be
  - 23:12 MAE_PROMOTE_DONE selected=ens_mean4 rule=same_fold_gain val=0.3994942836 test=0.4335810955 winner_val=0.3867123582 winner_test=0.4052762065 delta_test=+0.0283048890 delta5=+0.0068231977 delta2=+0.0100119436 k5_fold
  - 23:14 COMMIT ce02e88 MAE candidate
  - 23:15 SUPERVISOR: MAE ens_mean4 kept as candidate (not default). task_D -> MAE_WHY (test>val analysis: P/R by split/type, threshold effects); task_B -> MAE_SEEDS (3 seeds x mixes 3:1/4:1/5:1, bootstrap CI val/test/v10/v1
  - 23:15 SUPERVISOR: MEMBANK rejected (nested -0.0048). task_E -> MAEFT (MAE-pretrained U-Net encoder). v14 has no cache — not building (disk 88%)
  - 23:18 SUPERVISOR: DEADLINE 05:00 UTC set: improve to 04:15 (MAE seeds/why/MAEFT + GEN2 generator by ae), 04:15 RELEASE_DECISION, 04:15-04:50 EXPERT report (task_expert.md, report window), 04:55 push. QUEUE.md updated
- активные прогоны:
- дальше: см. /workspace/grok_logs/QUEUE.md
## 23:30 — ДЕДЛАЙН 05:00 UTC (план в /workspace/grok_logs/QUEUE.md). MAE: «MAE» в ens_mean4 = веса ИНИЦИАЛИЗАЦИИ (best_epoch=-1), сиды делили одну init → проверка init vs trained (task_B), сравнение с фиксированным фильтром (task_D). Идут: GEN2 генератор (ae), MAEFT (task_E). Кандидат ens_mean4 не по умолчанию.

### AUTO 23:36 interim
- последние события:
  - 23:34 MAEFT_TRAIN epoch=1 step=80 mae=0.1450 rand=0.1105 v810=0.5065 sec=69
  - 23:34 MAEFT_TRAIN epoch=2 step=120 mae=0.1479 rand=0.0963 v810=0.5241 sec=104
  - 23:35 MAEFT_TRAIN epoch=3 step=160 mae=0.1525 rand=0.1181 v810=0.4916 sec=138
  - 23:35 MAEFT_TRAIN epoch=4 step=200 mae=0.1645 rand=0.1467 v810=0.5088 sec=173
  - 23:36 SUPERVISOR: backups models_backup/ + /workspace/backup_weights_20261015.tar (214MB); ARCH_SPECS regenerated (MAE candidate); spec queued in SCHEMA_OUTBOX.md (no SendToAgent tool in supervisor)
  - 23:36 MAEFT_TRAIN epoch=5 step=240 mae=0.1560 rand=0.1061 v810=0.5047 sec=210
- активные прогоны:
  -       05:28 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_train.py --resume auto >> artifacts/ckpt/s3_ge
  -       05:28 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_train.py --resume auto
  -       04:00 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_maeft_train.py --resume auto >> artifacts/ckpt/s3_m
  -       04:00 /workspace/repo/run/.venv/bin/python -u scripts/s3_maeft_train.py --resume auto
- дальше: см. /workspace/grok_logs/QUEUE.md
## 23:50 — бэкапы готовы (models_backup/, /workspace/backup_weights_20261015.tar); ARCH_SPECS обновлён (MAE=кандидат, init-веса); спеки в SCHEMA_OUTBOX.md (нет SendToAgent). MAEFT от MAE-энкодера ≈0 (прокси) vs v810 0.49 → бесполезно. Идут сиды MAE (init vs trained), GEN2, MAE_WHY.

### AUTO 23:51 interim
- последние события:
  - 23:50 MAE_S2R_TRAIN epoch=8 loss=1.7336 score=0.0293 sep=5.760 bg=1.572 restore=0.438 sec=430
  - 23:50 MAE_S2R_TRAIN_DONE best_epoch=-1 score=0.4399 trained_epoch=0 trained_score=0.1310 trained_head_max=0.09170526266098022 sep=6.097714225451152 bg=2.4650871753692627 restore=0.4288753208798484 stop=patience sec=431 p
  - 23:50 MAEFT_TRAIN epoch=27 step=1120 mae=0.0152 rand=0.0355 v810=0.4634 sec=1046
  - 23:50 MAEFT_TRAIN epoch=28 step=1160 mae=0.0002 rand=0.1197 v810=0.4669 sec=1080
  - 23:51 MAEFT_TRAIN epoch=29 step=1200 mae=0.0118 rand=0.1439 v810=0.4705 sec=1115
  - 23:51 MAEFT_TRAIN_END step=1200 stopped=epochs sec=1116 mae_ep=0 mae_proxy=0.1793 rand_ep=4 rand_proxy=0.1467 v810_ep=16 v810_proxy=0.5293 copied=36 dw_mae=7.141386 dw_key=e1.0.1.running_var real_train=0
- активные прогоны:
  -       20:28 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_train.py --resume auto >> artifacts/ckpt/s3_ge
  -       20:28 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_train.py --resume auto
  -       00:25 /workspace/repo/run/.venv/bin/python -u scripts/s3_mae_seeds.py
- дальше: см. /workspace/grok_logs/QUEUE.md
## 00:05 — MAEFT отклонён. task_E делает черновик REAL_EXPERT_REPORT + scripts/make_expert_report.py (для финала в 04:15). Ждём MAE_SEEDS, MAE_WHY, GEN2.

### AUTO 00:06 interim
- последние события:
  - 23:50 MAEFT_TRAIN epoch=28 step=1160 mae=0.0002 rand=0.1197 v810=0.4669 sec=1080
  - 23:51 MAEFT_TRAIN epoch=29 step=1200 mae=0.0118 rand=0.1439 v810=0.4705 sec=1115
  - 23:51 MAEFT_TRAIN_END step=1200 stopped=epochs sec=1116 mae_ep=0 mae_proxy=0.1793 rand_ep=4 rand_proxy=0.1467 v810_ep=16 v810_proxy=0.5293 copied=36 dw_mae=7.141386 dw_key=e1.0.1.running_var real_train=0
  - 23:51 AUTOSTATE pushed 4409560
  - 00:01 MAEFT_DONE selected=ens_mean_v810 val=0.3889 test=0.3947 alone_val=0.3860 winner_val=0.386712 nested5_delta=+0.0055 nested2_delta=+0.0037 nested5_sel=0.3769 nested5_winner=0.3714 v10=0.3890 v10_winner=0.4039 v11=0.
  - 00:04 SUPERVISOR: MAEFT rejected (v10/v11 0.389). task_E -> EXPERT report draft + scripts/make_expert_report.py now (de-risk deadline)
- активные прогоны:
  -       35:30 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_train.py --resume auto >> artifacts/ckpt/s3_ge
  -       35:30 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_train.py --resume auto
  -       15:27 /workspace/repo/run/.venv/bin/python -u scripts/s3_mae_seeds.py
- дальше: см. /workspace/grok_logs/QUEUE.md
## 00:18 — MAE=фиксированный пост-фильтр CDF(|dT_space|) (голова нулевая); кандидат честно переименован; ждём ДИ (task_B), FUSENET (task_D), GEN2 (ae), черновик эксперт-отчёта (task_E).

### AUTO 00:21 interim
- последние события:
  - 23:51 AUTOSTATE pushed 4409560
  - 00:01 MAEFT_DONE selected=ens_mean_v810 val=0.3889 test=0.3947 alone_val=0.3860 winner_val=0.386712 nested5_delta=+0.0055 nested2_delta=+0.0037 nested5_sel=0.3769 nested5_winner=0.3714 v10=0.3890 v10_winner=0.4039 v11=0.
  - 00:04 SUPERVISOR: MAEFT rejected (v10/v11 0.389). task_E -> EXPERT report draft + scripts/make_expert_report.py now (de-risk deadline)
  - 00:14 SUPERVISOR: MAE_WHY: ens_mean4 == (4*winner + CDF(|dT_space|))/5 exactly (MAE head zero) -> candidate renamed honestly as winner + fixed spatial-residual post-filter (non-neural); gain = nuisance FP suppression (fp
  - 00:14 SUPERVISOR: task_D -> FUSENET (learned conv fusion of winner score + dT_space residual, trained on synth v10 train preds only)
  - 00:18 EXPERT_DRAFT_DONE release=ens_u2ft_tcn_lgbm prefix=draft n_files=6 n_zones=138 n_alarm=138 sec=213.3 json=artifacts/eval/real_draft.json csv=artifacts/eval/real_draft.csv fig=docs/figures/real_draft_<file>.png repo
- активные прогоны:
  -       50:30 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_train.py --resume auto >> artifacts/ckpt/s3_ge
  -       50:30 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_train.py --resume auto
  -       30:27 /workspace/repo/run/.venv/bin/python -u scripts/s3_mae_seeds.py
- дальше: см. /workspace/grok_logs/QUEUE.md
## 00:35 — REAL_EXPERT_REPORT черновик 2 (16 значимых зон, интерпретация, карточки); ждём DTSPACE CI (task_B), FUSENET (task_D), GEN2 (ae).

### AUTO 00:39 interim
- последние события:
  - 00:18 EXPERT_DRAFT_DONE release=ens_u2ft_tcn_lgbm prefix=draft n_files=6 n_zones=138 n_alarm=138 sec=213.3 json=artifacts/eval/real_draft.json csv=artifacts/eval/real_draft.csv fig=docs/figures/real_draft_<file>.png repo
  - 00:21 AUTOSTATE pushed 7a1d025
  - 00:25 RESUMABLE launch s3_fusenet pid 1418627 : /workspace/repo/run/.venv/bin/python -u scripts/s3_fusenet_train.py
  - 00:25 SUPERVISOR: (resumed, same supervisor) task_E was sent EXPERT_DRAFT2 at 00:21 (expert-level interpretation, --release ens_mean4/static flags)
  - 00:27 RESUMABLE launch s3_fusenet pid 1419813 : /workspace/repo/run/.venv/bin/python -u scripts/s3_fusenet_train.py
  - 00:31 EXPERT_DRAFT2_DONE release=ens_u2ft_tcn_lgbm prefix=draft static=off n_files=6 n_zones=138 n_alarm=138 n_shown=16 sec=0.0 json=artifacts/eval/real_draft.json report=docs/REAL_EXPERT_REPORT.md real_train=0
- активные прогоны:
  -       45:29 /workspace/repo/run/.venv/bin/python -u scripts/s3_mae_seeds.py
  -       09:30 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_fusenet_train.py --resume auto >> artifacts/ckpt/s3
  -       09:30 /workspace/repo/run/.venv/bin/python -u scripts/s3_fusenet_train.py --resume auto
- дальше: см. /workspace/grok_logs/QUEUE.md
## 00:47 — GEN2 сильный кандидат: ens_blend val 0.4106 / test 0.4579 / v10-v11 0.4576, nested k5 +0.031, real static 6/16; проверки GEN2_CHECK (ae). Спека в SCHEMA_OUTBOX.

### AUTO 00:54 interim
- последние события:
  - 00:44 EXPERT_DRAFT2_DONE release=ens_u2ft_tcn_lgbm prefix=draft static=off n_files=6 n_zones=138 n_alarm=138 n_shown=16 sec=0.0 json=artifacts/eval/real_draft.json report=docs/REAL_EXPERT_REPORT.md real_train=0
  - 00:44 EXPERT_DRAFT2_DONE release=ens_u2ft_tcn_lgbm prefix=draft static=off n_files=6 n_zones=138 n_alarm=138 n_shown=16 sec=0.0 json=artifacts/eval/real_draft.json report=docs/REAL_EXPERT_REPORT.md real_train=0
  - 00:46 FUSENET_INIT proxy_iou=0.0000 far=0.00100 params=4145
  - 00:47 SUPERVISOR: task_E -> GEN2W24 (gen2 finetune of unet_w24, proxy on v10 val not v8 val)
  - 00:51 FUSENET_TRAIN_DONE best_epoch=0 proxy_iou=0.5110437545832315 stop=patience sec=328 params=4145 n_train=48 n_hold=12 real_train=0 maps_deleted=1
  - 00:54 RESUMABLE launch s3_gen2w24 pid 1455711 : /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2w24_train.py
- активные прогоны:
  -    01:00:29 /workspace/repo/run/.venv/bin/python -u scripts/s3_mae_seeds.py
  -       02:50 /workspace/repo/run/.venv/bin/python -u scripts/s3_fusenet_eval.py
  -       02:30 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_check.py
  -       00:42 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2w24_train.py --resume auto >> artifacts/ckpt/s3
  -       00:42 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2w24_train.py --resume auto
- дальше: см. /workspace/grok_logs/QUEUE.md
## 01:01 — FUSENET отклонён; идут GEN2_CHECK (ae), GEN2W24 (task_E), GEN2TCN (task_D), DTSPACE CI (task_B). Спеки GEN2, FuseNet в SCHEMA_OUTBOX.md.
## 01:07 — MAE: решено (фиксированный фильтр dT_space; ДИ val включает 0 → не по умолчанию). Идёт COMBO (task_B).

### AUTO 01:09 interim
- последние события:
  - 00:57 SUPERVISOR: FUSENET rejected (val 0.351, nested -0.052). task_D -> GEN2TCN
  - 01:01 RESUMABLE launch s3_gen2tcn pid 1466663 : /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2tcn_train.py
  - 01:03 MAE_SEEDS_DONE seeds=20261015,20261016,20261017 ratios=3,4,5 v14=absent s0_init_w4_val=0.399494 init_w4_val_d=s0:+0.0128,s1:+0.0128,s2:+0.0128 trained_w4_val_d=s0:+0.0097,s1:-0.0087,s2:+0.0109 init_dmin=+0.0107 ini
  - 01:05 DTSPACE_CI_DONE filter=CDF(|dT_space|) ratios=3,4,5 n_boot=1000 v14=absent w3_val_d=+0.0107 ci=-0.0124,+0.0341 w3_test_d=+0.0335 ci=+0.0178,+0.0527 w3_v10_d=+0.0309 ci=+0.0151,+0.0515 w3_v11_d=+0.0308 ci=+0.0157,+0
  - 01:06 SUPERVISOR: MAE init question RESOLVED: gain = fixed CDF(|dT_space|) post-filter; trained MAE seeds unstable (-0.031..+0.014); dT_space blend: val CI includes 0 (4:1 -0.0023..+0.0297), test/v10/v11 CI exclude 0 -> 
  - 01:07 GEN2_CHECK_DONE hold41_refit_blend=0.396502 winner41=0.378442 delta=+0.018060 CI95=[-0.023092,+0.059109] covers0=yes frozen41_blend=0.395223 nested_k5=+0.050517 nested_k2=+0.008002 test_d=+0.052644 CI=[+0.025304,+0
- активные прогоны:
  -       15:44 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2w24_train.py --resume auto >> artifacts/ckpt/s3
  -       15:44 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2w24_train.py --resume auto
  -       08:24 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2tcn_train.py --resume auto >> artifacts/ckpt/s3
  -       08:24 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2tcn_train.py --resume auto
- дальше: см. /workspace/grok_logs/QUEUE.md
## 01:12 — GEN2_CHECK: val41 +0.018 (ДИ вкл. 0), nested k5 +0.051, test/v10/v11 +0.053 (ДИ искл. 0); карточки 7/15 0/5. Релиз-решение в 04:15; идёт сид 2 (ae).

### AUTO 01:26 interim
- последние события:
  - 01:03 MAE_SEEDS_DONE seeds=20261015,20261016,20261017 ratios=3,4,5 v14=absent s0_init_w4_val=0.399494 init_w4_val_d=s0:+0.0128,s1:+0.0128,s2:+0.0128 trained_w4_val_d=s0:+0.0097,s1:-0.0087,s2:+0.0109 init_dmin=+0.0107 ini
  - 01:05 DTSPACE_CI_DONE filter=CDF(|dT_space|) ratios=3,4,5 n_boot=1000 v14=absent w3_val_d=+0.0107 ci=-0.0124,+0.0341 w3_test_d=+0.0335 ci=+0.0178,+0.0527 w3_v10_d=+0.0309 ci=+0.0151,+0.0515 w3_v11_d=+0.0308 ci=+0.0157,+0
  - 01:06 SUPERVISOR: MAE init question RESOLVED: gain = fixed CDF(|dT_space|) post-filter; trained MAE seeds unstable (-0.031..+0.014); dT_space blend: val CI includes 0 (4:1 -0.0023..+0.0297), test/v10/v11 CI exclude 0 -> 
  - 01:07 GEN2_CHECK_DONE hold41_refit_blend=0.396502 winner41=0.378442 delta=+0.018060 CI95=[-0.023092,+0.059109] covers0=yes frozen41_blend=0.395223 nested_k5=+0.050517 nested_k2=+0.008002 test_d=+0.052644 CI=[+0.025304,+0
  - 01:11 SUPERVISOR: GEN2_CHECK: held-out 41 val +0.018 CI[-0.023,+0.059] covers 0; nested k5 +0.051 k2 +0.008; test/v10/v11 +0.053 CI excl 0; cards 7/15 0/5 (winner 8/15 0/5); narrow real 0/16. Decision at 04:15. ae -> GEN
  - 01:18 RESUMABLE launch s3_gen2_seed2 pid 1481157 : /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_seed2_train.py
- активные прогоны:
  -       30:44 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2w24_train.py --resume auto >> artifacts/ckpt/s3
  -       30:44 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2w24_train.py --resume auto
  -       23:24 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2tcn_train.py --resume auto >> artifacts/ckpt/s3
  -       23:24 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2tcn_train.py --resume auto
  -       06:44 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_seed2_train.py --resume auto >> artifacts/ckpt
  -       06:44 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_seed2_train.py --resume auto
  -       04:06 /workspace/repo/run/.venv/bin/python -u scripts/s3_combo.py
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 01:43 interim
- последние события:
  - 01:07 GEN2_CHECK_DONE hold41_refit_blend=0.396502 winner41=0.378442 delta=+0.018060 CI95=[-0.023092,+0.059109] covers0=yes frozen41_blend=0.395223 nested_k5=+0.050517 nested_k2=+0.008002 test_d=+0.052644 CI=[+0.025304,+0
  - 01:11 SUPERVISOR: GEN2_CHECK: held-out 41 val +0.018 CI[-0.023,+0.059] covers 0; nested k5 +0.051 k2 +0.008; test/v10/v11 +0.053 CI excl 0; cards 7/15 0/5 (winner 8/15 0/5); narrow real 0/16. Decision at 04:15. ae -> GEN
  - 01:18 RESUMABLE launch s3_gen2_seed2 pid 1481157 : /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_seed2_train.py
  - 01:26 AUTOSTATE pushed 40f60f1
  - 01:36 COMBO1 c=pending n_boot=1000 seed=20261016 proxy=seq0006,seq0021,seq0047,seq0048 a_val45_iou=0.410639 a_val45=d=+0.023926 ci=-0.014372,+0.068455 z=1 a_hold41_iou=0.396502 a_hold41=d=+0.018060 ci=-0.023092,+0.059109
  - 01:36 SUPERVISOR: resumed as sole supervisor (watchdog 01:36). trains alive gen2w24/gen2tcn/gen2_seed2; COMBO1 beats=0 (a: hold41 +0.018 CI covers 0; b=a+dT_space worse hold41 -0.022)
- активные прогоны:
  -       45:46 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2w24_train.py --resume auto >> artifacts/ckpt/s3
  -       45:46 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2w24_train.py --resume auto
  -       38:26 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2tcn_train.py --resume auto >> artifacts/ckpt/s3
  -       38:26 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2tcn_train.py --resume auto
  -       21:46 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_seed2_train.py --resume auto >> artifacts/ckpt
  -       21:46 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_seed2_train.py --resume auto
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 02:00 interim
- последние события:
  - 01:36 SUPERVISOR: resumed as sole supervisor (watchdog 01:36). trains alive gen2w24/gen2tcn/gen2_seed2; COMBO1 beats=0 (a: hold41 +0.018 CI covers 0; b=a+dT_space worse hold41 -0.022)
  - 01:45 SUPERVISOR: GEN2W24 train DONE best=step0 (proxy v10 0.3659 init; finetuned 0.289-0.292) -> gen2 finetune does not help w24 on v10 proxy; task_E evaluating. gen2tcn/seed2 alive. report -> EXPERT_GEN2_PREP (ens_gen2
  - 01:51 GEN2TCN_TRAIN_DONE step=2893 best_step=0 proxy_iou=0.367942 stop=budget sec=3000 params=78723 real_train=0
  - 01:52 EXPERT_DRAFT_DONE release=ens_gen2 prefix=draft static=off n_files=6 n_zones=95 n_alarm=95 n_shown=16 sec=225.4 json=artifacts/eval/real_draft.json csv=artifacts/eval/real_draft.csv fig=docs/figures/real_draft_<fil
  - 01:55 SUPERVISOR: GEN2TCN train DONE best_step=0 (finetune 0.295 < init 0.368 on v10 proxy) -> gen2 recipe helps only U-Net w16; seed2 proxy mean(seed2,winner) 0.3843->0.4025 (consistent with GEN2). task_D/task_E eval to
  - 01:56 EXPERT_GEN2_PREP_DONE release=ens_gen2 prefix=draft static=off gen2=on thr=0.8579896092424856 zones/alarms ens_gen2 32/8/18/13/4/20 total 95 vs winner 38/7/49/17/6/21 total 138; significant shared 11 only_winner 5 
- активные прогоны:
  -       36:46 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_seed2_train.py --resume auto >> artifacts/ckpt
  -       36:46 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_seed2_train.py --resume auto
  -       13:13 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2w24_eval.py
  -       03:37 /workspace/repo/run/.venv/bin/python -u scripts/s3_combo_c.py
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 02:19 interim
- последние события:
  - 02:00 AUTOSTATE pushed 11c880f
  - 02:05 SUPERVISOR: EXPERT_GEN2_PREP done (ens_gen2 in make_expert_report; 95 vs 138 alarms; cards 7/15 vs 8/15, no 0/5 both; 286 tests pass). report -> EXPERT_SUMMARY (exec summary, top-5 field checks, synth metrics table
  - 02:06 SUPERVISOR(main): ARCH_SPECS + outbox for GEN2-W24, GEN2-TCN
  - 02:09 EXPERT_DRAFT2_DONE release=ens_gen2 prefix=draft static=off n_files=6 n_zones=95 n_alarm=95 n_shown=16 sec=0.1 json=artifacts/eval/real_draft.json report=docs/REAL_EXPERT_REPORT.md real_train=0
  - 02:09 EXPERT_DRAFT2_DONE release=ens_gen2 prefix=draft static=off n_files=6 n_zones=95 n_alarm=95 n_shown=16 sec=0.0 json=artifacts/eval/real_draft.json report=docs/REAL_EXPERT_REPORT.md real_train=0
  - 02:14 EXPERT_SUMMARY_DONE release=ens_gen2 prefix=draft report-only no new inference; brief<=25 lines alarms 95 shown 16 top5 files 1/6/4/6/4; synth IoU val 0.3867/0.4106 test 0.4053/0.4579 v10 0.4039/0.4576 v11 0.4039/0
- активные прогоны:
  -       28:15 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2w24_eval.py
  -       18:39 /workspace/repo/run/.venv/bin/python -u scripts/s3_combo_c.py
  -       12:11 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2tcn_eval.py
  -       00:00 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_seed2_eval.py
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 02:34 interim
- последние события:
  - 02:09 EXPERT_DRAFT2_DONE release=ens_gen2 prefix=draft static=off n_files=6 n_zones=95 n_alarm=95 n_shown=16 sec=0.0 json=artifacts/eval/real_draft.json report=docs/REAL_EXPERT_REPORT.md real_train=0
  - 02:14 EXPERT_SUMMARY_DONE release=ens_gen2 prefix=draft report-only no new inference; brief<=25 lines alarms 95 shown 16 top5 files 1/6/4/6/4; synth IoU val 0.3867/0.4106 test 0.4053/0.4579 v10 0.4039/0.4576 v11 0.4039/0
  - 02:19 GEN2W24_DONE selected=ens_trio val=0.405578 test=0.451666 winner_val=0.386712 gen2_blend_val=0.410639 delta_winner=+0.018866 delta_gen2=-0.005061 v10=0.453856 v11=0.453815 nested_k5_vs_winner=+0.026967 nested_k2_vs
  - 02:21 SUPERVISOR: GEN2 seed2 train DONE (proxy v10 val mean(seed2,winner) 0.3843->0.4117, best at last step); ae evaluating on full v8 val (no v8 proxy files used -> clean held-out check). GEN2W24 ens_trio val 0.4056 < g
  - 02:32 GEN2TCN_DONE selected=ens_trio val=0.395196 test=0.450555 winner_val=0.386712 gen2_blend_val=0.410639 delta_winner=+0.008484 delta_gen2=-0.015442 v10=0.454521 v11=0.454427 nested_k5_vs_winner=+0.025930 nested_k2_vs
  - 02:32 BEATS_WINNER GEN2TCN ens_trio val 0.395196 test 0.450555 winner_val 0.386712
- активные прогоны:
  -       33:39 /workspace/repo/run/.venv/bin/python -u scripts/s3_combo_c.py
  -       09:40 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_seed2_eval.py
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 02:55 interim
- последние события:
  - 02:21 SUPERVISOR: GEN2 seed2 train DONE (proxy v10 val mean(seed2,winner) 0.3843->0.4117, best at last step); ae evaluating on full v8 val (no v8 proxy files used -> clean held-out check). GEN2W24 ens_trio val 0.4056 < g
  - 02:32 GEN2TCN_DONE selected=ens_trio val=0.395196 test=0.450555 winner_val=0.386712 gen2_blend_val=0.410639 delta_winner=+0.008484 delta_gen2=-0.015442 v10=0.454521 v11=0.454427 nested_k5_vs_winner=+0.025930 nested_k2_vs
  - 02:32 BEATS_WINNER GEN2TCN ens_trio val 0.395196 test 0.450555 winner_val 0.386712
  - 02:34 AUTOSTATE pushed f69ac5b
  - 02:34 COMBO_DONE members=gen2w24@step0,gen2tcn@step0 init_max_abs=0 formula=(gen2+winner+w24+tcn)/4 n_boot=1000 seed=20261016 c_val45_iou=0.392934 c_val45=d=+0.006222 ci=-0.016676,+0.044839 z=1 c_hold41_iou=0.381847 c_ho
  - 02:51 GEN2_SEED2_DONE seed=20261018 step=4507 proxy_v10=0.411677 mean2_41=0.396706 winner41=0.378442 d=+0.018264 CI=[-0.015689,+0.057203] covers0=yes mean3_41=0.386109 d=+0.007667 CI=[-0.043221,+0.056964] covers0=yes nes
- активные прогоны:
- дальше: см. /workspace/grok_logs/QUEUE.md

## 03:10 UTC — RELEASE = ens_gen2
- Decision logged 02:56 (docs/RELEASE_DECISION.md): mean(GEN2 U-Net w16, ens_u2ft_tcn_lgbm), thr 0.8580; rollback gen2="off".
- GEN2 seed2 replicates: held-out41 +0.018; test/v10/v11 +0.049 CI excl 0. GEN2W24/GEN2TCN trios and 4-way combo rejected (< GEN2 blend).
- Final REAL_EXPERT_REPORT.md (prefix final, 95 alarms, 16 significant zones, 6.4 min). pytest 286 passed.
- Backups: models_backup/{s3_gen2w24,s3_gen2tcn,s3_gen2_seed2}; /workspace/backup_weights_20261016.tar (261 MB).

### AUTO 03:12 interim
- последние события:
  - 02:34 COMBO_DONE members=gen2w24@step0,gen2tcn@step0 init_max_abs=0 formula=(gen2+winner+w24+tcn)/4 n_boot=1000 seed=20261016 c_val45_iou=0.392934 c_val45=d=+0.006222 ci=-0.016676,+0.044839 z=1 c_hold41_iou=0.381847 c_ho
  - 02:51 GEN2_SEED2_DONE seed=20261018 step=4507 proxy_v10=0.411677 mean2_41=0.396706 winner41=0.378442 d=+0.018264 CI=[-0.015689,+0.057203] covers0=yes mean3_41=0.386109 d=+0.007667 CI=[-0.043221,+0.056964] covers0=yes nes
  - 02:56 RELEASE_DECISION ens_gen2 (prev ens_u2ft_tcn_lgbm) reason: 2 seeds held-out41 +0.018 each, test/v10/v11 +0.05 CI excl 0, nested k5 +0.05; val CI covers 0 (documented deviation); cards 7/15 vs 8/15, false 0/5 both; 
  - 03:03 EXPERT_DONE release=ens_gen2 prefix=final static=off n_files=6 n_zones=95 n_alarm=95 n_shown=16 sec=384.3 json=artifacts/eval/real_final.json csv=artifacts/eval/real_final.csv fig=docs/figures/real_final_<file>.png
  - 03:07 SUPERVISOR(main): FINAL report ens_gen2 done, pytest 286 green, pushing
  - 03:09 SUPERVISOR(main): report window: do NOT regenerate REAL_EXPERT_REPORT.md (final ens_gen2 is locked; guard_report.sh restores). Release helper dts.infer.release_analyzer(); verify_release_gen2 IoU 0.625080 vs logged
- активные прогоны:
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 03:43 interim
- последние события:
  - 03:09 SUPERVISOR(main): report window: do NOT regenerate REAL_EXPERT_REPORT.md (final ens_gen2 is locked; guard_report.sh restores). Release helper dts.infer.release_analyzer(); verify_release_gen2 IoU 0.625080 vs logged
  - 03:12 AUTOSTATE pushed 985b14d
  - 03:37 SUPERVISOR: resumed as sole supervisor (watchdog 03:36). RELEASE ens_gen2 adopted 02:56; REAL_EXPERT_REPORT locked 03:09; AUTOSTATE 03:12. Plan: relaunch dead trains if any; neural synth gains to ~04:45; wrap 04:55
  - 03:38 SUPERVISOR: QUEUE refreshed. No dead trains (all DONE). Launching GEN2_SOUP eval + tasking ae=GEN3 (cold/weak/splice gen). Report stays locked; release ens_gen2 unless NEW beats under nested/CI.
  - 03:39 RESUMABLE launch s3_gen2_soup pid 1537696 : python -u scripts/s3_gen2_soup_eval.py
  - 03:39 SUPERVISOR: launched s3_gen2_soup eval + tasked ae=GEN3 (cold/weak/splice). CPU free for both.
- активные прогоны:
  -       03:45 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_soup_eval.py >> artifacts/ckpt/s3_gen2_soup/lo
  -       03:45 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_soup_eval.py
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 03:59 interim
- последние события:
  - 03:37 SUPERVISOR: resumed as sole supervisor (watchdog 03:36). RELEASE ens_gen2 adopted 02:56; REAL_EXPERT_REPORT locked 03:09; AUTOSTATE 03:12. Plan: relaunch dead trains if any; neural synth gains to ~04:45; wrap 04:55
  - 03:38 SUPERVISOR: QUEUE refreshed. No dead trains (all DONE). Launching GEN2_SOUP eval + tasking ae=GEN3 (cold/weak/splice gen). Report stays locked; release ens_gen2 unless NEW beats under nested/CI.
  - 03:39 RESUMABLE launch s3_gen2_soup pid 1537696 : python -u scripts/s3_gen2_soup_eval.py
  - 03:39 SUPERVISOR: launched s3_gen2_soup eval + tasked ae=GEN3 (cold/weak/splice). CPU free for both.
  - 03:45 RESUMABLE launch s3_gen3 pid 1540174 : /workspace/repo/run/.venv/bin/python -u scripts/s3_gen3_train.py
  - 03:52 SUPERVISOR: GEN2_SOUP val41 interim soup=0.3972 gen2=0.3965 d=+0.0007 CI covers0 nested5_soup=+0.0500 nested5_gen2=+0.0505 — NOT a beat; finishing test/v10/v11 for completeness. GEN3 train alive step~325 proxy_in
- активные прогоны:
  -       18:45 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_soup_eval.py >> artifacts/ckpt/s3_gen2_soup/lo
  -       18:45 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen2_soup_eval.py
  -       12:53 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen3_train.py --resume auto >> artifacts/ckpt/s3_ge
  -       12:53 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen3_train.py --resume auto
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 04:16 interim
- последние события:
  - 03:39 SUPERVISOR: launched s3_gen2_soup eval + tasked ae=GEN3 (cold/weak/splice). CPU free for both.
  - 03:45 RESUMABLE launch s3_gen3 pid 1540174 : /workspace/repo/run/.venv/bin/python -u scripts/s3_gen3_train.py
  - 03:52 SUPERVISOR: GEN2_SOUP val41 interim soup=0.3972 gen2=0.3965 d=+0.0007 CI covers0 nested5_soup=+0.0500 nested5_gen2=+0.0505 — NOT a beat; finishing test/v10/v11 for completeness. GEN3 train alive step~325 proxy_in
  - 03:59 AUTOSTATE pushed 56c4cb9
  - 04:03 GEN2_SOUP_DONE val41_soup=0.397220 val41_gen2=0.396502 d_soup_gen2=+0.000719 CI=[-0.0023,+0.0047] covers0=true nested5_soup=+0.0500 nested5_gen2=+0.0505 test_soup=0.4557 test_gen2=0.4576 beats=0 real_train=0
  - 04:06 SUPERVISOR: GEN2_SOUP rejected (hold41 +0.0007 CI covers0; test/v10/v11 slightly below ens_gen2). Watching GEN3 train (proxy mid-run). Wrap plan 04:55.
- активные прогоны:
  -       27:54 bash -c OMP_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen3_train.py --resume auto >> artifacts/ckpt/s3_ge
  -       27:54 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen3_train.py --resume auto
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 04:36 interim
- последние события:
  - 03:45 RESUMABLE launch s3_gen3 pid 1540174 : /workspace/repo/run/.venv/bin/python -u scripts/s3_gen3_train.py
  - 03:52 SUPERVISOR: GEN2_SOUP val41 interim soup=0.3972 gen2=0.3965 d=+0.0007 CI covers0 nested5_soup=+0.0500 nested5_gen2=+0.0505 — NOT a beat; finishing test/v10/v11 for completeness. GEN3 train alive step~325 proxy_in
  - 03:59 AUTOSTATE pushed 56c4cb9
  - 04:03 GEN2_SOUP_DONE val41_soup=0.397220 val41_gen2=0.396502 d_soup_gen2=+0.000719 CI=[-0.0023,+0.0047] covers0=true nested5_soup=+0.0500 nested5_gen2=+0.0505 test_soup=0.4557 test_gen2=0.4576 beats=0 real_train=0
  - 04:06 SUPERVISOR: GEN2_SOUP rejected (hold41 +0.0007 CI covers0; test/v10/v11 slightly below ens_gen2). Watching GEN3 train (proxy mid-run). Wrap plan 04:55.
  - 04:32 SUPERVISOR: resumed as sole supervisor (watchdog 04:31). RELEASE ens_gen2 adopted 02:56; REAL_EXPERT_REPORT locked; GEN2_SOUP rejected 04:03. GEN3 eval LIVE pid 1555169 since 04:29 (+watch). Plan: finish GEN3 vs en
- активные прогоны:
  -       06:54 /workspace/repo/run/.venv/bin/python -u scripts/s3_gen3_eval.py
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 04:47 wrap
- последние события:
  - 04:03 GEN2_SOUP_DONE beats=0
  - 04:06 SUPERVISOR: GEN2_SOUP rejected; watching GEN3
  - 04:32 SUPERVISOR: resumed (watchdog 04:31); GEN3 eval LIVE
  - 04:44 GEN3_DONE selected=mean_w hold41 d=+0.0002 CI covers0 nested_k5=-0.0018 beats=0
  - 04:44 SUPERVISOR: GEN3 rejected; RELEASE stays ens_gen2; report locked; wrap start
  - 04:47 SUPERVISOR WRAP DONE: GEN3 reject; pytest 293; verify_release_gen2 IoU 0.625080; Analyzer smoke OK; arch_specs+SCHEMA_OUTBOX GEN3; sync_push 6019f6f ml_grok_bot; quiet to 05:15
- активные прогоны:
- дальше: quiet until 05:15; no report regen; no release flip

### AUTO 04:51 interim
- последние события:
  - 04:06 SUPERVISOR: GEN2_SOUP rejected (hold41 +0.0007 CI covers0; test/v10/v11 slightly below ens_gen2). Watching GEN3 train (proxy mid-run). Wrap plan 04:55.
  - 04:32 SUPERVISOR: resumed as sole supervisor (watchdog 04:31). RELEASE ens_gen2 adopted 02:56; REAL_EXPERT_REPORT locked; GEN2_SOUP rejected 04:03. GEN3 eval LIVE pid 1555169 since 04:29 (+watch). Plan: finish GEN3 vs en
  - 04:36 AUTOSTATE pushed 3c548ca
  - 04:44 GEN3_DONE selected=mean_w val41=0.396729 ens_gen2=0.396502 d=+0.000228 CI=[-0.0075,+0.0085] covers0=yes nested_k5=-0.0018 nested_k2=+0.0025 test=0.4600/0.4576 v10=0.4597/0.4576 v11=0.4597/0.4575 beats=0 real_train=
  - 04:44 SUPERVISOR: GEN3 rejected (hold41 mean_w +0.0002 CI covers0; nested_k5 -0.0018; test/v10/v11 CI covers0; gen3 alone 0.142). RELEASE stays ens_gen2. Report locked. Starting wrap: arch_specs+SCHEMA_OUTBOX, pytest, An
  - 04:47 SUPERVISOR: WRAP DONE. GEN3 rejected (mean_w hold41 +0.0002 CI covers0 nested_k5 -0.0018; gen3 alone 0.142). RELEASE=ens_gen2 unchanged. REAL_EXPERT_REPORT locked (mtime 03:07). pytest 293; verify_release_gen2 IoU 
- активные прогоны:
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 05:06 interim
- последние события:
  - 04:06 SUPERVISOR: GEN2_SOUP rejected (hold41 +0.0007 CI covers0; test/v10/v11 slightly below ens_gen2). Watching GEN3 train (proxy mid-run). Wrap plan 04:55.
  - 04:32 SUPERVISOR: resumed as sole supervisor (watchdog 04:31). RELEASE ens_gen2 adopted 02:56; REAL_EXPERT_REPORT locked; GEN2_SOUP rejected 04:03. GEN3 eval LIVE pid 1555169 since 04:29 (+watch). Plan: finish GEN3 vs en
  - 04:36 AUTOSTATE pushed 3c548ca
  - 04:44 GEN3_DONE selected=mean_w val41=0.396729 ens_gen2=0.396502 d=+0.000228 CI=[-0.0075,+0.0085] covers0=yes nested_k5=-0.0018 nested_k2=+0.0025 test=0.4600/0.4576 v10=0.4597/0.4576 v11=0.4597/0.4575 beats=0 real_train=
  - 04:44 SUPERVISOR: GEN3 rejected (hold41 mean_w +0.0002 CI covers0; nested_k5 -0.0018; test/v10/v11 CI covers0; gen3 alone 0.142). RELEASE stays ens_gen2. Report locked. Starting wrap: arch_specs+SCHEMA_OUTBOX, pytest, An
  - 04:47 SUPERVISOR: WRAP DONE. GEN3 rejected (mean_w hold41 +0.0002 CI covers0 nested_k5 -0.0018; gen3 alone 0.142). RELEASE=ens_gen2 unchanged. REAL_EXPERT_REPORT locked (mtime 03:07). pytest 293; verify_release_gen2 IoU 
- активные прогоны:
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 05:21 interim
- последние события:
  - 04:32 SUPERVISOR: resumed as sole supervisor (watchdog 04:31). RELEASE ens_gen2 adopted 02:56; REAL_EXPERT_REPORT locked; GEN2_SOUP rejected 04:03. GEN3 eval LIVE pid 1555169 since 04:29 (+watch). Plan: finish GEN3 vs en
  - 04:36 AUTOSTATE pushed 3c548ca
  - 04:44 GEN3_DONE selected=mean_w val41=0.396729 ens_gen2=0.396502 d=+0.000228 CI=[-0.0075,+0.0085] covers0=yes nested_k5=-0.0018 nested_k2=+0.0025 test=0.4600/0.4576 v10=0.4597/0.4576 v11=0.4597/0.4575 beats=0 real_train=
  - 04:44 SUPERVISOR: GEN3 rejected (hold41 mean_w +0.0002 CI covers0; nested_k5 -0.0018; test/v10/v11 CI covers0; gen3 alone 0.142). RELEASE stays ens_gen2. Report locked. Starting wrap: arch_specs+SCHEMA_OUTBOX, pytest, An
  - 04:47 SUPERVISOR: WRAP DONE. GEN3 rejected (mean_w hold41 +0.0002 CI covers0 nested_k5 -0.0018; gen3 alone 0.142). RELEASE=ens_gen2 unchanged. REAL_EXPERT_REPORT locked (mtime 03:07). pytest 293; verify_release_gen2 IoU 
  - 05:06 AUTOSTATE pushed d1eb5d4
- активные прогоны:
- дальше: см. /workspace/grok_logs/QUEUE.md

### AUTO 05:37 interim
- последние события:
  - 04:32 SUPERVISOR: resumed as sole supervisor (watchdog 04:31). RELEASE ens_gen2 adopted 02:56; REAL_EXPERT_REPORT locked; GEN2_SOUP rejected 04:03. GEN3 eval LIVE pid 1555169 since 04:29 (+watch). Plan: finish GEN3 vs en
  - 04:36 AUTOSTATE pushed 3c548ca
  - 04:44 GEN3_DONE selected=mean_w val41=0.396729 ens_gen2=0.396502 d=+0.000228 CI=[-0.0075,+0.0085] covers0=yes nested_k5=-0.0018 nested_k2=+0.0025 test=0.4600/0.4576 v10=0.4597/0.4576 v11=0.4597/0.4575 beats=0 real_train=
  - 04:44 SUPERVISOR: GEN3 rejected (hold41 mean_w +0.0002 CI covers0; nested_k5 -0.0018; test/v10/v11 CI covers0; gen3 alone 0.142). RELEASE stays ens_gen2. Report locked. Starting wrap: arch_specs+SCHEMA_OUTBOX, pytest, An
  - 04:47 SUPERVISOR: WRAP DONE. GEN3 rejected (mean_w hold41 +0.0002 CI covers0 nested_k5 -0.0018; gen3 alone 0.142). RELEASE=ens_gen2 unchanged. REAL_EXPERT_REPORT locked (mtime 03:07). pytest 293; verify_release_gen2 IoU 
  - 05:06 AUTOSTATE pushed d1eb5d4
- активные прогоны:
- дальше: см. /workspace/grok_logs/QUEUE.md
