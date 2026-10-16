# realnet

Конвейер поверх предобученного `unet_v810ft`: гармоническая базовая линия, AdaBN по фону, маскированная голова (слепой остаток) и смесь трёх скоров. Код: `experiments/realnet/realnet.py`, `experiments/realnet/run_realnet.py`, `experiments/realnet/run_realnet_bc.py`, `experiments/realnet/ablate_realnet.py`, описание `experiments/realnet/REAL_PIPELINE.md`. Реальные `1–6.csv` в обучение победителя не входили; у самого realnet AdaBN и маска смотрели фон синтетики, не метки CSV.

После чистки модуль уезжает в `experiments/realnet/`. В дефолтный `dts/infer.py` он не подключается.

## Цифры

Источник абляции: `artifacts/eval/realnet_ablation.json`, v8 val, FAR \(10^{-3}\), 45 файлов, чекпоинт `artifacts/stage1/unet_v810ft.pt`.

| строка | IoU | F1 |
| --- | ---: | ---: |
| a_unet, каналы кэша | 0.3748 | 0.5452 |
| b_adabn | 0.3041 | 0.4663 |
| c_baseline_input, замена входа | 0.0912 | 0.1672 |
| e_shipped_style (AdaBN + замена входа + маска) | 0.2820 | 0.4399 |

Первый прогон (веса 0.75 / 0 / 0.25) зафиксирован в `docs/REAL_PIPELINE.md`: val IoU 0.104, test v8 IoU 0.138. `STATE.md` пишет тот же test 0.138.

Выбранный по val вариант A — это исходный U-Net, веса seg/mask/wide = 1/0/0, AdaBN выключен (`artifacts/eval/realnet_synth.json`):

| сплит | IoU | F1 |
| --- | ---: | ---: |
| v8 val | 0.3750 | 0.5455 |
| v8+v10 val pooled | 0.3769 | 0.5475 |
| v8 test | 0.3724 | 0.5427 |
| v10 test | 0.3370 | 0.5041 |

Победитель в том же файле: v8 val 0.3867, v8 test 0.4053, v10 test 0.4039.

Варианты B и C (`artifacts/eval/realnet_bc_synth.json`), веса 0.5/0.25/0.25. B: порог с pooled val. C: квантиль каждого файла.

| вариант | v8 val IoU | v8 test IoU | v10 test IoU |
| --- | ---: | ---: | ---: |
| B | 0.2716 | 0.2383 | 0.2362 |
| C | 0.0460 | 0.0446 | 0.0457 |

Карточки, порог ими не выбирался:

| вариант | yes | no | источник |
| --- | --- | --- | --- |
| A | 7/15 | 0/5 | `realnet_real.json`, `presence_alarm` |
| B | 5/15 | 0/5 | `realnet_bc_real.json` |
| C | 3/15 | 0/5 | `realnet_bc_real.json` |

У победителя на тех же карточках 8/15 и 0/5 (`docs/REAL_PIPELINE.md`, `artifacts/eval/real_v8.json`).

## Почему отклонён

Новые блоки хуже исходной головы: AdaBN роняет val IoU с 0.375 до 0.304, подмена входа остатком базовой линии — до 0.091. То, что остаётся после абляции, совпадает с уже входящим в ансамбль `unet_v810ft` и не добирает val IoU победителя (0.3769 против 0.3867). B и C ниже и по синтетике, и по карточкам. В релиз `ens_u2ft_tcn_lgbm` realnet не входит.
