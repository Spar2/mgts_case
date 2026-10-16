# Research SSL benchmark

Рабочая пара не сменена: `median_x3 + spatial_z_halo at |Z| >= 3.5`

Денойзер — «кандидат», только если MAD на 12×80 ниже 0.233 °C и на 48×160 ниже 0.239 °C, удержание спайка на обоих полях не меньше 0.92, контраст на `5.csv` не ниже `median_x3` больше чем на 5%, hot/cold IoU при `|Z| >= 3.5` не хуже `median_x3`, и `dead_tail_alarms = 0`. Иначе «не принят».

Детектор — «кандидат», только если IoU на `local_scene` и `half_cable_shift` не ниже `spatial_z_halo`, а `alarm_fraction_inside` на полукабельном сдвиге не выше. Иначе «не принят».

Устройство прогона: `cpu`.

## Денойз

| метод | статус | MAD 12×80 | MAD 48×160 | keep 12×80 | keep 48×160 | контраст 5.csv | dead_tail |
|---|---|---:|---:|---:|---:|---:|---:|
| `median_x3` | baseline | 0.233063 | 0.238562 | 0.952519 | 0.923883 | 3.0464 | 0 |
| `n2n_tsr` | не принят | 0.101579 | 0.101536 | 0.959668 | 0.933342 | 3.192979 | 0 |
| `blindspot_x` | не принят | 0.105311 | 0.072815 | 0.353425 | 0.329987 | 2.63815 | 0 |
| `zsn2n` | не принят | 0.129064 | 0.130577 | 1.011359 | 0.981067 | 3.108096 | 0 |

## Детекторы

| метод | статус | hot IoU | cold IoU | half IoU | alarm_fraction_inside |
|---|---|---:|---:|---:|---:|
| `spatial_z_halo` | baseline | 1.0 | 1.0 | 0.111111 | 0.125 |
| `rpca` | не принят | 0.135762 | 0.060241 | 0.026396 | 0.027083 |
| `matrix_profile` | не принят | 0.054138 | 0.047393 | 0.042 | 0.04375 |
| `chronos_bolt` | не принят | 0.013423 | 0.016529 | 0.0 | 0.0 |
| `patchcore` | не принят | 0.191406 | 0.154948 | 0.166667 | 0.2 |

Chronos-Bolt: запрошены квантили 0.05/0.95, в счёт пошли 0.1 и 0.9.

## Очередь разметки

очередь разметки без экспертных меток; precision/recall против spatial Z не считается

## Порог под денойзер и гибридное уточнение (после d288333)

Рабочая пара не сменена: `median_x3` и `spatial_z` при `|Z| >= 3.5`.
Порог ниже подобран под тот же FAR и в `analyze_field` не записан.
Правило порога: grid point closest to 3.5 among those with FAR closest to the target.
Устройство прогона: `cuda`.

| метод | шкала | порог | hot IoU | cold IoU | FAR holdout | weak hot IoU | статус | условие |
|---|---|---:|---:|---:|---:|---:|---|---|
| `median_x3` | `mad_profile` | 3.5 | 0.584657 | 0.670422 | 0.004786 | 0.473795 | baseline | — |
| `median_x3` | `mad_residual_global` | 3.5 | 0.584657 | 0.670422 | 0.004786 | 0.473795 | не принят | нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall |
| `median_x3` | `mad_raw_residual` | 3.5 | 0.584657 | 0.670422 | 0.004786 | 0.473795 | не принят | нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall |
| `n2n_tsr` | `mad_profile` | 4.85 | 0.50178 | 0.54579 | 0.004741 | 0.378822 | не принят | held-out hot IoU ниже median_x3 на 3.5 больше чем на 0.01, held-out cold IoU ниже median_x3 на 3.5 больше чем на 0.01, нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall, нестабилен на реальных данных |
| `n2n_tsr` | `mad_residual_global` | 4.85 | 0.501778 | 0.54579 | 0.004741 | 0.378822 | не принят | held-out hot IoU ниже median_x3 на 3.5 больше чем на 0.01, held-out cold IoU ниже median_x3 на 3.5 больше чем на 0.01, нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall, нестабилен на реальных данных |
| `n2n_tsr` | `mad_raw_residual` | 4.85 | 0.501778 | 0.54579 | 0.004741 | 0.378822 | не принят | held-out hot IoU ниже median_x3 на 3.5 больше чем на 0.01, held-out cold IoU ниже median_x3 на 3.5 больше чем на 0.01, нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall, нестабилен на реальных данных |
| `zsn2n` | `mad_profile` | 3.6 | 0.581214 | 0.67549 | 0.004789 | 0.468159 | не принят | нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall |
| `zsn2n` | `mad_residual_global` | 3.6 | 0.581214 | 0.67549 | 0.004789 | 0.468159 | не принят | нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall |
| `zsn2n` | `mad_raw_residual` | 3.6 | 0.581214 | 0.67549 | 0.004789 | 0.468159 | не принят | нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall |

Кривые: [iou_vs_threshold_hot.png](images/ssl/iou_vs_threshold_hot.png), [iou_vs_threshold_cold.png](images/ssl/iou_vs_threshold_cold.png), [far_vs_threshold.png](images/ssl/far_vs_threshold.png).

### Реальные файлы

| метод | шкала | файл | зоны | зоны median_x3 | доля тревоги | покрытие | пересечение |
|---|---|---|---:|---:|---:|---:|---:|
| `median_x3` | `mad_profile` | 1.csv | 23 | 23 | 0.041531 | 1 | 1 |
| `median_x3` | `mad_profile` | 2.csv | 19 | 19 | 0.063441 | 1 | 1 |
| `median_x3` | `mad_profile` | 3.csv | 5 | 5 | 0.005593 | 1 | 1 |
| `median_x3` | `mad_profile` | 4.csv | 8 | 8 | 0.027899 | 1 | 1 |
| `median_x3` | `mad_profile` | 5.csv | 134 | 134 | 0.019562 | 1 | 1 |
| `median_x3` | `mad_profile` | 6.csv | 43 | 43 | 0.076512 | 1 | 1 |
| `median_x3` | `mad_residual_global` | 1.csv | 22 | 23 | 0.041521 | 1 | 0.913043 |
| `median_x3` | `mad_residual_global` | 2.csv | 13 | 19 | 0.063067 | 1 | 0.947368 |
| `median_x3` | `mad_residual_global` | 3.csv | 13 | 5 | 0.007507 | 1 | 0.8 |
| `median_x3` | `mad_residual_global` | 4.csv | 10 | 8 | 0.031575 | 1 | 0.875 |
| `median_x3` | `mad_residual_global` | 5.csv | 146 | 134 | 0.019438 | 1 | 0.902985 |
| `median_x3` | `mad_residual_global` | 6.csv | 59 | 43 | 0.077142 | 1 | 0.906977 |
| `median_x3` | `mad_raw_residual` | 1.csv | 22 | 23 | 0.040857 | 1 | 0.913043 |
| `median_x3` | `mad_raw_residual` | 2.csv | 18 | 19 | 0.057746 | 1 | 0.947368 |
| `median_x3` | `mad_raw_residual` | 3.csv | 8 | 5 | 0.006208 | 1 | 0.8 |
| `median_x3` | `mad_raw_residual` | 4.csv | 14 | 8 | 0.028884 | 1 | 0.875 |
| `median_x3` | `mad_raw_residual` | 5.csv | 129 | 134 | 0.006336 | 1 | 0.41791 |
| `median_x3` | `mad_raw_residual` | 6.csv | 65 | 43 | 0.045085 | 1 | 0.674419 |
| `n2n_tsr` | `mad_profile` | 1.csv | 19 | 23 | 0.01588 | 1 | 0.521739 |
| `n2n_tsr` | `mad_profile` | 2.csv | 6 | 19 | 0.028284 | 1 | 0.315789 |
| `n2n_tsr` | `mad_profile` | 3.csv | 0 | 5 | 0 | 1 | 0 |
| `n2n_tsr` | `mad_profile` | 4.csv | 4 | 8 | 0.014544 | 1 | 0.375 |
| `n2n_tsr` | `mad_profile` | 5.csv | 142 | 134 | 0.030178 | 1 | 0.992537 |
| `n2n_tsr` | `mad_profile` | 6.csv | 71 | 43 | 0.055664 | 1 | 0.72093 |
| `n2n_tsr` | `mad_residual_global` | 1.csv | 14 | 23 | 0.015514 | 1 | 0.434783 |
| `n2n_tsr` | `mad_residual_global` | 2.csv | 7 | 19 | 0.026279 | 1 | 0.263158 |
| `n2n_tsr` | `mad_residual_global` | 3.csv | 1 | 5 | 7.3e-05 | 1 | 0.2 |
| `n2n_tsr` | `mad_residual_global` | 4.csv | 4 | 8 | 0.017248 | 1 | 0.375 |
| `n2n_tsr` | `mad_residual_global` | 5.csv | 126 | 134 | 0.029986 | 1 | 0.992537 |
| `n2n_tsr` | `mad_residual_global` | 6.csv | 55 | 43 | 0.060209 | 1 | 0.674419 |
| `n2n_tsr` | `mad_raw_residual` | 1.csv | 14 | 23 | 0.016232 | 1 | 0.434783 |
| `n2n_tsr` | `mad_raw_residual` | 2.csv | 6 | 19 | 0.020676 | 1 | 0.263158 |
| `n2n_tsr` | `mad_raw_residual` | 3.csv | 0 | 5 | 0 | 1 | 0 |
| `n2n_tsr` | `mad_raw_residual` | 4.csv | 3 | 8 | 0.015731 | 1 | 0.25 |
| `n2n_tsr` | `mad_raw_residual` | 5.csv | 32 | 134 | 0.001042 | 1 | 0.186567 |
| `n2n_tsr` | `mad_raw_residual` | 6.csv | 14 | 43 | 0.015126 | 1 | 0.302326 |
| `zsn2n` | `mad_profile` | 1.csv | 23 | 23 | 0.03757 | 1 | 1 |
| `zsn2n` | `mad_profile` | 2.csv | 16 | 19 | 0.055395 | 1 | 0.947368 |
| `zsn2n` | `mad_profile` | 3.csv | 2 | 5 | 0.003347 | 1 | 0.4 |
| `zsn2n` | `mad_profile` | 4.csv | 9 | 8 | 0.025055 | 1 | 1 |
| `zsn2n` | `mad_profile` | 5.csv | 83 | 134 | 0.0084 | 1 | 0.470149 |
| `zsn2n` | `mad_profile` | 6.csv | 40 | 43 | 0.046854 | 1 | 0.790698 |
| `zsn2n` | `mad_residual_global` | 1.csv | 21 | 23 | 0.037811 | 1 | 0.869565 |
| `zsn2n` | `mad_residual_global` | 2.csv | 15 | 19 | 0.055471 | 1 | 0.947368 |
| `zsn2n` | `mad_residual_global` | 3.csv | 12 | 5 | 0.005343 | 1 | 0.8 |
| `zsn2n` | `mad_residual_global` | 4.csv | 11 | 8 | 0.027735 | 1 | 0.875 |
| `zsn2n` | `mad_residual_global` | 5.csv | 85 | 134 | 0.008262 | 1 | 0.485075 |
| `zsn2n` | `mad_residual_global` | 6.csv | 47 | 43 | 0.049111 | 1 | 0.790698 |
| `zsn2n` | `mad_raw_residual` | 1.csv | 22 | 23 | 0.038396 | 1 | 0.913043 |
| `zsn2n` | `mad_raw_residual` | 2.csv | 12 | 19 | 0.056984 | 1 | 0.947368 |
| `zsn2n` | `mad_raw_residual` | 3.csv | 7 | 5 | 0.006329 | 1 | 0.8 |
| `zsn2n` | `mad_raw_residual` | 4.csv | 11 | 8 | 0.027925 | 1 | 0.875 |
| `zsn2n` | `mad_raw_residual` | 5.csv | 85 | 134 | 0.006601 | 1 | 0.425373 |
| `zsn2n` | `mad_raw_residual` | 6.csv | 48 | 43 | 0.047985 | 1 | 0.744186 |

Нестабилен на реальных данных: `n2n_tsr`/`mad_profile`, `n2n_tsr`/`mad_residual_global`, `n2n_tsr`/`mad_raw_residual`.

### Гибридное уточнение

| режим | hot IoU | cold IoU | ошибка пика | ошибка x при ramp=0 | статус | условие |
|---|---:|---:|---:|---:|---|---|
| `half_max` | 0.627584 | 0.738336 | 0.162475 | 0 | не принят | held-out hot IoU ниже median_x3 больше чем на 0.005, held-out cold IoU ниже median_x3 больше чем на 0.005, медианная ошибка пикового контраста не ниже на 20%, группа амплитуды хуже больше чем на 5%, witness pass count ниже median_x3 |
| `region_grow` | 0.571608 | 0.618732 | 0.162475 | 2 | не принят | held-out hot IoU ниже median_x3 больше чем на 0.005, held-out cold IoU ниже median_x3 больше чем на 0.005, IoU на ramp=0 упал, медианная ошибка пикового контраста не ниже на 20%, группа амплитуды хуже больше чем на 5%, медианная ошибка границы по x на ramp=0 хуже median_x3, медианная ошибка границы по x на ramp=0 больше одного шага сетки, ошибка границы по времени хуже median_x3 |

Ошибки пика и границ по амплитуде, ширине и ramp лежат в `docs/evidence/hybrid_refine.json`. Картинки: `docs/images/ssl/refine_*.png`.

Решение по правилу отчёта: `median_x3` / `mad_residual_global` — не принят: нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall; `median_x3` / `mad_raw_residual` — не принят: нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall; `n2n_tsr` / `mad_profile` — не принят: held-out hot IoU ниже median_x3 на 3.5 больше чем на 0.01; `n2n_tsr` / `mad_residual_global` — не принят: held-out hot IoU ниже median_x3 на 3.5 больше чем на 0.01; `n2n_tsr` / `mad_raw_residual` — не принят: held-out hot IoU ниже median_x3 на 3.5 больше чем на 0.01; `zsn2n` / `mad_profile` — не принят: нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall; `zsn2n` / `mad_residual_global` — не принят: нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall; `zsn2n` / `mad_raw_residual` — не принят: нет прироста на слабых событиях (амплитуда ≤ 2 °C) на 0.05 IoU или recall; гибрид `half_max` — не принят: held-out hot IoU ниже median_x3 больше чем на 0.005; гибрид `region_grow` — не принят: held-out hot IoU ниже median_x3 больше чем на 0.005.
