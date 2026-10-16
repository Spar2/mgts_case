# Пояснение к операционному разбору DTS

## Задача

Шесть записей температуры вдоль кабеля — это таблицы время на метры. Нужно найти локальные нагревы и остывания и назвать форму уже найденного участка. Операционный поиск и имя формы — разные шаги. Поиск остаётся парой `median_x3` и `spatial_z_halo` при `|Z| >= 3.5`. Имя формы — `shape_hypothesis`. Это измерение правил, не экспертная метка. На зонах `expert_confirmed` false.

Viewer ниже показывает этот operational путь на встроенном примере: heatmap температуры после чистки, зоны, гипотезу и evidence с видом `spatial` и видом `temporal`.

## Методы

Чистка — `median_x3`. Детектор — `spatial_z_halo`: клетка сравнивается с кольцом соседей, внутренний радиус 8 м, внешний 40 м, бин 2 м, пол 0.30 °C. Зона остаётся, если ширина не меньше 3 м, профилей не меньше 6, клеток тревоги не меньше 24, и разрыв по времени не длиннее 300 с. Статус `Anomaly` ставится, когда такая зона есть. `trained_classifier_operational` false.

У каждой зоны есть `contrast` в °C (нагрев положительный, остывание отрицательный), `width_m`, `observed_duration_s`, `support` и `periodicity_CV`. `periodicity_CV` равен null, если зона не входит в повторяющийся кластер на том же месте. Temporal evidence читает причинный остаток и к той же зоне приписан. Число зон он не меняет.

Отрицательные результаты, уже посчитанные в репозитории:

- hybrid refine не принят. Уточнения `half_max` и `region_grow` дали held-out hot IoU ниже `median_x3` больше чем на 0.005. В операционную пару они не входят. Отчёт `docs/evidence/hybrid_refine.json`.
- temporal_innovation не ставит anomaly_status. Это причинный остаток по прошлым профилям той же колонки. На целом сдвиге кабеля spatial halo молчит, а temporal residual ненулевой. Статус при этом `Normal`.
- Research CNN остаётся `serve=research`, не operational. На слабых метках он не заменяет `spatial_z_halo`. `POST /detect/field` эти веса не читает.

`POST /detect/field` по-прежнему отдаёт зоны без плотного поля. Heatmap рисует отдельный viewer.

## Найденные аномалии

Каталог `docs/evidence/zone_catalog.json` — это зоны `analyze_field` на файлах 1–6. Число событий файла равно `n_zones` этого файла. Гипотезы в строках — `shape_hypothesis`, не экспертная точность. `expert_confirmed` false.

| Файл | Зоны |
|---|---|
| 1 | 23 |
| 2 | 19 |
| 3 | 5 |
| 4 | 8 |
| 5 | 134 |
| 6 | 43 |

Все шесть файлов в каталоге имеют статус `Anomaly`, потому что зоны есть. Всего 232 зоны.

На файле 1 карточки `review-00034` и `review-00098` попадают в горячие зоны. `review-00187` (несколько пиков около 230–270 м) и `review-00223` (остывание около 363–403 м) клеток с `|Z| >= 3.5` не имеют, поэтому в этот каталог как зоны не входят. Порог из-за этого не опускался.

## 15 экспертных карточек

Сеть research, `serve=research`. Это не операционный детектор. `analyze_field` и `POST /detect/field` эти веса не читают. Совпадение класса — восстановление окна, не точность детектора. Промах остаётся промахом. Классы ниже — поля `expert_class` и `network_class` из `docs/evidence/research_cnn_report.json`. Оба cooling и все четыре `presence=no` в этом списке. Предыдущий промер был 10/15, sha256 `fed62abe36d24bdce9a07be14677506e41606dc4491f00bd3f80462cd87f4e51`. Этот промер 9/15, sha256 `601a2b5633a986738f9df4d60a33b08a1424286571a0e558a88aa68346ae59b5`.

- `review-00034`: эксперт `Local_Heating`, сеть `Local_Heating`.
- `review-00098`: эксперт `Local_Heating`, сеть `Local_Heating`.
- `review-00187`: эксперт `Local_Heating`, сеть `Local_Heating`.
- `review-00223`: эксперт `Local_Cooling`, сеть `Local_Cooling`.
- `next-0001`: эксперт `Local_Heating`, сеть `Local_Cooling`.
- `next-0002`: эксперт `Local_Cooling`, сеть `Local_Cooling`.
- `next-0003`: эксперт `Local_Heating`, сеть `Local_Heating`.
- `next-0004`: эксперт `Local_Heating`, сеть `Local_Cooling`.
- `next-0005`: эксперт `Background`, сеть `Local_Heating`. `presence=no`.
- `next-0006`: эксперт `Background`, сеть `Local_Heating`. `presence=no`.
- `next-0007`: эксперт `Background`, сеть `Local_Cooling`. `presence=no`.
- `next-0008`: эксперт `Local_Heating`, сеть `Local_Heating`.
- `next-0009`: эксперт `Local_Heating`, сеть `Local_Heating`.
- `next-0010`: эксперт `Background`, сеть `Local_Cooling`. `presence=no`.
- `next-0011`: эксперт `Local_Heating`, сеть `Local_Heating`.

## норма vs контраст

Промер `analyze_field` в `docs/evidence/spatial_vs_expert.json`. `presence=no` остаётся `Background`. Медиана температуры `presence=no` не переписывает. Операционная пара по-прежнему `median_x3` и `spatial_z_halo` при `|Z| >= 3.5`.

- `next-0010`: `presence=no`, эксперт `Background`. Baseline `spatial_z` держит холодные зоны, contrast -3.905, -3.96, -4.295, -4.274, -4.254, -4.267 °C. Сеть на этой карточке сказала `Local_Cooling`.
- `next-0006`: заметка «в норме», эксперт `Background`. Baseline внутри прямоугольника зону не оставил. Контраст держат фланги: `Local_Heating` 17.414 °C и `Local_Heating` 16.077 °C.
- `next-0003`: эксперт `Local_Heating`. Baseline внутри прямоугольника зону не оставил. Окно остаётся `Local_Heating`.
- `next-0005`: `presence=no`, эксперт `Background`. Baseline внутри прямоугольника зону не оставил. На фланге `Local_Heating` 3.008 °C и `Local_Cooling` -2.726 °C.
- `next-0007`: `presence=no`, эксперт `Background`. Baseline внутри прямоугольника зону не оставил. На фланге `Local_Cooling` -5.93 °C и `Local_Heating` 4.783 °C.

## Ограничение halo на широком участке

`spatial_z_halo` сравнивает клетку с кольцом 8–40 м. Участок шире этого кольца может лежать внутри собственного фона и остаться ниже `|Z| >= 3.5`. Эксперт на таком участке смотрит на весь прямоугольник против кабеля дальше 40 м или против флангов, а не против кольца. Это разный масштаб. Это не операционная точность.

Контекстный канал читает медиану прямоугольника, медиану кабеля дальше 40 м, долю кольца 8–40 м внутри прямоугольника и зоны в ящике или на фланге. Канал `serve=research`. Операционная пара остаётся `median_x3` и `spatial_z_halo` при `|Z| >= 3.5`. Канал не ставит anomaly_status. В файлах ниже `expert_confirmed` false: контекстный файл карточку не подтверждает. Числа лежат в [docs/evidence/context_evidence.json](docs/evidence/context_evidence.json) и [docs/evidence/spatial_vs_expert.json](docs/evidence/spatial_vs_expert.json).

- `next-0003`: ширина 65.981 м, это шире 40 м. Эксперт `Local_Heating`, окно остаётся `Local_Heating`. В ящике зон нет. Медиана прямоугольника 32.060 °C. Медиана кабеля дальше 40 м 26.372 °C. Доля кольца внутри 0.786. Остаток против фона кольца 0.878 °C, поэтому сравнение с кольцом зону внутри не держит. Дальний левый 23.786 °C, дальний правый 26.918 °C.
- `next-0006`: `presence=no`, эксперт `Background`, заметка «в норме». Ширина 39.899 м. В ящике зон нет. Медиана прямоугольника 19.432 °C. Медиана кабеля дальше 40 м 21.068 °C. Доля кольца внутри 0.375. Фланги `Local_Heating` 17.414 °C и `Local_Heating` 16.077 °C. Остаток против фона кольца -0.714 °C: кольцо видит горячие фланги, а кабель дальше 40 м от медианы прямоугольника отличается слабее. `presence=no` остаётся `Background`.
- `next-0010`: `presence=no`, эксперт `Background`. Ширина 26.392 м. Медиана прямоугольника 14.316 °C. Медиана кабеля дальше 40 м 19.035 °C. Доля кольца внутри 0.159. В ящике контрасты -3.905, -3.96, -4.295, -4.274, -4.254, -4.267 °C. Фланги `Local_Heating` 3.5 °C и `Local_Heating` 4.231 °C. Кольцо 8–40 м захватывает эти фланги, поэтому детектор держит холод внутри. `presence=no` не становится классом по температуре и остаётся `Background`.

## Контекстная research-сеть

Вторая сеть берёт residual-кроп и три числа context evidence: медиану прямоугольника, медиану far cable и долю кольца внутри. `serve=research`, operational false. `analyze_field` и `POST /detect/field` эти веса не читают. Fit seeds — прежние 15 expert seeds и каждый ask expert seed веса 10. Gap-карточки не были fit seeds и в этот счёт не входят. Совпадение класса — восстановление окна, не операционная точность. На новых ask-карточках восстановление in-sample: эти карточки были fit seeds. Промах остаётся промахом. Классы ниже — поля `expert_class` и `network_class` из `docs/evidence/research_context_cnn_report.json`. Восстановление на новых ask seeds 63/110. На прежних 15 seeds восстановление 2/15. Рядом прежний контекстный промер 15/22 и прежний промер на этих 15 seeds 10/15. Опубликованный residual-промер 9/15. sha256 весов `0eb6c52b132b4838f85190cc03567c0a2fcbf48c8ef4710a38ea68ee07f0eea5`. `next-0003` остаётся экспертом `Local_Heating`. Неразмеченный gap-ответ не совпадение и не промах, класса у него нет: `gap-r5-x0.000-x127.000-t0-1537`, `gap-r5-x155.000-x213.000-t0-1537`, `gap-r6-x1116.100-x1129.296-t251-383`. Порог spatial_z и чистка median_x3 не менялись. Операционная пара остаётся `median_x3` и `spatial_z_halo` при `|Z| >= 3.5`.

- `review-00034`: эксперт `Local_Heating`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `review-00098`: эксперт `Local_Heating`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `review-00187`: эксперт `Local_Heating`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `review-00223`: эксперт `Local_Cooling`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `next-0001`: эксперт `Local_Heating`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `next-0002`: эксперт `Local_Cooling`, контекстная сеть `Local_Cooling`.
- `next-0003`: эксперт `Local_Heating`, контекстная сеть `Broad_Shift`. Эксперт остаётся `Local_Heating`. Промах остаётся промахом.
- `next-0004`: эксперт `Local_Heating`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `next-0005`: эксперт `Background`, контекстная сеть `Broad_Shift`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `next-0006`: эксперт `Background`, контекстная сеть `Periodic_Thermal`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `next-0007`: эксперт `Background`, контекстная сеть `Broad_Shift`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `next-0008`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `next-0009`: эксперт `Local_Heating`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `next-0010`: эксперт `Background`, контекстная сеть `Local_Cooling`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `next-0011`: эксперт `Local_Heating`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `ask-r1-x0.000-x110.000-t172-258`: эксперт `Background`, контекстная сеть `Periodic_Thermal`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r1-x1045.000-x1125.000-t431-463`: эксперт `Background`, контекстная сеть `Periodic_Thermal`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r1-x111.574-x130.761-t0-295`: эксперт `Local_Heating`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `ask-r1-x111.777-x130.761-t306-338`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r1-x1125.000-x1185.000-t399-431`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x150.000-x210.000-t0-86`: эксперт `Background`, контекстная сеть `Periodic_Thermal`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r1-x150.000-x210.000-t172-258`: эксперт `Background`, контекстная сеть `Periodic_Thermal`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r1-x150.000-x210.000-t258-344`: эксперт `Background`, контекстная сеть `Periodic_Thermal`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r1-x150.000-x210.000-t86-172`: эксперт `Background`, контекстная сеть `Periodic_Thermal`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r1-x285.000-x365.000-t399-431`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x375.000-x435.000-t431-463`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x380.000-x460.000-t399-431`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x417.259-x421.015-t0-344`: эксперт `Local_Heating`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `ask-r1-x425.787-x429.036-t0-230`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r1-x434.619-x455.330-t0-244`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x475.000-x555.000-t399-431`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x475.000-x555.000-t431-463`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x491.777-x494.822-t0-344`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r1-x525.000-x585.000-t258-344`: эксперт `Background`, контекстная сеть `Periodic_Thermal`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r1-x570.000-x650.000-t0-86`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x570.000-x650.000-t172-258`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x570.000-x650.000-t431-463`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x570.000-x650.000-t86-172`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x600.000-x660.000-t258-344`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x620.000-x760.000-t399-431`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x687.208-x691.980-t0-292`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r1-x750.000-x860.000-t431-463`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x803.858-x808.934-t0-344`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r1-x812.893-x816.142-t0-344`: эксперт `Local_Heating`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `ask-r1-x874.518-x894.112-t0-344`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r1-x880.609-x888.629-t399-463`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r1-x889.746-x894.010-t399-463`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r2-x0.000-x15.226-t372-409`: эксперт `Broad_Shift`, контекстная сеть `Broad_Shift`.
- `ask-r2-x0.000-x80.000-t0-102`: эксперт `Local_Heating`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `ask-r2-x0.000-x80.000-t102-205`: эксперт `Local_Heating`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `ask-r2-x0.000-x80.000-t425-456`: эксперт `Local_Heating`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `ask-r2-x0.000-x80.000-t456-487`: эксперт `Local_Heating`, контекстная сеть `Periodic_Thermal`. Промах остаётся промахом.
- `ask-r2-x100.241-x108.362-t425-549`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r2-x190.000-x270.000-t425-456`: эксперт `Periodic_Thermal`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r2-x190.000-x270.000-t456-487`: эксперт `Periodic_Thermal`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r2-x190.000-x270.000-t487-518`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r2-x190.000-x270.000-t518-549`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r2-x276.614-x320.263-t425-549`: эксперт `Local_Cooling`, контекстная сеть `Local_Cooling`.
- `ask-r2-x320.517-x329.907-t0-409`: эксперт `Local_Heating`, контекстная сеть `Local_Cooling`. Промах остаётся промахом.
- `ask-r2-x380.000-x460.000-t102-205`: эксперт `Background`, контекстная сеть `Broad_Shift`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r2-x380.000-x460.000-t205-307`: эксперт `Background`, контекстная сеть `Broad_Shift`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r2-x380.000-x460.000-t307-410`: эксперт `Background`, контекстная сеть `Broad_Shift`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r2-x380.000-x460.000-t425-456`: эксперт `Background`, контекстная сеть `Broad_Shift`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r2-x380.000-x460.000-t456-487`: эксперт `Background`, контекстная сеть `Broad_Shift`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r2-x380.000-x460.000-t487-518`: эксперт `Background`, контекстная сеть `Broad_Shift`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r2-x464.915-x472.528-t425-549`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r2-x468.214-x472.274-t0-409`: эксперт `Local_Heating`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r2-x495.875-x501.712-t425-549`: эксперт `Local_Cooling`, контекстная сеть `Local_Cooling`.
- `ask-r2-x99.987-x108.362-t0-410`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r3-x341.322-x386.747-t0-465`: эксперт `Broad_Shift`, контекстная сеть `Broad_Shift`.
- `ask-r3-x341.830-x386.240-t480-623`: эксперт `Broad_Shift`, контекстная сеть `Broad_Shift`.
- `ask-r3-x388.016-x399.182-t480-623`: эксперт `Local_Cooling`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r3-x406.034-x428.366-t0-465`: эксперт `Local_Cooling`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r3-x406.541-x428.366-t480-623`: эксперт `Local_Cooling`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r3-x95.000-x175.000-t0-116`: эксперт `Broad_Shift`, контекстная сеть `Broad_Shift`.
- `ask-r3-x95.000-x175.000-t116-232`: эксперт `Broad_Shift`, контекстная сеть `Broad_Shift`.
- `ask-r3-x95.000-x175.000-t551-587`: эксперт `Broad_Shift`, контекстная сеть `Broad_Shift`.
- `ask-r4-x0.000-x3.299-t0-417`: эксперт `Local_Heating`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r4-x31.214-x37.812-t0-417`: эксперт `Periodic_Thermal`, контекстная сеть `Local_Cooling`. Промах остаётся промахом.
- `ask-r4-x75.116-x78.669-t90-332`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x1021.000-x1036.000-t49-67`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x1086.000-x1100.000-t382-403`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x1108.000-x1120.000-t397-408`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x1146.000-x1150.000-t1328-1339`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x1147.000-x1151.000-t1461-1469`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x128.000-x142.000-t1215-1223`: эксперт `Local_Cooling`, контекстная сеть `Local_Cooling`.
- `ask-r5-x129.000-x146.000-t1091-1101`: эксперт `Local_Cooling`, контекстная сеть `Local_Cooling`.
- `ask-r5-x133.000-x147.000-t624-648`: эксперт `Local_Cooling`, контекстная сеть `Local_Cooling`.
- `ask-r5-x134.000-x147.000-t601-616`: эксперт `Local_Cooling`, контекстная сеть `Local_Cooling`.
- `ask-r5-x135.000-x146.000-t760-782`: эксперт `Local_Cooling`, контекстная сеть `Local_Cooling`.
- `ask-r5-x135.000-x149.000-t739-756`: эксперт `Local_Cooling`, контекстная сеть `Local_Cooling`.
- `ask-r5-x1350.000-x1354.000-t1299-1309`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x1372.000-x1381.000-t1217-1241`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x1549.000-x1566.000-t1496-1530`: эксперт `Periodic_Thermal`, контекстная сеть `Local_Heating`. Промах остаётся промахом.
- `ask-r5-x336.000-x376.000-t151-161`: эксперт `Background`, контекстная сеть `Broad_Shift`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r5-x387.000-x395.000-t126-157`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x410.000-x422.000-t100-124`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x490.000-x504.000-t192-211`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x793.000-x798.000-t1444-1460`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x886.000-x891.000-t1430-1441`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x980.000-x985.000-t1414-1431`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r5-x995.000-x999.000-t1397-1408`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r6-x1076.765-x1110.263-t753-969`: эксперт `Local_Heating`, контекстная сеть `Local_Cooling`. Промах остаётся промахом.
- `ask-r6-x1077.019-x1110.263-t0-705`: эксперт `Broad_Shift`, контекстная сеть `Local_Cooling`. Промах остаётся промахом.
- `ask-r6-x1116.100-x1127.774-t418-705`: эксперт `Local_Heating`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r6-x125.000-x235.000-t753-807`: эксперт `Background`, контекстная сеть `Broad_Shift`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r6-x125.000-x235.000-t807-861`: эксперт `Background`, контекстная сеть `Broad_Shift`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r6-x150.000-x210.000-t362-543`: эксперт `Background`, контекстная сеть `Broad_Shift`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r6-x190.000-x270.000-t915-969`: эксперт `Background`, контекстная сеть `Broad_Shift`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r6-x225.000-x285.000-t861-915`: эксперт `Periodic_Thermal`, контекстная сеть `Periodic_Thermal`.
- `ask-r6-x407.308-x410.607-t461-705`: эксперт `Local_Heating`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r6-x407.308-x410.861-t753-969`: эксперт `Local_Heating`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r6-x450.000-x510.000-t362-543`: эксперт `Local_Heating`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r6-x450.000-x510.000-t543-724`: эксперт `Local_Heating`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r6-x450.000-x510.000-t753-807`: эксперт `Local_Heating`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r6-x450.000-x510.000-t807-861`: эксперт `Local_Heating`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r6-x450.000-x510.000-t861-915`: эксперт `Local_Heating`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r6-x450.000-x510.000-t915-969`: эксперт `Local_Heating`, контекстная сеть `Broad_Shift`. Промах остаётся промахом.
- `ask-r6-x510.340-x550.340-t705-721`: эксперт `Local_Cooling`, контекстная сеть `Local_Cooling`.
- `ask-r6-x514.908-x555.512-t753-969`: эксперт `Local_Cooling`, контекстная сеть `Local_Cooling`.
- `ask-r6-x515.670-x554.497-t0-705`: эксперт `Local_Cooling`, контекстная сеть `Local_Cooling`.
- `ask-r6-x54.562-x59.383-t0-724`: эксперт `Broad_Shift`, контекстная сеть `Broad_Shift`.
- `ask-r6-x590.026-x598.400-t460-705`: эксперт `Local_Heating`, контекстная сеть `Local_Heating`.
- `ask-r6-x628.092-x631.391-t753-969`: эксперт `Background`, контекстная сеть `Local_Cooling`. `presence=no` остаётся `Background`. Промах остаётся промахом.
- `ask-r6-x641.542-x654.992-t753-969`: эксперт `Local_Cooling`, контекстная сеть `Local_Cooling`.

## Около 150 ответов ask-shortlist

Файл `docs/evidence/zakhar_ask_shortlist_answers.jsonl` содержит 150 ответов. 110 из них — expert seeds веса 10. 40 ответов с `presence=unclear` не совпадение и не промах: класса у них нет, в `docs/evidence/research_context_cnn_report.json` их нет, и `/view/windows` их не показывает.

Восстановление до fit: на прежних 15 seeds `10/15`, исторический контекстный промер `15/22`. Восстановление после fit: на прежних 15 seeds `2/15`, на ask seeds `63/110`. Эти четыре дроби — `previous_match_label`, `historical_context_match_label`, `expert_seed_match_label` и `ask_seed_match_label` того же отчёта. Восстановление новых карточек in-sample и не является операционной точностью. Промах остаётся промахом. `serve=research` и operational false не менялись. Порог spatial_z и чистка median_x3 не менялись. Операционная пара остаётся `median_x3` и `spatial_z_halo` при `|Z| >= 3.5`.

`next-0003` остаётся экспертом `Local_Heating`. Карточка `presence=no` остаётся `Background`.

Основные оставшиеся расхождения на ask seeds, эксперт → контекстная сеть: `Background` → `Broad_Shift` 11, `Local_Heating` → `Broad_Shift` 11, `Background` → `Periodic_Thermal` 7, `Local_Heating` → `Periodic_Thermal` 7. Реже: `Local_Cooling` → `Broad_Shift` 3, `Periodic_Thermal` → `Broad_Shift` 2, `Local_Heating` → `Local_Cooling` 2, `Background` → `Local_Cooling` 1, `Periodic_Thermal` → `Local_Cooling` 1, `Periodic_Thermal` → `Local_Heating` 1, `Broad_Shift` → `Local_Cooling` 1. Промах остаётся промахом. На прежних 15 совпали `next-0002` и `next-0008`.

## Расхождения и context evidence

Ниже каждая из 15 экспертных карточек, где класс residual-сети отличается от эксперта или класс контекстной сети отличается от эксперта. Числа — медиана прямоугольника, медиана far cable и доля кольца внутри из `docs/evidence/context_evidence.json`. Отсутствующая медиана far cable не заполняется медианой прямоугольника. Промах остаётся промахом. `presence=no` остаётся `Background`. Эти измерения не ставят anomaly_status и не являются операционной точностью. Операционная пара остаётся `median_x3` и `spatial_z_halo` при `|Z| >= 3.5`.

- `review-00034`: эксперт `Local_Heating`. Residual-сеть `Local_Heating`. Контекстная сеть `Periodic_Thermal`. Сети разошлись: residual сказал `Local_Heating`, контекстная сеть сказала `Periodic_Thermal`. Промах остаётся промахом. Медиана прямоугольника 29.342 °C. Медиана кабеля дальше 40 м 22.202 °C. Доля кольца внутри 0.000.
- `review-00098`: эксперт `Local_Heating`. Residual-сеть `Local_Heating`. Контекстная сеть `Periodic_Thermal`. Сети разошлись: residual сказал `Local_Heating`, контекстная сеть сказала `Periodic_Thermal`. Промах остаётся промахом. Медиана прямоугольника 36.204 °C. Медиана кабеля дальше 40 м 21.525 °C. Доля кольца внутри 0.000.
- `review-00187`: эксперт `Local_Heating`. Residual-сеть `Local_Heating`. Контекстная сеть `Periodic_Thermal`. Сети разошлись: residual сказал `Local_Heating`, контекстная сеть сказала `Periodic_Thermal`. Промах остаётся промахом. Медиана прямоугольника 21.382 °C. Медиана кабеля дальше 40 м 22.105 °C. Доля кольца внутри 0.376.
- `review-00223`: эксперт `Local_Cooling`. Residual-сеть `Local_Cooling`. Контекстная сеть `Periodic_Thermal`. Сети разошлись: residual сказал `Local_Cooling`, контекстная сеть сказала `Periodic_Thermal`. Промах остаётся промахом. Медиана прямоугольника 19.077 °C. Медиана кабеля дальше 40 м 22.059 °C. Доля кольца внутри 0.376.
- `next-0001`: эксперт `Local_Heating`. Residual-сеть `Local_Cooling`. Контекстная сеть `Broad_Shift`. Сети разошлись: residual сказал `Local_Cooling`, контекстная сеть сказала `Broad_Shift`. Промах остаётся промахом. Медиана прямоугольника 33.906 °C. Медиана кабеля дальше 40 м 27.075 °C. Доля кольца внутри 0.000.
- `next-0003`: эксперт `Local_Heating`. Residual-сеть `Local_Heating`. Контекстная сеть `Broad_Shift`. Сети разошлись: residual сказал `Local_Heating`, контекстная сеть сказала `Broad_Shift`. Промах остаётся промахом. Медиана прямоугольника 32.060 °C. Медиана кабеля дальше 40 м 26.372 °C. Доля кольца внутри 0.786.
- `next-0004`: эксперт `Local_Heating`. Residual-сеть `Local_Cooling`. Контекстная сеть `Periodic_Thermal`. Сети разошлись: residual сказал `Local_Cooling`, контекстная сеть сказала `Periodic_Thermal`. Промах остаётся промахом. Медиана прямоугольника 23.192 °C. Медиана кабеля дальше 40 м 18.407 °C. Доля кольца внутри 0.484.
- `next-0005`: эксперт `Background`. Residual-сеть `Local_Heating`. Контекстная сеть `Broad_Shift`. Сети разошлись: residual сказал `Local_Heating`, контекстная сеть сказала `Broad_Shift`. Промах остаётся промахом. `presence=no` остаётся `Background`. Медиана прямоугольника 25.537 °C. Медиана кабеля дальше 40 м 24.678 °C. Доля кольца внутри 0.000.
- `next-0006`: эксперт `Background`. Residual-сеть `Local_Heating`. Контекстная сеть `Periodic_Thermal`. Сети разошлись: residual сказал `Local_Heating`, контекстная сеть сказала `Periodic_Thermal`. Промах остаётся промахом. `presence=no` остаётся `Background`. Медиана прямоугольника 19.432 °C. Медиана кабеля дальше 40 м 21.068 °C. Доля кольца внутри 0.375.
- `next-0007`: эксперт `Background`. Residual-сеть `Local_Cooling`. Контекстная сеть `Broad_Shift`. Сети разошлись: residual сказал `Local_Cooling`, контекстная сеть сказала `Broad_Shift`. Промах остаётся промахом. `presence=no` остаётся `Background`. Медиана прямоугольника 18.803 °C. Медиана кабеля дальше 40 м 18.866 °C. Доля кольца внутри 0.369.
- `next-0009`: эксперт `Local_Heating`. Residual-сеть `Local_Heating`. Контекстная сеть `Periodic_Thermal`. Сети разошлись: residual сказал `Local_Heating`, контекстная сеть сказала `Periodic_Thermal`. Промах остаётся промахом. Медиана прямоугольника 27.871 °C. Медиана кабеля дальше 40 м 24.486 °C. Доля кольца внутри 0.163.
- `next-0010`: эксперт `Background`. Residual-сеть `Local_Cooling`. Контекстная сеть `Local_Cooling`. Промах остаётся промахом. `presence=no` остаётся `Background`. Медиана прямоугольника 14.316 °C. Медиана кабеля дальше 40 м 19.035 °C. Доля кольца внутри 0.159.
- `next-0011`: эксперт `Local_Heating`. Residual-сеть `Local_Heating`. Контекстная сеть `Periodic_Thermal`. Сети разошлись: residual сказал `Local_Heating`, контекстная сеть сказала `Periodic_Thermal`. Промах остаётся промахом. Медиана прямоугольника 39.575 °C. Медиана кабеля дальше 40 м 21.506 °C. Доля кольца внутри 0.070.
## Широкий канал

Отдельный soft-флаг. Он не входит в раздел ограничения halo и не заменяет список 15 экспертных карточек. Правило одно на все карточки файла `docs/evidence/wide_channel.json`: доля кольца внутри не меньше 0.70 и абсолютный разрыв медиан не меньше 2.0 °C. Отсутствующая медиана far cable флаг не включает и не копируется из медианы прямоугольника. Кандидат `Broad_Shift` пишется только когда флаг true. Кандидат стоит рядом с экспертным классом и не заменяет его. Это не седьмой класс. Флаг не ставит anomaly_status и не ставит expert_confirmed. `next-0003` остаётся `Local_Heating`. `serve=research`, operational false. Операционная пара остаётся `median_x3` и `spatial_z_halo` при `|Z| >= 3.5`.

- `next-0003`: флаг true. Кандидат `Broad_Shift` рядом с классом, эксперт `Local_Heating`. Эксперт остаётся `Local_Heating`. Доля кольца внутри 0.786. Медиана прямоугольника 32.060 °C. Медиана кабеля дальше 40 м 26.372 °C. Разрыв медиан 5.688 °C.
- `gap-r1-x295.736-x358.173-t0-344`: флаг true. Кандидат `Broad_Shift` рядом с классом, эксперт `Local_Cooling`. Эксперт остаётся `Local_Cooling`. Доля кольца внутри 0.727. Медиана прямоугольника 18.809 °C. Медиана кабеля дальше 40 м 22.296 °C. Разрыв медиан -3.487 °C.
- `gap-r2-x202.258-x266.971-t0-410`: флаг true. Кандидат `Broad_Shift` рядом с классом, эксперт `Background`. `presence=no` остаётся `Background`. Доля кольца внутри 0.762. Медиана прямоугольника 16.516 °C. Медиана кабеля дальше 40 м 19.227 °C. Разрыв медиан -2.712 °C.
- `gap-r3-x0.000-x72.325-t0-465`: флаг true. Кандидат `Broad_Shift` рядом с классом, эксперт `Local_Cooling`. Эксперт остаётся `Local_Cooling`. Левая дальняя сторона отсутствует, медиана far cable взята со второй стороны и не скопирована из прямоугольника. Доля кольца внутри 0.937. Медиана прямоугольника 22.391 °C. Медиана кабеля дальше 40 м 28.869 °C. Разрыв медиан -6.479 °C.
- `gap-r5-x0.000-x127.000-t0-1537`: флаг true. Кандидат `Broad_Shift` рядом с пустым экспертным классом. Эксперт null. `presence=unclear` класса не даёт. Левая дальняя сторона отсутствует, медиана far cable взята со второй стороны и не скопирована из прямоугольника. Доля кольца внутри 1.000. Медиана прямоугольника 23.742 °C. Медиана кабеля дальше 40 м 20.507 °C. Разрыв медиан 3.235 °C.
- `gap-r5-x155.000-x213.000-t0-1537`: флаг false. Кандидата нет. Эксперт null. Доля кольца внутри 0.667 ниже 0.70, абсолютный разрыв медиан 1.714 °C ниже 2.0 °C, поэтому флаг остаётся false. Доля кольца внутри 0.667. Медиана прямоугольника 22.291 °C. Медиана кабеля дальше 40 м 20.577 °C. Разрыв медиан 1.714 °C.

## Ограничения

Кольцо 8–40 м видит край широкой ступеньки, а не всё пятно. Гладкий или слабый участок часто остаётся ниже 3.5, даже если на графике он виден. Тихий контроль `4-early-quiet-control` при этом пороге молчит. Восемь witness-прямоугольников — наблюдения агента, `expert_confirmed` false, и 8/8 не является точностью класса.

Слабые метки и research-сеть учебные. Совпадение с ними не повод менять детектор. В образ viewer не входят шесть CSV и GPU. Запуск не зависит от каталога `artifacts/`.

## Viewer

Показан operational путь: `median_x3`, затем `spatial_z_halo` при `|Z| >= 3.5`. Hybrid refine на экране нет и статус он не ставит.

Запуск из корня репозитория:

```
docker compose -f compose.viewer.yaml up --build -d
```

Страница: `http://127.0.0.1:8092/`. Тот же разбор JSON: `http://127.0.0.1:8092/view/example`. Пример — локальный нагрев на ровном кабеле, столбцы 45–50 м, плюс 6 °C к фону 22 °C, 12 профилей по 60 с. Окна research рядом с baseline: `http://127.0.0.1:8092/view/windows`. В таблице класс эксперта, класс residual-сети, confirmed или refuted, contrast baseline в °C либо фраза `baseline kept no zone`, медиана прямоугольника, медиана far cable и доля halo. `next-0010` входит в таблицу. Там же строки широкого канала: флаг, кандидат или его отсутствие, экспертный класс или null и те же измерения. `serve=research`, operational false. Совпадение класса не является операционной точностью. Останов: `docker compose -f compose.viewer.yaml down`.
