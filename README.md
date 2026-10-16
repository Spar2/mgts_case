# haklolarp

## Актуальный веб и дизайн

Работа над интерфейсом начинается с [design/README](design/README.md), [Design Guide](docs/web/STYLE-AND-IDENTITY.md) и текущего кода `apps/web/src`. Один актуальный дизайн. Исторические Pen/PNG перенесены в `design/archive/` и не входят в обычную веб-задачу.

## Ветка `ml`

## Захару: новые материалы

- [Независимый разбор PiorlZ/ml-test](docs/ML_TEST_REVIEW.md): все шесть CSV,
  проверка большого REPORT, сравнение с нашим DSP, baseline и план второго пути.
- [Реальные данные и подготовка](docs/REAL_DATA_WORK.md): 340 фрагментов,
  24 вопроса для быстрого просмотра, маски качества и предварительные метки.
- [Скачать готовые пакеты](https://github.com/werd0y45/haklolarp/releases/tag/zakhar-real-data-2026-10-06):
  каталог реальных фрагментов и компактный архив аудита с весами.
- 10 октября 2026: четыре ответа на карточки `1.csv` в
  [zakhar_shortlist_2026-10-10.jsonl](docs/evidence/zakhar_shortlist_2026-10-10.jsonl).
  Ступень `fast common-mode` посчитана рядом с `median_x3` и пару не сменила:
  [решение](docs/DECISIONS.md), [отчёт](docs/evidence/common_mode_report.json).
- [Шесть небольших CNN-весов и контракт](research/ml-test-review/README.md)
  сохранены прямо в Git; [веса real-only ridge](research/real-data-v2/real_masked_ridge.json)
  тоже доступны. Все эти модели исследовательские, без экспертного подтверждения.

Рабочая обработка и API сохраняют прежний detector. Второй путь пока
подготовлен к обоснованной интеграции: качество входа, разрывы времени и
типы событий требуют исправлений. Исторические ветки сохраняют происхождение
экспериментов; актуальная сборка ML-команды находится в `ml`.

## Работа на реальных данных

Подготовлены 340 реальных фрагментов с графиками, масками,
предварительными метками и короткой подборкой для Захара. Новый эксперимент
обучается только на измерениях и не заменяет рабочую очистку:
**[объяснение и результаты](docs/REAL_DATA_WORK.md)**.
Для продолжительного исследования — [промпт Goal](docs/REAL_DATA_GOAL.md).

## Доработка после кросс-ревью, 6 октября 2026

Захару: подробное объяснение изменений, проверок, новой версии признаков и
границ применимости — **[docs/ML_HARDENING.md](docs/ML_HARDENING.md)**.
Рабочая пара `median_x3 + spatial_z_halo` сохранена; исправлены загрузка,
временные разрывы зон, availability/Unknown и граничные метрики. Все шесть
CSV проверены целиком, DSP проходит 8/8 development-контролей.

Отдельная физическая модель обучена и сохранена как research-артефакт;
она не назначает операционные тревоги. Для экспертной разметки подготовлены
232 предложения зон и 36 случайных окон с пустыми метками. См.
[протокол разметки](docs/ANNOTATION_PROTOCOL.md).

Ниже — история предыдущих экспериментов; старые таблицы относятся к их
протоколам, а не к результатам новой версии.

ML-команда хакатона МГТС «Телеком будущего»: indoor DTS (температура
вдоль волокна в помещении), DSP + зоопарк моделей.

Постановка кейса — [docs/CASE.md](docs/CASE.md).
Цели ветки — [docs/GOALS.md](docs/GOALS.md).
Решения — [docs/DECISIONS.md](docs/DECISIONS.md).
Команды пайплайна — [docs/PIPELINE.md](docs/PIPELINE.md).
Стенд и пуш — [TEAM_GUIDE.md](TEAM_GUIDE.md).

## Сейчас сделано

1. Ingestion шести CSV → массивы `[время × метры]` + маска артефактов (`-300`).
2. Денойз-зоопарк: median, Wiener, Savitzky–Golay, Gaussian, bilateral, VisuShrink, BayesShrink, NLM, TV, 2D median. Скор: [docs/DENOISE_SCORES.md](docs/DENOISE_SCORES.md).
3. 1D CNN Noise2Self и 2D U-Net обучены на 4080. Оба хуже `tv_x` / `median_x3` как clean-канал. U-Net остаётся карточкой скора.
4. FastAPI на порту `8082` и `compose.ml.yaml` для staging.
5. Детекция аномалий на `median_x3`: spatial Z (halo 8–40 м), temporal Z, CUSUM, conv-AE. [docs/ANOMALY.md](docs/ANOMALY.md).
6. Бенчмарк: IoU-события, свидетели, и тот же детектор на пяти clean-каналах. Канал остаётся `median_x3`. [docs/BENCHMARK.md](docs/BENCHMARK.md).
7. Цифры по файлам: [docs/DATA.md](docs/DATA.md).
8. Классы: правила формы, CNN имени окна, сканеры кропа и логистическая модель на ширине контекста (`serve=scorecard`). Два CNN-сканера, которые смотрели свидетелей, проходят 1/8. Логистическая модель проходит 7/8; тихий контроль `4.csv` не проходит. LightGBM на тех же признаках даёт 20/20 и 64/250 ложных окон против 57/250. Сеть на кропе 28×120 даёт 11/20. Модель согласия окна на семенах 700000–700007 даёт согласие с меткой посаженного окна 0.959, поиск 39/40 и 5/494 ложных окон против 36/40 и 235/494 у логистической модели на тех же кабелях. Это не точность по кабелю. Свидетели той модели 1/8. Остаточная CNN, граф колонок и трансформер на семенах 800000–800003 дают согласие окна 0.3021, 0.3021 и 0.3919 при поиске 0/20, 0/20 и 4/20. Повтор на семенах 900000–900003 даёт planted-window agreement 0.5221, 0.5586 и 0.6719 при поиске 19/20, 18/20 и 20/20 и ложном фоне 219/238, 179/238 и 170/238. Это не точность по кабелю. Правило выбора оставляет модель согласия, сырые файлы снова не сканировались. Статус тревоги по-прежнему ставит spatial Z на `median_x3`. Поле `plant_room` ближе к шести записям, чем плоская посадка (разрывы уровня, вдоль кабеля и доли невалидных 4.8e-7, 0.923 и 0.000147 против 0.758, 2.301 и 0.0522). Логистическая модель на этом поле даёт witness pass count 7/8 при базе 7/8 и не принята. Текстура фона даёт witness pass count 6/8 при той же базе 7/8 и не принята. Временной избыток над медианой колонки даёт witness pass count 7/8 при той же базе и не принят. Длинное плато даёт witness pass count 3/8 при той же базе и не принято. Контраст плеч даёт witness pass count 6/8 при той же базе и не принят. Форма соседа даёт witness pass count 5/8 при той же базе и не принята. Ранг колонки даёт witness pass count 7/8 при той же базе и не принят. MAD колонки даёт witness pass count 7/8 при той же базе и не принят. Лаг колонки даёт witness pass count 7/8 при той же базе и не принят. Тёплый хвост даёт witness pass count 4/8 при той же базе и не принят. Режим колонки даёт witness pass count 5/8 при той же базе и не принят. Опубликованной классификацией остаётся логистическая модель со witness pass count 7/8. Статус по-прежнему ставит `spatial_z_halo`. [docs/CLASSIFY.md](docs/CLASSIFY.md).
9. Витрина моделей: семья, вход, время, вес, latency, метрика своего протокола. [docs/SCORECARD.md](docs/SCORECARD.md).

## Research: самообучение и детекторы без учителя

Эти методы семейства `research`. Они не ставят тревогу и не сменяют
`median_x3` + `spatial_z_halo`. Скор: [docs/SSL_BENCHMARK.md](docs/SSL_BENCHMARK.md).

| Метод | Откуда |
|---|---|
| Neighbor2Neighbor + гладкость по времени и метрам (`n2n_tsr`) | [Huang et al., CVPR 2021](https://github.com/TaoHuang2018/Neighbor2Neighbor) |
| Слепая колонка ±5 м (`blindspot_x`), обучение только на `MGTS_data` | [jDAS, van den Ende et al.](https://github.com/martijnende/jDAS) |
| Zero-Shot Noise2Noise на одном поле (`zsn2n`) | [Mansour and Heckel, CVPR 2023, arXiv:2303.11253](https://arxiv.org/abs/2303.11253) |
| Прогноз остатка Chronos-Bolt | [amazon/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) |
| Robust PCA, principal component pursuit | Candès, Li, Ma, Wright, Journal of the ACM 2011 |
| Matrix profile, `stump` / `mstump` | [STUMPY](https://github.com/stumpy-dev/stumpy) |
| PatchCore на тайлах с низким spatial Z | [Anomalib](https://github.com/openvinotoolkit/anomalib) |
| Перцентиль тихих окон и peaks-over-threshold | SPOT / EVT, Siffer et al., KDD 2017; запасной POT на `scipy.stats.genpareto` |

`pip install -e .[ssl,anomaly]` ставит необязательные пакеты. Базовый список зависимостей тот же.

## Быстрый старт

```powershell
python -m pytest tests -q
python scripts/transform_csv.py
python scripts/denoise_records.py --only 5
```

Сырые CSV лежат в `MGTS_data/` локально и в git не входят.
## Ветка `web`

- [Доступ участника: приглашения, пароли и роли](docs/web/TEAM-ACCESS.md)
- [Production-конфигурация, ежедневные age-копии и восстановление](docs/web/PRODUCTION-OPERATOR.md)
- [Приёмка публичного демо и командного контура](docs/web/PRODUCTION-ACCEPTANCE.md)

- [Рабочее веб-приложение: запуск, реальные данные, экспертная проверка и API](docs/web/README.md)

- [Актуальный источник дизайна и правила работы](design/README.md)
- [Design Guide для участников и моделей](docs/web/STYLE-AND-IDENTITY.md)
- [Гайд Front-команды](TEAM_GUIDE.md)
