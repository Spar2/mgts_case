# Как гонять данные (ветка `ml`)

Команды из корня репозитория. Python 3.13 + CUDA torch уже стоят на машине
с 4080 SUPER.

## 1. CSV → native массивы

```powershell
python scripts/transform_csv.py --raw-root MGTS_data --out-root artifacts/records
```

На каждый файл: `raw_c.npy`, `valid.npy`, `x_m.npy`, `time_s.npy`, `meta.json`.
Индекс всех шести — `artifacts/records/index.json`.

## 2. Денойз

```powershell
python scripts/denoise_records.py
```

По умолчанию: `median_x3`, `median_xt`, `savgol_x`, `bilateral_x`,
`wavelet_x`, `tv_x`. Для отладки одного объекта:

```powershell
python scripts/denoise_records.py --only 5 --methods median_x3 wavelet_x
```

Скор денойза (синтетический спайк и контраст `5.csv`):

```powershell
python scripts/score_denoise.py
```

Таблица: [DENOISE_SCORES.md](DENOISE_SCORES.md).

## 3. Аномалии на median_x3

```powershell
python scripts/detect_anomalies.py
python scripts/plot_anomaly.py
python scripts/train_conv_ae.py --device cuda
python scripts/score_ae.py
```

Решение `Normal` / `Anomaly` и зоны — spatial Z. Описание: [ANOMALY.md](ANOMALY.md).

Классы зон по форме (не причина события):

```powershell
python scripts/classify_zones.py
```

Описание: [CLASSIFY.md](CLASSIFY.md).

## 4. Обучить CNN-денойзер (GPU)

```powershell
python scripts/train_conv1d_n2self.py --device cuda --epochs 8
python scripts/train_unet2d_denoise.py --device cuda --epochs 6 --crops 4000
```

Веса пишутся в `models/weights/` (gitignore). 2D U-Net — карточка скора,
не clean-канал. См. [DENOISE_SCORES.md](DENOISE_SCORES.md).

Применить:

```powershell
python -c "from pathlib import Path; from fiber.io import Record; from fiber.denoise import apply_method; r=Record.load(Path('artifacts/records/5')); d=apply_method('conv1d_n2self', r.raw_c, r.valid, weights=Path('models/weights/conv1d_n2self.pt')); print(d.shape)"
```

## 5. Бенчмарк

```powershell
python scripts/benchmark_models.py
```

Синтетика считается всегда. Свидетели 4/5/6.csv требуют `artifacts/records`
и `median_x3`. Таблица: [BENCHMARK.md](BENCHMARK.md).

Тот же детектор на нескольких clean-каналах:

```powershell
python scripts/score_channels.py
```

Классификатор кропа на посаженных классах (GPU, веса в gitignore):

```powershell
python scripts/train_patch_classifier.py --device cuda --epochs 16
```

Поиск класса по длинной трассе. Сначала обучение и выбор весов на посаженном
кабеле, затем скан сырых записей. Скан читает выбор из
`artifacts/classify/scanner_report.json` и сам порог не меняет.

```powershell
python scripts/train_patch_scanner.py --device cuda --epochs 16
python scripts/scan_traces.py
```

Витрина всех измеренных моделей: [SCORECARD.md](SCORECARD.md).
Скрипт ниже только мерит latency и не пересчитывает скоры.

```powershell
python scripts/score_vitrine.py
```

Сканер на кабеле с профилем помещения. Выбор весов — по посаженному поиску.
Скан свидетелей — тем же замороженным правилом, в отдельный файл.

```powershell
python scripts/train_patch_scanner.py --profile --device cuda --epochs 16
python scripts/scan_traces.py --report artifacts/classify/profile_report.json --out artifacts/classify/profile_scan.json
```

Сканер длинной полосы на том же профиле. Скан свидетелей запускается только
если в `persistent_report.json` поле `choice` равно `persistent`.
Измеренный прогон выбрал `profile`, поэтому вторая команда не выполнялась.

```powershell
python scripts/train_patch_scanner.py --persistent --device cuda --epochs 16
python scripts/scan_traces.py --report artifacts/classify/persistent_report.json --out artifacts/classify/persistent_scan.json
```

Логистическая модель на ширине контекста. Скан запускается только если в
`context_report.json` поле `choice` равно `context`.

```powershell
python scripts/train_context_scanner.py --device cuda
python scripts/scan_context.py
```

LightGBM на тех же признаках контекста. Скан запускается только если в
`lgb_report.json` поле `choice` равно `lgb`. Измеренный прогон выбрал
`logistic`, поэтому вторая команда не выполнялась.

```powershell
python -m pip install lightgbm
python scripts/train_lgb_scanner.py
python scripts/scan_lgb.py
```

CNN на кропе 28×120. Скан запускается только если в `wide_report.json`
поле `choice` равно `wide`. Измеренный прогон выбрал `logistic`, поэтому
вторая команда не выполнялась.

```powershell
python scripts/train_wide_context.py --device cuda
python scripts/scan_wide.py
```

Модель согласия окна. Скан запускается только если в `agree_report.json`
поле `choice` равно `agree`. Измеренный прогон выбрал `agree`, поэтому
вторая команда выполнена. Порог свидетеля после просмотра не менялся.

```powershell
python scripts/train_context_agree.py
python scripts/scan_agree.py
```

Остаточная CNN, граф колонок и трансформер. Скан запускается только если в
отчёте поле `choice` равно имени этой модели. Прогон на семенах
900000–900003 тоже оставил `agree`, поэтому команды скана не выполнялись.

```powershell
python scripts/train_depth_cnn.py
python scripts/scan_depth_cnn.py
python scripts/train_column_graph.py
python scripts/scan_column_graph.py
python scripts/train_column_transformer.py
python scripts/scan_column_transformer.py
```

Поле, похожее на шесть записей. Команда учит логистическую модель и сразу
считает восемь свидетелей. Веса пишутся только если witness pass count
строго выше 7/8. Измеренный прогон дал 7/8 и не был принят.

```powershell
python scripts/train_room_context.py
```

Текстура фона. Команда учит логистическую модель и сразу считает восемь
свидетелей через `decide_witnesses`. Веса пишутся только если witness pass
count строго выше 7/8. Измеренный прогон дал 6/8 и не был принят.

```powershell
python scripts/train_texture_context.py
```

Временной избыток над медианой колонки. Команда учит логистическую модель и
сразу считает восемь свидетелей через `decide_witnesses`. Веса пишутся только
если witness pass count строго выше 7/8. Измеренный прогон дал 7/8 и не был
принят.

```powershell
python scripts/train_excess_context.py
```

Длинное плато. Команда учит логистическую модель и сразу считает восемь
свидетелей через `decide_witnesses`. Веса пишутся только если witness pass
count строго выше 7/8. Измеренный прогон дал 3/8 и не был принят.

```powershell
python scripts/train_plateau_context.py
```

Контраст плеч. Команда учит логистическую модель и сразу считает восемь
свидетелей через `decide_witnesses`. Веса пишутся только если witness pass
count строго выше 7/8. Измеренный прогон дал 6/8 и не был принят.

```powershell
python scripts/train_contrast_context.py
```

Форма соседа. Команда учит логистическую модель и сразу считает восемь
свидетелей через `decide_witnesses`. Веса пишутся только если witness pass
count строго выше 7/8. Измеренный прогон дал 5/8 и не был принят.

```powershell
python scripts/train_neighbor_context.py
```

Ранг колонки. Команда учит логистическую модель и сразу считает восемь
свидетелей через `decide_witnesses`. Веса пишутся только если witness pass
count строго выше 7/8. Измеренный прогон дал 7/8 и не был принят.

```powershell
python scripts/train_quantile_context.py
```

MAD колонки. Команда учит логистическую модель и сразу считает восемь
свидетелей через `decide_witnesses`. Веса пишутся только если witness pass
count строго выше 7/8. Измеренный прогон дал 7/8 и не был принят.

```powershell
python scripts/train_stability_context.py
```

Лаг колонки. Команда учит логистическую модель и сразу считает восемь
свидетелей через `decide_witnesses`. Веса пишутся только если witness pass
count строго выше 7/8. Измеренный прогон дал 7/8 и не был принят.

```powershell
python scripts/train_persist_context.py
```

Тёплый хвост. Команда учит логистическую модель и сразу считает восемь
свидетелей через `decide_witnesses`. Вне хвоста окно становится фоном.
Веса пишутся только если witness pass count строго выше 7/8. Измеренный
прогон дал 4/8 и не был принят.

```powershell
python scripts/train_bulk_context.py
```

Режим колонки. Команда учит логистическую модель и сразу считает восемь
свидетелей через `decide_witnesses`. Нагрев — тёплый хвост, шумный пик или
холодный избыток. Веса пишутся только если witness pass count строго выше
7/8. Измеренный прогон дал 5/8 и не был принят.

```powershell
python scripts/train_regime_context.py
```

Опубликованной классификацией остаётся логистическая модель со witness pass
count 7/8. Порог, подобранный по ящикам свидетелей, не входит в витрину.

## 6. Тесты (без CSV, без GPU)

```powershell
python -m pytest tests -q
```

## 7. Локальный API

```powershell
python -m uvicorn fiber.api:app --host 127.0.0.1 --port 8082
```

`GET /health`, `GET /methods`, `POST /denoise/profile`.
