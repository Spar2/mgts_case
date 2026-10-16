# mgts_case

Снимок ML-части хакатона МГТС: температура вдоль оптического волокна в помещении. Коммит источника — ветка `ml` репозитория haklolarp, `71fc4eb`. Ветка `main` того репозитория сюда не вливалась.

Подробный запуск: [АРХИТЕКТУРА-И-ЗАПУСК.md](АРХИТЕКТУРА-И-ЗАПУСК.md). Команды дня в исходной ветке: [docs/PIPELINE.md](docs/PIPELINE.md).

## Что внутри

| Папка | Зачем |
| --- | --- |
| `fiber/` | Пайплайн: чтение CSV, очистка, зоны, классы формы, API. Шаги собраны в `fiber/pipeline.py`. |
| `fiber/models/` | Код сетей: 1D CNN, 2D U-Net, автоэнкодер, классификатор кропа, граф колонок, трансформер. |
| `scripts/` | Команды прогона и обучения. |
| `tests/` | Проверки без CSV и без видеокарты. |
| `models/weights/` | Сюда обучение кладёт веса. В git только пояснение: сами `.pt` появляются после локального обучения. |
| `research/ml-test-review/weights/` | Шесть маленьких CNN, уже лежащих в git. Тревогу не ставят. |
| `solutions/larpathon/` | Отдельный прототип ансамбля и вес `weights/synth_patch_net.pt`. |

Сырых `1.csv`–`6.csv` в репозитории нет. Значение `−300` — пропуск, не температура.

Рабочая обработка: очистка `median_x3`, кандидат зоны — `spatial_z_halo`. Нейросети это не заменяют.

## Запуск

Нужен Python 3.11 или новее. Команды из корня репозитория. CSV положить в каталог `MGTS_data`.

```powershell
python -m pip install -e ".[dev]"
python scripts/transform_csv.py --raw-root MGTS_data --out-root artifacts/records
python scripts/denoise_records.py
python scripts/detect_anomalies.py
python scripts/classify_zones.py
python -m pytest tests -q
```

API:

```powershell
python -m pip install -e ".[api]"
python -m uvicorn fiber.api:app --host 127.0.0.1 --port 8082
```

Обучение на GPU дописывает файлы в `models/weights/`:

```powershell
python -m pip install -e ".[train]"
python scripts/train_conv1d_n2self.py --device cuda --epochs 8
```
