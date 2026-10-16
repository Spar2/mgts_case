Сюда кладутся обученные веса (`*.pt`, `*.onnx`, `*.pkl`).

Каталог в `.gitignore`. Обучение — на локальной RTX 4080 SUPER:

```powershell
python scripts/train_conv1d_n2self.py --device cuda
python scripts/train_patch_classifier.py --device cuda --epochs 16
python scripts/train_patch_scanner.py --device cuda --epochs 16
python scripts/train_patch_scanner.py --profile --device cuda --epochs 16
python scripts/train_patch_scanner.py --persistent --device cuda --epochs 16
python scripts/train_context_scanner.py --device cuda
python scripts/train_lgb_scanner.py
python scripts/train_wide_context.py --device cuda
python scripts/train_context_agree.py
python scripts/train_depth_cnn.py
python scripts/train_column_graph.py
python scripts/train_column_transformer.py
```

Три последние команды учат 48 эпох и считают семена 900000–900003.
Скан сырых файлов они не запускают, пока правило выбора не отдаст им трассу.

```powershell
python scripts/train_room_context.py
```

Эта команда учит логистическую модель на поле `plant_room` и пишет
`room_context.json` только если witness pass count строго выше 7/8.
Измеренный прогон дал 7/8, файл не создан.

```powershell
python scripts/train_texture_context.py
```

Эта команда учит логистическую модель на поле `plant_texture` и пишет
`texture_context.json` только если witness pass count строго выше 7/8.
Измеренный прогон дал 6/8, файл не создан. Опубликованной классификацией
остаётся логистическая модель со witness pass count 7/8. Статус по-прежнему
ставит `spatial_z_halo`.

```powershell
python scripts/train_excess_context.py
```

Эта команда учит логистическую модель на поле `plant_excess` и пишет
`excess_context.json` только если witness pass count строго выше 7/8.
Измеренный прогон дал 7/8, файл не создан. Статус по-прежнему ставит
`spatial_z_halo`.

```powershell
python scripts/train_plateau_context.py
```

Эта команда учит логистическую модель на поле `plant_plateau` и пишет
`plateau_context.json` только если witness pass count строго выше 7/8.
Измеренный прогон дал 3/8, файл не создан. Статус по-прежнему ставит
`spatial_z_halo`.

```powershell
python scripts/train_contrast_context.py
```

Эта команда учит логистическую модель на поле `plant_contrast` и пишет
`contrast_context.json` только если witness pass count строго выше 7/8.
Измеренный прогон дал 6/8, файл не создан. Статус по-прежнему ставит
`spatial_z_halo`.

```powershell
python scripts/train_neighbor_context.py
```

Эта команда учит логистическую модель на поле `plant_neighbor` и пишет
`neighbor_context.json` только если witness pass count строго выше 7/8.
Измеренный прогон дал 5/8, файл не создан. Статус по-прежнему ставит
`spatial_z_halo`.

```powershell
python scripts/train_quantile_context.py
```

Эта команда учит логистическую модель на поле `plant_quantile` и пишет
`quantile_context.json` только если witness pass count строго выше 7/8.
Измеренный прогон дал 7/8, файл не создан. Статус по-прежнему ставит
`spatial_z_halo`.

```powershell
python scripts/train_stability_context.py
```

Эта команда учит логистическую модель на поле `plant_stability` и пишет
`stability_context.json` только если witness pass count строго выше 7/8.
Измеренный прогон дал 7/8, файл не создан. Статус по-прежнему ставит
`spatial_z_halo`.

```powershell
python scripts/train_persist_context.py
```

Эта команда учит логистическую модель на поле `plant_persist` и пишет
`persist_context.json` только если witness pass count строго выше 7/8.
Измеренный прогон дал 7/8, файл не создан. Статус по-прежнему ставит
`spatial_z_halo`.

```powershell
python scripts/train_bulk_context.py
```

Эта команда учит логистическую модель на поле `plant_bulk` и пишет
`bulk_context.json` только если witness pass count строго выше 7/8.
Измеренный прогон дал 4/8, файл не создан. Статус по-прежнему ставит
`spatial_z_halo`.

```powershell
python scripts/train_regime_context.py
```

Эта команда учит логистическую модель на поле `plant_regime` и пишет
`regime_context.json` только если witness pass count строго выше 7/8.
Измеренный прогон дал 5/8, файл не создан. Статус по-прежнему ставит
`spatial_z_halo`.

В git только этот README и код архитектур в `fiber/models/`.
