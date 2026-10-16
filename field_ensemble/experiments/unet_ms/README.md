# unet_ms

U-Net с фиксированными усредняющими окнами на входе. Класс `Integrate` добавляет четыре карты (окна (7, 15) и (21, 61) по двум температурным каналам, нормировка на \(\sqrt{N}\)), `DTSUNetMS` подаёт 7 каналов в обычный `DTSUNet`. Код живёт в `dts/models.py`, обучение — `scripts/train_stage1.py --arch unet_ms`, вес `artifacts/stage1/unet_ms.pt`. Отдельного модуля нет: классы `Integrate` и `DTSUNetMS` не вырезались и остаются в `dts/models.py`. `build("unet_ms")` берёт их оттуда.

Многомасштабное слияние уже посчитанной карты победителя — это `readmes/multiscale.md`, другой эксперимент.

## Цифры

`artifacts/eval/stage1_v8_round3.json`, порог с val, точка FAR \(10^{-3}\).

| метод | val IoU | test IoU |
| --- | ---: | ---: |
| unet (тот же протокол) | 0.368268 | 0.362516 |
| unet_ms | 0.358951 | 0.344823 |

Val loss в `docs/NOTE.md` у unet_ms ниже (0.275 за 40 эпох против 0.382 за 24 у U-Net, `artifacts/stage1/train_log_unet_ms.json`), но рабочая точка FAR \(10^{-3}\) хуже.

Ранний ансамбль, куда unet_ms входил (`ens_cnn3_lgbm`, U-Net + unet_ms + TCN + LightGBM), на v8 test давал IoU 0.401 (`docs/STATE.md` / таблицы `docs/RESULTS.md`). Текущий победитель без unet_ms: val 0.3867, test 0.4053. В `models_release/stage1_thresholds.json` члена `unet_ms` нет.

## Почему отклонён

На val при FAR \(10^{-3}\) интегрирующий вход ниже обычного U-Net того же раунда (0.359 против 0.368) и ниже победителя (0.387). Меньшая val loss селектором не была. Класс остаётся доступен через `build("unet_ms")`, но дефолтный `Analyzer` его не строит.
