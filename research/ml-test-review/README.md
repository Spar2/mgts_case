# Захару: веса повторного обучения ml-test

Шесть `weights/1.pt` … `weights/6.pt` — CNN `state_dict`, около 25 КБ каждый.
Они обучены в нашем аудите на соответствующих реальных CSV без генерации
синтетических событий. Это веса повторного CPU-прогона, не исходные веса
внешнего отчёта. Все полные score maps после загрузки воспроизведены точно.

Исходная архитектура — `ConvAutoencoder` из PiorlZ/ml-test, commit
`33e12ac19b7b6318dc48b5f3e8c1239aaaf9244e`.
Загрузка: `torch.load(path, map_location="cpu", weights_only=True)`, затем
`model.load_state_dict(state, strict=True)`.

Вход — CV residual именно исходного пайплайна. В [manifest.json](manifest.json)
для каждого файла сохранены источник/SHA, центр и масштаб нормализации,
patch/stride, ретроспективный порог, версии и проверка score. Это не общий
детектор для произвольного нового кабеля; рабочий API эти веса не использует.
Оригинальное обучение не учитывало valid mask. Экспертная точность неизвестна.
Веса Isolation Forest в этот пакет не входят; CatBoost в основном прогоне
не обучался.

[Разбор подхода и ограничений](../../docs/ML_TEST_REVIEW.md).
[Готовые архивы с реальными материалами](https://github.com/werd0y45/haklolarp/releases/tag/zakhar-real-data-2026-10-06).
