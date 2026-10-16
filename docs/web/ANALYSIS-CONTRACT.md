# Захару: поля расчёта и графики на сайте

Версия контракта: `dts-analysis-result-v2`. Это передача выполненного расчёта, не весов. Сайт не запускает обучение и не объявляет исследовательский score вероятностью аварии.

## Быстрый путь

1. Узнать паспорт исходника: `GET /api/v1/records`, затем `GET /api/v1/records/5`. Нужны `source_sha256`, `x_m`, `time_s`, `time_kind`. Рабочие API требуют входа.
2. Выбрать исходные полуоткрытые индексы `[row_start,row_end) × [column_start,column_end)`.
3. Сохранить результат как NumPy-массивы и metadata.json. Координаты брать из подготовленного исходника без округления. Время — от начала записи, не от начала окна. Не угадывать часовой пояс.
4. Упаковать и проверить на сервере:

```text
python scripts/analysis-package.py package --records /data/records --directory /data/calculation --metadata metadata.json
python scripts/analysis-package.py validate --records /data/records --directory /data/calculation
```

5. Оператор импортирует с явным проектом и существующим ID аккаунта автора:

```text
python scripts/analysis-package.py import --records /data/records --directory /data/calculation --project local --author ACCOUNT_ID
```

Команда читает `DTS_DATABASE_URL`, `DTS_RUNTIME` из безопасного окружения. Выводит `run_id` и `/analysis?run_id=...`. Повтор идентичного пакета для того же автора/проекта возвращает прежний ID. Существующие пользователи и ответы не меняются.

## Файлы

Обязательны `x_m.npy`, `time_s.npy` (Float64, точное совпадение с соответствующими срезами нативных осей). Необязательны пары:

| Значения | Маска доступности | Единица |
|---|---|---|
| processed_temperature_c.npy | processed_temperature_c_valid.npy | °C |
| background_temperature_c.npy | background_temperature_c_valid.npy | °C |
| score.npy | score_valid.npy | Задаётся в паспорте |

Массивы — Float32/Float64 формы `[row_end-row_start,column_end-column_start]`; маски — bool той же формы. NaN допустим только под false; Infinity запрещён везде. Object/pickle не принимаются. Минимум один слой или zones.json. Sentinel исходника нельзя превращать в измерение: маска источника применяется дополнительно даже к заполненным алгоритмом клеткам.

```python
np.save("calculation/x_m.npy", record.x_m[c0:c1])
np.save("calculation/time_s.npy", record.elapsed_s[r0:r1])
np.save("calculation/processed_temperature_c.npy", actual_clean)
np.save("calculation/processed_temperature_c_valid.npy", actual_valid.astype(bool))
```

Здесь `actual_clean` — настоящий результат вашего метода; пример его не вычисляет. Пакет не должен содержать исходный CSV, пароли или пользовательские журналы.

## metadata.json

```json
{
  "region": {"record_id":"5","row_start":359,"row_end":539,"column_start":150,"column_end":260},
  "source_sha256":"SHA_ИЗ_ПАСПОРТА",
  "method":"ИМЯ_ВАШЕГО_МЕТОДА",
  "method_version":"COMMIT_ИЛИ_SHA_АРТЕФАКТА",
  "parameters":{},
  "calculated_at":"2026-10-13T12:00:00+03:00",
  "training_origin":"unknown",
  "research_status":"research",
  "limitations":"Независимое качество на авариях не установлено",
  "score":null
}
```

Упаковщик добавляет `schema_version` и `files` с SHA каждого файла. Пример требует замены SHA/метода/версии на фактические. `training_origin`: real/synthetic/mixed/untrained/unknown. `research_status`: reference/research/candidate/not_accepted; это заявленный статус расчёта, не сертификация.

При наличии score:

```json
{"name":"Spatial Z","unit":"Z","meaning":"Знаковое отклонение от пространственного фона","kind":"signed","threshold":3.5,"comparison":"abs_ge"}
```

`kind`: signed/nonnegative. Порог необязателен; при наличии конечный неотрицательный и `comparison`: ge/abs_ge. Отсутствующий порог не подбирается сайтом.

`zones.json` — список объектов с `id`, `row_start/row_end`, `column_start/column_end`, необязательными `shape_hypothesis` и описанием. Индексы абсолютные, исходные, внутри покрытия. Физические причины не считаются доказанными. Упаковщик не интерполирует пересчитанные сетки.

## API просмотра — уже авторизованный веб-контур

| Запрос | Результат |
|---|---|
| GET /api/v2/result-runs?record_id=5&method=...&method_version=...&research_status=... | Паспорта, без больших массивов |
| GET /api/v2/result-runs/{id} | Паспорт, слои, зоны |
| GET /api/v2/result-runs/{id}/tiles?layer=score&width=256&height=256 | Бинарный блок; также x0/x1/t0/t1 внутри покрытия |
| GET /api/v2/result-runs/{id}/slices?row=359&column=192&zone_id=... | Точные профили, ряды, покрытие, зональные p10/median/p90 |

Слои: raw/processed_temperature_c/background_temperature_c/score/difference. Температура: медиана валидных ячеек в физическом бине. Score: знаковое значение с максимальным модулем (не медиана). Разность считается по исходным парам до агрегации. `time_s` рядов содержит null-маркеры разрывов; `native_time_s` содержит только наблюдённые времена. p10–p90 описывает разброс пространственных измерений, не доверительный интервал модели.

Первый импорт полей выполняет оператор, HTTP-загрузки пакетов нет. Предел пакета 512 MiB. Старый POST /api/v1/results для полного одиночного профиля остаётся (256 KiB). Большие матрицы через него не отправлять. Старые совместимые профильные запуски доступны через адаптер каталога.

## Адаптер текущей ML-ветки

```text
python scripts/package-ml-field.py --ml-checkout /code/ml --records /data/records --record 5 --rows 359 539 --columns 150 260 --output /data/new-calculation
```

Это явный запуск `fiber.pipeline.analyze_field` на реальном окне до 1M клеток. Источник ML фиксируется коммитом; незакоммиченные изменения `fiber` отклоняются. Сохраняются clean/background/Z/availability, manifest и зоны. Референсная пара median_x3+spatial_z_halo обозначается untrained/reference и не выдаётся за финальную обученную модель. Закруглённые границы зон ML привязываются к ближайшим нативным центрам; происхождение этой операции сохранено.

## Сохранение кейса

Страница /analysis позволяет выбрать запуск, точку и предложенную зону, затем создать обычный неизменяемый кейс. Новый снимок сохраняет точный профиль каждого слоя и явно обозначенный обзор-подвыборку, без полного CSV. Старые кейсы/ZIP не переписываются. При превышении лимита снимка уменьшите окно; усечения нет. SHA пакета подтверждают целостность относительно manifest, не авторство или качество модели.

Снимок также удерживает выбранную точку, её временные ряды, покрытие и выбранную зону. POST /api/v2/cases принимает необязательные analysis_column и analysis_zone_id; прежние запросы и их ключи сохранения остаются совместимыми. Обработка не передана — показатели её изменений не создаются, даже при наличии score.

## Внешние карточки

Четыре материала из ML-коммита 3653994 сохранены в [external/zakhar-shortlist-20261010.jsonl](external/zakhar-shortlist-20261010.jsonl). Их SHA, оси, маски и Float32-ячейки сверены с real-data-v2 и нативным архивом. UTC epoch фрагментов проверен по явно заданному смещению исходного timestamp. Содержание заметок и заявленное авторство не становятся автоматически подтверждённым заключением.

```text
python scripts/import-external-reviews.py --reviews notes.jsonl --pack /data/real-data-v2 --records /data/records --source-commit ML_COMMIT --project local --validate-only
```

Без --validate-only оператор сохраняет материалы отдельно от reviews. GET /api/v2/result-runs/{id}/external-reviews возвращает пересекающиеся материалы своего проекта; они показаны с предупреждением. Для рабочего ответа автор использует существующий вход и экспертную проверку.

Бинарные артефакты включаются в существующий backup/restore. Общий деплой и новый ML runtime остаются отдельным этапом.

Примеры интерфейса на настоящем reference-расчёте: [1440 px](screenshots/analysis-1440.png), [390 px](screenshots/analysis-390.png). Графики не подтверждают физическую причину или качество аварийной классификации.

У /slices есть необязательные row_start/row_end/column_start/column_end внутри покрытия запуска. Все возвращаемые профили, ряды и покрытие относятся к этому окну. Выбор зоны передаёт её границы, а ссылка из кейса сохраняет run_id, профиль, точку и zone_id.
