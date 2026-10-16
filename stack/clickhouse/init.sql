-- LAN-стенд haklolarp: схема ClickHouse v1.
-- Правка этого файла на уже существующем томе не применяется: ClickHouse
-- выполняет initdb.d только при первой инициализации. Новая схема — только
-- новым файлом в stack/clickhouse/migrations/. Уже применённый файл
-- миграции не переписывать.
-- Монтируется в /docker-entrypoint-initdb.d, выполняется по алфавиту до старта.
-- Источник контрактов: ветка origin/sol-ml-analyze (docs/DATA_CONTRACT.md,
-- docs/INFERENCE_CONTRACT.md, models/synthetic-v2/manifest.json).

CREATE DATABASE IF NOT EXISTS haklolarp;

-- Сырые профили DTS: одна строка = один профиль одного прибора/канала.
-- x_m/temp_c/invalid_reason — параллельные native-массивы (DATA_CONTRACT v1):
-- invalid_reason: 0=valid, 1=sentinel -300, 2=nonfinite,
-- 4=missing tail, 8=masked out.
-- ts — UTC-эпоха из исходной строки; ts_raw — исходная строка с offset.
CREATE TABLE IF NOT EXISTS haklolarp.profiles_raw
(
    dts_id String,
    channel_id String,
    ts_raw String,
    ts DateTime64(6, 'UTC'),
    row_idx UInt32,
    segment_id UInt32,
    x_m Array(Float64),
    temp_c Array(Float32),
    invalid_reason Array(UInt8)
)
ENGINE = MergeTree()
ORDER BY (dts_id, channel_id, ts);

-- Кандидаты событий от инференса (INFERENCE_CONTRACT.md):
-- 4-связность, merge aliases — на стороне analyzer, сюда только итог.
-- cause/alarm всегда unknown_pending_review на этом этапе.
CREATE TABLE IF NOT EXISTS haklolarp.events
(
    model_key String,
    method String,
    event_id String,
    dts_id String,
    channel_id String,
    segment_id UInt32,
    t_first DateTime64(6, 'UTC'),
    t_last DateTime64(6, 'UTC'),
    x0 Float64,
    x1 Float64,
    max_score Float32,
    status String
)
ENGINE = MergeTree()
ORDER BY (model_key, method, dts_id, channel_id, t_first);

-- Реестр моделей: name -> manifest.json целиком (см. stack/registry/).
-- Веса лежат в registry-volume, здесь только манифесты.
CREATE TABLE IF NOT EXISTS haklolarp.models_registry
(
    name String,
    model_key String,
    manifest_json String,
    created DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (name);
