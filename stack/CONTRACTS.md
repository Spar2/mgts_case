# Замороженные контракты LAN-стенда v1

Источник истины по смыслу — ветка `origin/sol-ml-analyze`
(`docs/DATA_CONTRACT.md`, `docs/INFERENCE_CONTRACT.md`,
`docs/VIEWER_CONTRACT.md`, `docs/CATALOG_CONTRACT.md`).
Здесь — только то, что обязаны соблюдать backend/streamer/CH на стенде.

## Профиль (вход инференса)

```json
{"dts_id": "str, непустая", "channel_id": "str, непустая",
 "timestamp": "ISO8601, напр. 2026-01-01T00:00:00+03:00",
 "points": [[x_m, T], ...], "valid_mask": [0/1, ...]}
```

- `points` — непустой список `[метры, °C]`; `x` конечный, `>= 0`, строго растёт.
- Пропуск температуры — `null` (не NaN/Infinity); sentinel `-300` хранить как есть.
- `valid_mask` опциональна, длина = длине `points`.
- Ошибка схемы/геометрии/времени — отказ **до** изменения состояния (HTTP 400).
- Разрыв `> 300 с` — новый сегмент, сброс временной базы и подтверждения.
- Время строго растёт; смешивать zoned/naive в одной сессии нельзя.

## HTTP API backend (порт 8000)

| Метод | Маршрут | Тело/параметры | Ответ |
|---|---|---|---|
| GET | `/healthz` | — | `{"ok": true}` |
| GET | `/api/manifest` | — | records `1..6`, model `{name, model_key}`, `mutation_token`, fields |
| POST | `/api/session` | `{"mode": "live"}` + `X-DTS-Token` | `{session_id, model_key, mode}`; макс 16 сессий, TTL 300 с |
| POST | `/api/infer` | `{session_id, profile}` + `X-DTS-Token` | `{events, closed_events, model_key, segment_id, profile_index, queued, stream_id}`; Redis недоступен → 503 |
| POST | `/api/session/close` | `{session_id}` + `X-DTS-Token` | `{"closed": true}` |
| GET | `/api/snapshot` | `?record=&row=` | события на шаге (заглушка: пусто) |

POST требует `Content-Type: application/json`, тело 1..2097152 байт,
заголовок `X-DTS-Token` = `mutation_token` из `/api/manifest`.

Ошибки токена/тела (см. `stack/smoke.sh`): чужой/битый токен → `403`,
не-объект в теле → `400`, чужая сессия → `404`. Токен протухает при
рестарте backend — обновить из `/api/manifest`, смотреть `/tmp/smoke.log`.

## manifest.json модели (реестр)

Обязательные поля: `model_key`, `thresholds` (по методам),
`scaling {center, scale, clip}`, `decision_policies`,
`artifacts_sha256` (весa), `code_sha256`, `purpose`.
Смена весов/порогов = новый `model_key` + новая сессия/replay.

## Redis Streams

- `dts:profiles` — принятые backend профили (поле `payload`: канонический JSON
  профиля как в `profiles_raw`, плюс `kind=profile`). Группа `ingesters`,
  читает сервис ingester (XREADGROUP → INSERT → XACK). `MAXLEN ~ 20000`.
- `dts:events` — резерв под analyzer; группа создаётся с MKSTREAM, писателя нет.
- Redis: `appendonly yes`, `maxmemory 4gb`, `noeviction` (переполнение —
  явная ошибка продюсеру, не молчаливый дроп; стрим держит весь бэклог
  в RAM ~1ГБ на полный датасет, 4gb — запас; перед каноническим replay
  стрим сбрасывается DEL в `stream-all.sh`, иначе повторы копятся до OOM).
- Аутентификации Redis/CH нет осознанно: только loopback, стенд LAN.
  Опция на будущее — `requirepass` + `AUTH` в miniresp + `REDIS_PASSWORD`
  (потянет вопрос хранения секрета; по умолчанию выключено).
- Python-сервисы: `cap_drop: [ALL]`, `no-new-privileges:true`.
  `read_only` не включаем: python пишет `__pycache__`, ingester — heartbeat.
- Семантика at-least-once: крэш ingester между INSERT и XACK даст дубли
  (дедупликации в CH нет). Данные стенда — воспроизводимый replay:
  лечится TRUNCATE + повторный прогон. Follow-up: ReplacingMergeTree
  по `(dts_id, channel_id, ts, row_idx)`.
- `MAXLEN` режет даже неподтверждённое — запас 20000 ≫ 6749; лаг/pending
  смотреть `python3 verify.py streams` (порог для ручной проверки: lag > 1000).

## ClickHouse (`stack/clickhouse/init.sql`)

- `haklolarp.profiles_raw` — сырые профили, `ORDER BY (dts_id, channel_id, ts)`.
- `haklolarp.events` — кандидаты событий, `ORDER BY (model_key, method, dts_id, channel_id, t_first)`.
- `haklolarp.models_registry` — манифесты по `name`.

Отклонения заглушки от полного контракта (закрыть до демо):
- backend не считает DSP/trees/CNN — `events` всегда `[]`, score-полей нет;
- `registered_sensor` всегда `false` (перенос не валидирован);
- без loopback-Origin-проверок (контейнерная сеть, не браузер);
- `queued=true` означает приём в стрим, а не запись в CH (пишет ingester).

## Known gaps (пост-деплой)

Таблица выше заморожена (v1) и не меняется; ниже — что код уже умеет сверх неё.

- `GET /readyz` — есть в `stack/backend/app.py` (`PING` + `XLEN`), в таблице выше только `/healthz`; healthcheck в `stack/compose.yaml` тоже только `/healthz`.
- `STREAM_NO_TRIM=1` + `STREAM_BACKLOG_MAX` — есть в `stack/backend/app.py` (`stream_publish`, 503 при переполнении) и `XDEL` в `stack/ingester/ingest.py`, но в `stack/compose.yaml` не заданы (по умолчанию трим `STREAM_MAXLEN ~ 20000`).
- `profiles_v2` / `profile_conflicts` — созданы миграцией `stack/clickhouse/migrations/0001-profiles-v2.sql`, пишет `stack/ingester/ingest.py` (ветка `replay_id`); в разделе ClickHouse выше только `profiles_raw`/`events`/`models_registry`.
- `schema_version: 2` / replay (`replay_id`, `source_sha256`, `source_row`…) — принимает `stack/backend/app.py` (`/api/session`, `/api/infer`); в разделе «Профиль» выше только поля v1.
