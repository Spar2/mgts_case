# Ветка data для млщика

Стенд уже крутится на Xeon: проект `haklolarp-stack`, каталог `/opt/haklolarp-stack`. Эта страница — что туда класть и как понять, что прошло. Экрана импорта нет.

## Что коммитить в git

Только код и новая схема. Файл `stack/clickhouse/migrations/NNNN-имя.sql` в ветке `data`, затем пуш. Сервис `migrate` в `stack/compose.yaml` считает `SELECT count() FROM haklolarp.schema_migrations WHERE name = '…'`. Если count не `0`, пишет `migrate: skip` и файл второй раз не выполняет. Уже применённый файл миграции не переписывать: добавь новый.

`stack/clickhouse/init.sql` смонтирован в `/docker-entrypoint-initdb.d`. По документации образа ClickHouse скрипты оттуда выполняются после `initdb` и заканчивают инициализацию до старта сервиса. На живом томе этот файл заново не прогоняем. Ответ Context7 не содержит отдельной фразы, что уже инициализированная база инициализируется повторно.

## Что не коммитить

CSV в git не кладутся. Пуш CSV в ветку `data` строки в базу не загружает. CI эти файлы не читает.

Копировать на Xeon так:

```bash
mkdir -p /opt/haklolarp-stack/data
scp 1.csv 2.csv hakloboss@xeon-staging:/opt/haklolarp-stack/data/
```

Это каталог `DATA_DIR` из `stack/.env`.

## Как прогнать

Сервис `streamer` — профиль `manual`, сам не поднят. На Xeon:

```bash
cd /opt/haklolarp-stack
docker compose -f stack/compose.yaml run --rm streamer \
  --backend http://backend:8000 --csv /data/2.csv --limit 50 --delay 0.1
```

Полный прогон — `stack/stream-all.sh` из того же каталога (`/opt/haklolarp-stack`). Скрипт деструктивен (TRUNCATE таблиц CH + DEL стримов), требует `--yes`/`CONFIRM=1` и свободного flock-лока.

## Как смотреть, что прошло

Зелёный Actions workflow `Deploy data` и сообщение Discord значат только контейнеры и схему: новые миграции применились, стек поднялся. Число строк CSV они не проверяют.

Строки, с Xeon:

```bash
docker compose -f stack/compose.yaml exec clickhouse \
  clickhouse-client --query "SELECT count() FROM haklolarp.profiles_raw"
python3 stack/verify.py streams   # pending/lag
python3 stack/verify.py report    # записи/сегменты
```

Применённые файлы — таблица `haklolarp.schema_migrations`. Replay-записи с `replay_id` ingester кладёт в `profiles_v2` (конфликты — в `profile_conflicts`). В Redis, как в `DATA_GUIDE`:

```bash
docker compose -f stack/compose.yaml exec redis redis-cli
# в redis-cli:
XLEN dts:profiles
XINFO GROUPS dts:profiles
XRANGE dts:profiles - + COUNT 1
```

## Чего нет относительно web R3

Учётных записей, проекта, паспорта качества, проверки дубликата по SHA и экрана импорта на этом стенде нет. Это макеты на ветке `web` (`design/archive/carbon-r3`, исторический архив, не источник текущего UI), не сервис здесь.
