# Data-стенд: доступ, ClickHouse, Redis, данные, пробуждение

Изолированный LAN-стенд ветки `data` на Xeon. ClickHouse и Redis живут
**только в контейнерах** (`stack/compose.yaml`), на хост через apt/systemd
их не ставить. Деплой — каноном `.github/workflows/deploy-reusable.yml@main`
через `deploy-data.yml` (`mode: plain`, каталог `/opt/haklolarp-stack`).

## 1. Что где живёт

| Что | Где |
| --- | --- |
| Клон репо (ветка `data`) | `/opt/haklolarp-stack` на Xeon |
| Compose-файл | `/opt/haklolarp-stack/stack/compose.yaml` |
| Compose-проект | `haklolarp-stack` (`stack/.env`, пишет job `prepare`) |
| ClickHouse (HTTP) | `127.0.0.1:8123`, база `haklolarp` |
| Redis | `127.0.0.1:6379`, стрим `dts:profiles` |
| Backend API | `127.0.0.1:8000` |
| CSV-данные | `/opt/haklolarp-stack/data` (`DATA_DIR` из `stack/.env`) |

Слияние `data` в `main` — только через PR. Пуш в `main` ClickHouse не обновляет:
деплой данных включён лишь на ветку `data`. В `web` и `ml` эту ветку не мержить.
Одновременный деплой `data` и `ml` делит RAM Xeon (Redis до 4 ГБ, ClickHouse 3 ГБ);
порты у них разные.

Для млщика, что класть и как смотреть результат: [stack/FOR_ML.md](FOR_ML.md).

Пуш в `data` обновляет этот стек. Новая схема — новым файлом в
`stack/clickhouse/migrations/`; `init.sql` существующий том не переигрывает.
Job `prepare` один раз отодвигает старое scp-дерево в
`/opt/haklolarp-stack.prev` и кладёт на его место git-клон. Тома проекта
`haklolarp-stack` при этом не удаляются. `docker compose up` делает следующий
job, канонный plain. Без секрета `STAGING_BECOME_PASSWORD` (sudo на Xeon)
`prepare` останавливается — секрет задать и перезапустить workflow через
`workflow_dispatch`.

Порты слушают только loopback. Снаружи — либо SSH на хост, либо SSH-туннель:
`ssh -L 8123:127.0.0.1:8123 -L 8000:127.0.0.1:8000 hakloboss@xeon-staging`.

## 2. Как зайти на машину

- С админской машины: `ssh hakloboss@xeon-staging` (tailnet), либо
  `ssh hakloboss@100.66.240.22`, либо `ssh hakloboss@192.168.1.123` (LAN).
- Через дев-ВМ: `ssh gritsko@192.168.1.20`, дальше на Xeon.
- `sudo`-пароль и пароль `root` на KVM — только у админа по памяти
  (см. `ansible/KEYS.md`, раздел «Пароли»).

## 3. Как разбудить Xeon вручную

1. **NanoKVM** (LAN): веб-UI `http://192.168.1.4` → кнопка Power (короткое
   нажатие). Reset/ISO — не трогать без необходимости.
2. **Wake-on-LAN** из LAN: MAC `00:e0:26:5c:23:20`, broadcast
   `192.168.1.255`, UDP-порт 9 (с дев-ВМ: `wakeonlan 00:e0:26:5c:23:20`).
3. **CI** (после пуша ветки `data`): `workflow_dispatch` workflow
   «Deploy data (staging)» — job `wake` сам будит хост через KVM и деплоит.
4. Форс-команда ключа `kvm_wake` умеет только будить (WoL + короткое нажатие
   Power), reset/консоль/ISO недоступны (см. `ansible/files/kvm/xeon-wake-only`).

## 4. ClickHouse

```bash
cd /opt/haklolarp-stack
docker compose -f stack/compose.yaml ps          # статус стека
docker compose -f stack/compose.yaml exec clickhouse clickhouse-client
```

Разовые запросы:

```bash
docker compose -f stack/compose.yaml exec clickhouse \
  clickhouse-client --query "SELECT count() FROM haklolarp.profiles_raw"
docker compose -f stack/compose.yaml exec clickhouse \
  clickhouse-client --query "SELECT * FROM haklolarp.profiles_raw LIMIT 5"
```

Свой SQL-файл (`my.sql` на хосте):

```bash
docker compose -f stack/compose.yaml exec -T clickhouse \
  clickhouse-client --multiquery < my.sql
```

Схема — `stack/clickhouse/init.sql` (таблицы `haklolarp.profiles_raw`,
`haklolarp.events`, `haklolarp.models_registry`). Файл выполняется
автоматически при первом создании тома `ch-data`; для существующего стенда
его можно выполнить вручную тем же `--multiquery` (там `IF NOT EXISTS`).

### Миграции (авто)

Положи `stack/clickhouse/migrations/NNNN-имя.sql` в ветку `data` и запушь —
канонный деплой применит только новые файлы и запишет их в
`haklolarp.schema_migrations` (уже применённые пропускаются; менять их нельзя,
добавляй новый файл). Раннер — одноразовый сервис `migrate` в
`stack/compose.yaml` (скрипт инлайн). Для ad-hoc применения без деплоя —
команда `--multiquery` выше.

## 5. Redis

```bash
docker compose -f stack/compose.yaml exec redis redis-cli
# в redis-cli:
XLEN dts:profiles
XINFO GROUPS dts:profiles
XRANGE dts:profiles - + COUNT 1
```

`dts:profiles` — принятые backend профили (пишет ingester в ClickHouse).
По умолчанию compose режет стрим тримом `STREAM_MAXLEN ~ 20000`; `STREAM_NO_TRIM` в compose не задан.
Перед каноническим replay стрим сбрасывается (`DEL dts:profiles` это делает
`stack/stream-all.sh`), иначе повторы копятся до OOM. Аутентификации у
Redis/CH нет осознанно: только loopback, стенд LAN.

## 6. Данные (CSV) и прогон стримера

CSV в git **не коммитятся** (оно и в `.gitignore`). Только `scp`:

```bash
mkdir -p /opt/haklolarp-stack/data
scp 1.csv 2.csv hakloboss@xeon-staging:/opt/haklolarp-stack/data/
```

Прогон (сервис `streamer` — профиль `manual`, по умолчанию не поднят):

```bash
cd /opt/haklolarp-stack
docker compose -f stack/compose.yaml run --rm streamer \
  --backend http://backend:8000 --csv /data/2.csv --limit 50 --delay 0.1
```

Проверка:

```bash
docker compose -f stack/compose.yaml exec clickhouse \
  clickhouse-client --query "SELECT count() FROM haklolarp.profiles_raw"
python3 stack/verify.py streams   # pending/lag
python3 stack/verify.py report    # записи/сегменты
```

## 7. Бэкап (вручную; `stack/backup.sh` заточен под Xeon `/opt/haklolarp-stack`)

```bash
cd /opt/haklolarp-stack
STAMP=$(date +%F-%H%M); mkdir -p ~/haklo-backup/$STAMP
docker exec haklolarp-stack-clickhouse-1 clickhouse-client --query 'SELECT * FROM haklolarp.profiles_raw FORMAT Native' > ~/haklo-backup/$STAMP/profiles_raw.native
cp -r /opt/haklolarp-stack/stack/registry ~/haklo-backup/$STAMP/registry
cp /opt/haklolarp-stack/stack/compose.yaml ~/haklo-backup/$STAMP/compose.yaml
```

## 8. Правила

- Никаких CSV в репозитории и никаких секретов в коммитах.
- ClickHouse/Redis — только контейнерами; apt/systemd на хост запрещён.
- Стенд = LAN/staging, не прод: данные воспроизводимы, ошибки лечим повторным
  прогоном (см. `stack/CONTRACTS.md`).
