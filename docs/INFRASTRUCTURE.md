# Инфраструктура haklolarp: как всё работает, разворачивается, автоматизируется

Живое состояние: оба домена 200, tailnet-URL 200, backend :8443 healthy,
CI Xeon тикает, VPS-timer 90с, бэкап LE снят.

## 1. Edge: домены и вход

- Домены: `xn--80aubodcdd.xn--p1ai` (основной) + `larpcorp.ru` (алиас); `www.` обоих — только редирект на apex (`redir ... permanent` в `ops/dual-access/Caddyfile`).
- VPS — единственная точка входа из интернета: наружу слушают только `80/TCP` + `443/TCP` (Caddy) и `8080/TCP` (заглушка); всё остальное — только loopback/tailnet.
- Caddy на VPS — TLS-терминация с авто-LE: запущен как `docker run` с volume `caddy_data` (`/data` — сертификаты/ACME-state); конфиг — `ops/dual-access/Caddyfile`.
- Апстрим Caddy: `reverse_proxy https://<XEON-TAILNET-IP>:8443` с `header_up Host {host}` и `tls_insecure_skip_verify` (за nginx — приватный/self-signed серт, проверка не нужна внутри tailnet).
- Xeon: `ops/dual-access/xeon-proxy-8443.service` — `socat TCP-LISTEN:8443,bind=<XEON-TAILNET-IP>,fork TCP:127.0.0.1:8443`; tailnet-IP слушает socat, контейнер — только `127.0.0.1:${DTS_TLS_PORT:-8443}:443` (`compose.production.yaml:4`).
- Nginx (`ops/web/nginx-production.conf.template`, 46 строк): `:80` — `server_name ${DTS_PUBLIC_HOST}`, `/api/v1/health` — прокси, остальное — `return 308 ${DTS_PUBLIC_ORIGIN}$request_uri`; `:443 ssl` — тот же `server_name`, `fullchain.pem/privkey.pem`, `TLSv1.2 TLSv1.3`.
- Лимиты/заголовки nginx: `client_max_body_size 64k` (только `= /api/v1/results` — `256k`), `proxy_read_timeout 300s`, `proxy_buffering off`, `X-Forwarded-Proto https`, `X-Content-Type-Options/Referrer-Policy/X-Frame-Options`.
- Host/Origin-чеки: `DTS_PUBLIC_HOST/DTS_PUBLIC_ORIGIN` обязательны; `api` сверяет `DTS_ALLOWED_HOSTS` (`HOST,EXTRA_HOSTS,127.0.0.1,localhost`) и `DTS_ALLOWED_ORIGINS`, траст прокси — только `FORWARDED_ALLOW_IPS=172.30.89.10` (nginx).

## Оглавление
- 1. Edge (выше)
- 2. Автоматизация: CI и мониторинг — см. `vps-notes/ci-xeon.md`, ранбук, `ops/ci/`, `ops/dual-access/monitor-*`, `.github/workflows/monitor-dual-access.yml`
- 3. Данные и бэкенд — см. `docs/web/PRODUCTION-OPERATOR.md` (бэкап/метрики/миграции) + `stack/CONTRACTS.md` (контракты стенда)

## 2. Автоматизация: CI и мониторинг

- **Xeon CI (основной):** `ops/ci/haklolarp-ci.timer` — каждые 5 мин (`OnBootSec=2min`, `Persistent=true`) → `haklolarp-ci.service` (oneshot, `User=hakloboss`, `ExecStart=/opt/ci/run/poll-and-run.sh` → `python3 /opt/ci/poll/poller.py`, лог — journal).
- **`ops/ci/poll/poller.py` (stdlib only):** bare-mirror `/opt/ci/repos/haklolarp.git`, `fetch origin main web` по SSH read-only deploy-ключу (`StrictHostKeyChecking=yes`); сверка `rev-parse origin/<branch>` с `/opt/ci/state.json` — без изменений no-op, иначе `/opt/ci/run/run.sh <branch> <sha>` и запись `sha/rc/at` даже при красном.
- **`ops/ci/run/run.sh`:** worktree `/opt/ci/run/wt-<SHA>` (`trap`: cleanup + notify); гейты в `GRC`: `pip install requirements-dev.txt`, `ruff check .`, `yamllint` workflows/compose, `pytest`, `docker compose config`; деплой в `RC`: staging `/opt/haklolarp-staging` — всегда (`pull --ff-only` + `up -d --wait` + `curl :8080/`); web `/opt/haklolarp-web` — только `branch=web && GRC=0` (3 compose-файла + `curl :8081/api/v1/health`).
- **Уведомление:** `notify()` в `run.sh` читает `/opt/ci/ntfy.env` (`TOPIC=<name>`) и шлёт `branch/sha7/deploy/gates` в `ntfy.sh`; без файла — молча.
- **VPS-таймер dual-access:** `ops/dual-access/monitor-dual-access.timer` — каждые 90 с → oneshot-сервис (`EnvironmentFile=/etc/dual-access/monitor.env` 0600, секрета в репо нет); `monitor-dual-access.sh` проверяет `TAILNET_BASE/ → 200` + `PUBLIC_BASE/api/v1/health`, счётчик `/tmp/monitor-dual-access.fails`, 2 фейла подряд → `POST $MONITOR_WEBHOOK` (Discord), без webhook — только лог.
- **CI-cron (страховка после биллинга):** `.github/workflows/monitor-dual-access.yml` — `cron: "*/15 * * * *"` + `workflow_dispatch`; fail-job идёт через `notify-runs` → Discord.
- **Ручной регламент:** `vps-notes/manual-deploy.md` (вариант B): staging `main → :8080`, web `main → :8443` (`pull --ff-only` + `up -d --wait` + health-check), preview `9000+N`, откат `checkout <prev-SHA> + up -d` (запрещены `down -v`, правки мимо PR, секреты в git).

## 3. Данные и бэкенд

- **Postgres / Redis:** только внутренний контур; наружу — лишь loopback `127.0.0.1:${DTS_PG_PORT:-5434}:5432`, у Redis публикуемых портов нет. Healthcheck-и: `pg_isready -U workbench -d workbench` и `redis-cli ping` (`interval 5s`, `retries 10–20`); `api`/`worker`/`stream-backend`/`ingester` стартуют по `service_healthy`. `redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy noeviction`; тома `platform-db:/var/lib/postgresql/data`, `platform-redis:/data`. Без PG/worker/очереди пути аккаунтов, кейсов, заданий и control отдают `503`.
- **Маунты runtime/artifacts:** `workbench-runtime:/runtime` — единственный `rw` (артефакты, массивы `runtime/artifacts/arrays`, очередь заданий); всё входное — `ro`: `${DTS_RECORDS}: /data/records:ro`, `${DTS_QUEUE}:/data/queue:ro`, `${DTS_RESEARCH_PACK}:/data/research:ro`, `${DTS_IMPORT_ROOT}:/imports:ro`. Сохраняйте весь runtime-том.
- **Backup-job:** `workbench.backup_job --daily` — проверка раз в час по UTC, одна копия в день, хранит 7 последних **успешно зашифрованных** `dts-YYYY-MM-DD.tar.age` (age, получатель `age1…`; приватный ключ только у оператора, вне сервера). Plaintext только в tmpfs. Входит: проекты, аккаунты с хэшами, членства, документы, ревизии, идемпотентность, аудит, задачи, заявки, приглашения/восстановления; сессии не входят; исходные CSV/NPY и ClickHouse не входят.
- **Миграции и очереди:** аддитивные, транзакционно под advisory lock схемы. Квоты: 1 queued/running на пользователя; 2 исполняемых + 10 ожидающих на проект. Тот же ключ идемпотентности — возврат задачи, другое тело — `409`.
- **Метрики:** `GET /api/v2/metrics` (только admin): счётчики/статусы, buckets задержек, состояния очереди, возраст копии. OTLP — через `OTEL_EXPORTER_OTLP_ENDPOINT`. Логи без паролей/cookie/токенов.
- **ML-линк:** `DTS_ML_URL` (по умолчанию `http://host.docker.internal:8082`). Без ML — деградация: `503 "ML service is unavailable. Raw data remains available"`.
- **Лимиты:** тело `64 KiB`, только `POST /api/v1/results` — `256 KiB`. Перегрузка — `429 + Retry-After: 10`; реплей — макс. 8 сессий; анализ — один за раз.
