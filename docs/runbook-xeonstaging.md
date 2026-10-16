# Runbook xeonstaging (live, main)

Live: `/opt/haklolarp-web`, `http://127.0.0.1:8081` (только loopback).
Проект: `haklolarp-web`, pilot `compose.web.yaml` (+ `compose.platform.yaml --profile stream`).

## Деплой live
```bash
cd /opt/haklolarp-web
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse HEAD  # == origin/main
mkdir -p .runtime/records .runtime/queue .runtime/research-pack .runtime/imports
docker compose -f compose.web.yaml config -q
docker compose -f compose.web.yaml up -d --build --wait
docker compose -f compose.web.yaml ps
```

## Проверка
```bash
curl -fsS http://127.0.0.1:8081/ -o /dev/null
curl -fsS http://127.0.0.1:8081/api/v1/health
curl -fsS http://127.0.0.1:8081/api/v1/session
```
Ожидание: `200`, `{"status":"ok",...}`, `{"mutation_token":...}`.

## Откат
```bash
cd /opt/haklolarp-web
git checkout <предыдущий-SHA>  # из записи ниже
docker compose -f compose.web.yaml up -d --wait
curl -fsS http://127.0.0.1:8081/api/v1/health
```
Запрещено: `down -v` (сносит тома), правки вне PR, секреты в репо.

## Записи
- 2026-10-14Z: main live, pilot Healthy, gates green.
