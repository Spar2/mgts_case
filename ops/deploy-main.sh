#!/bin/bash
# Долгий деплой main внутри tmux (переживает обрыв SSH).
# Вызывает fire-шаг workflow: tmux new-session -d ... "bash ops/deploy-main.sh <sess>"
# Код выхода скрипта fire-шаг забирает из .deploy-<sess>.done (см. finish-шаг).
# Порт берётся из APP_PORT (default 8081); остальные значения продублированы из env workflow (STAGING_DIR/BRANCH/PROJECT/COMPOSE_FILE).
# Разворачивает live main из compose.web.yaml и проверяет здоровье API.
# Использование: bash ops/deploy-main.sh deploy-main-<run_id>
set -eu
export GIT_SSH_COMMAND="ssh -i ~/.ssh/haklolarp_deploy -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts"
SESS="${1:?usage: deploy-main.sh <tmux-session>}"
DIR=/opt/haklolarp-web

cd "$DIR"
git fetch origin
git checkout main
git pull --ff-only origin main
# Live всегда с platform (postgres/redis/worker): дефолт true.
# Без DTS_POSTGRES_PASSWORD скрипт падает с понятной ошибкой (см. :? ниже).
# Web-only пилот — через ops/deploy-web.sh, этот скрипт его не касается.
compose_args=(-p haklolarp-web -f compose.web.yaml)
if [ "${DTS_PLATFORM_ENABLED:-true}" = true ]; then
  # Пароль берёт compose из $DIR/.env (project-dir), не из shell-env CI.
  grep -q '^DTS_POSTGRES_PASSWORD=.\+' .env || { echo "DTS_POSTGRES_PASSWORD missing in $DIR/.env" >&2; exit 1; }
  compose_args+=(-f compose.platform.yaml)
  if [ "${DTS_STREAM_ENABLED:-false}" = true ]; then compose_args+=(--profile stream); fi
fi
docker compose "${compose_args[@]}" config -q
docker compose "${compose_args[@]}" up -d --build --wait --remove-orphans
curl --fail --max-time 10 http://127.0.0.1:${APP_PORT:-8081}/api/v1/health
if [ "${DTS_PLATFORM_ENABLED:-true}" = true ]; then
  curl --fail --max-time 10 http://127.0.0.1:${APP_PORT:-8081}/api/v2/health/ready
fi
