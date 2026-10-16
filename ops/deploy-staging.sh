#!/bin/bash
# Долгий деплой staging (main) внутри tmux (переживает обрыв SSH).
# Вызывает fire-шаг workflow: tmux new-session -d ... "bash ops/deploy-staging.sh <sess>"
# Код выхода скрипта fire-шаг забирает из .deploy-<sess>.done (см. finish-шаг).
# Использование: bash ops/deploy-staging.sh deploy-staging-<run_id>
set -eu
SESS="${1:?usage: deploy-staging.sh <tmux-session>}"
DIR=/opt/haklolarp-staging

cd "$DIR"
git fetch origin
git checkout main
git pull --ff-only origin main
docker compose pull
docker compose up -d --remove-orphans --wait
docker ps
