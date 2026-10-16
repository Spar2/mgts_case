#!/bin/bash
# Долгий деплой ml внутри tmux (переживает обрыв SSH).
# Вызывает fire-шаг workflow: tmux new-session -d ... "bash ops/deploy-ml.sh <sess>"
# Код выхода скрипта fire-шаг забирает из .deploy-<sess>.done (см. finish-шаг).
# Значения продублированы из env workflow (STAGING_DIR/BRANCH/PROJECT/COMPOSE_FILE/APP_PORT).
# Использование: bash ops/deploy-ml.sh deploy-ml-<run_id>
set -eu
export GIT_SSH_COMMAND="ssh -i ~/.ssh/haklolarp_deploy -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts"
SESS="${1:?usage: deploy-ml.sh <tmux-session>}"
DIR=/opt/haklolarp-ml
DONE="$DIR/.deploy-$SESS.done"

cd "$DIR"
git fetch origin
git checkout ml
git pull --ff-only origin ml
docker compose -p haklolarp-ml -f compose.ml.yaml up -d --build --remove-orphans
docker ps
# Холодный старт uvicorn с ML-зависимостями дольше 0 секунд — ждём по циклу.
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:${APP_PORT:-8082}/health; then break; fi
  sleep 5
done
curl -fsS http://127.0.0.1:${APP_PORT:-8082}/health
