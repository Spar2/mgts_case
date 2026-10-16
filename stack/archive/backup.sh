#!/bin/bash
# АРХИВ. НЕ ЗАПУСКАТЬ. Перемещён в stack/archive/backup.sh (см. stack/archive/README.md).
# Restore-point бэкап стенда на Xeon. Запуск в tmux:
# tmux new -d -s backup -c /opt/haklolarp-stack bash stack/backup.sh ; cat /tmp/backup.log
set -u
exec > /tmp/backup.log 2>&1
cd /opt/haklolarp-stack || exit 1
STAMP=$(date +%F-%H%M)
mkdir -p "$HOME/haklo-backup/$STAMP"
docker exec haklolarp-stack-clickhouse-1 clickhouse-client --query 'SELECT * FROM haklolarp.profiles_raw FORMAT Native' > "$HOME/haklo-backup/$STAMP/profiles_raw.native"
cp -r /opt/haklolarp-stack/stack/registry "$HOME/haklo-backup/$STAMP/registry"
cp /opt/haklolarp-stack/stack/compose.yaml "$HOME/haklo-backup/$STAMP/compose.yaml"
ls -la "$HOME/haklo-backup/$STAMP"
python3 stack/verify.py report
echo BACKUP_DONE
