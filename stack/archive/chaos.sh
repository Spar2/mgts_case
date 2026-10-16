#!/bin/bash
# АРХИВ. НЕ ЗАПУСКАТЬ. Перемещён в stack/archive/chaos.sh (см. stack/archive/README.md).
# Хаос-тесты стенда на Xeon. Запуск в tmux: tmux new -d -s chaos -c /opt/haklolarp-stack bash stack/chaos.sh
# Опрос: tmux capture-pane -p -t chaos ; лог: /tmp/chaos.log
# Сценарий:
#   1. TRUNCATE, полный 2.csv фоном, kill -9 ingester mid-drain, рестарт.
#      Критерий: CH-count(2.csv) >= 696 (потерянных нет), pending=0.
#      count > 696 = наблюдаемая ределивери (дубли) — ожидаемо, at-least-once.
#   2. Рестарт redis: XLEN до == XLEN после (AOF-реплей), группа на месте.
set -u
exec > /tmp/chaos.log 2>&1
cd /opt/haklolarp-stack || exit 1
echo "=== truncate ==="
python3 stack/verify.py truncate
echo "=== streamer 2.csv background ==="
docker compose -f stack/compose.yaml run -d --name chaos-streamer streamer \
  --backend http://backend:8000 --csv /data/2.csv --limit 1000000 --delay 0
sleep 45
echo "=== kill -9 ingester ==="
docker kill haklolarp-stack-ingester-1
sleep 15
echo "=== start ingester ==="
docker start haklolarp-stack-ingester-1
echo "=== wait streamer ==="
docker wait chaos-streamer
docker rm chaos-streamer
echo "=== drain 240s (claim-цикл 120s idle) ==="
sleep 240
echo "=== report ==="
python3 stack/verify.py report
python3 stack/verify.py streams
echo "=== redis restart / AOF ==="
docker exec haklolarp-stack-redis-1 redis-cli XLEN dts:profiles
docker restart haklolarp-stack-redis-1
sleep 20
docker exec haklolarp-stack-redis-1 redis-cli XLEN dts:profiles
python3 stack/verify.py streams
echo CHAOS_DONE
