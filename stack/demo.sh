#!/bin/bash
# demo.sh — стенд с нуля + smoke. Запуск: bash stack/demo.sh
# из корня клона (Xeon: /opt/haklolarp-stack). Скрипт сам переходит в stack/.
# CSV берёт DATA_DIR из stack/.env (/opt/haklolarp-stack/data), не ./data
# рядом с compose по умолчанию.
# Только 2.csv (контроль, 696 строк), полный прогон — флагом FULL=1.
set -e
cd "$(dirname "$0")" || exit 1

docker compose up -d --build

echo '-- wait backend'
for _ in $(seq 1 60); do
  if curl -sf 127.0.0.1:8000/api/manifest >/dev/null; then break; fi
  sleep 2
done
curl -sf 127.0.0.1:8000/api/manifest

LIMIT=50
if [ "${FULL}" = "1" ]; then LIMIT=1000000; fi
docker compose run --rm streamer \
  --backend http://backend:8000 --csv /data/2.csv \
  --limit "$LIMIT" --delay 0.05

echo '-- profiles in CH:'
curl -sf '127.0.0.1:8123/?query=SELECT%20count()%20FROM%20haklolarp.profiles_raw'
echo
