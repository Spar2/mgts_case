#!/bin/bash
# stream-all.sh — полный прогон всех 6 записей через стрим. Запуск:
#   tmux kill-session -t stream-all 2>/dev/null; tmux new -d -s stream-all -c /opt/haklolarp-stack bash stack/stream-all.sh --yes
# из корня клона (Xeon: /opt/haklolarp-stack). Скрипт сам переходит в stack/.
# Опрос: tmux capture-pane -p -t stream-all. Guard: строго один паблишер
# (два параллельных дали +409 дублей на дев-ВМ).
# CSV берёт DATA_DIR из stack/.env (/opt/haklolarp-stack/data), не ./data рядом с compose.
# Деструктивен: TRUNCATE таблиц CH + DEL dts:* — только с --yes или CONFIRM=1 (иначе DRY-RUN refuse).
# Порядок for f 2 4 3 1 6 5 зафиксирован, не сортировать. Лог/лок: /tmp/stream-all.log, /tmp/stream-all.lock.
set -euo pipefail
exec > /tmp/stream-all.log 2>&1
exec 9>/tmp/stream-all.lock
flock -n 9 || { echo ALREADY_RUNNING; exit 1; }
cd "$(dirname "$0")" || exit 1
# DATA-GUARD: по умолчанию DRY-RUN refuse — прогон трёт таблицы CH и DEL-ит стримы.
CONFIRMED=0
for a in "$@"; do
  if [ "$a" = "--yes" ]; then CONFIRMED=1; fi
done
if [ "${CONFIRM:-0}" = "1" ]; then CONFIRMED=1; fi
if [ "$CONFIRMED" -ne 1 ]; then
  echo "DRY-RUN refuse: stream-all.sh wipes CH tables (truncate) + DEL dts:* streams." >&2
  echo "Re-run with --yes or CONFIRM=1." >&2
  exit 2
fi
# Preflight: состояние стримов до любых деструктивных шагов.
echo "=== streams preflight ==="
python3 verify.py streams
# truncate сам требует --yes и preflight pending==0 (exit 3).
python3 verify.py truncate --yes
# Стрим — растущий лог: без DEL каждый replay складывается поверх
# (3 повтора упёрлись в maxmemory/503). Свежий стрим на каждый канон.
# Группа пересоздастся инжестером (start 0 — ничего не потеряется).
docker exec haklolarp-stack-redis-1 redis-cli DEL dts:profiles
for f in 2 4 3 1 6 5; do
  echo "=== $f.csv ==="
  # DATA_DIR задаёт stack/.env. Без него compose взял бы ./data рядом с собой.
  docker compose -f compose.yaml run --rm --build streamer \
    --backend http://backend:8000 --csv /data/$f.csv --limit 1000000 --delay 0
done
echo "=== report ==="
python3 verify.py report
echo "=== streams ==="
python3 verify.py streams
echo ALLDONE
