#!/bin/bash
# Хаос битым сообщением. Запуск в tmux на Xeon:
# tmux new -d -s poison -c STACK_DIR bash stack/poison.sh --yes ; cat /tmp/poison.log
# Критерий: dead-letter в логе, pending=0, CH-count не изменился, цикл не завис.
# Скрипт лежит в stack/ и работает из собственного каталога.
# Лог отделённого прогона: /tmp/poison.log.
set -euo pipefail
exec > /tmp/poison.log 2>&1
exec 9>/tmp/poison.lock
flock -n 9 || { echo ALREADY_RUNNING; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
STACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "STACK_DIR=$STACK_DIR"
# DATA-GUARD: по умолчанию DRY-RUN refuse — прогон пишет битые записи в стрим.
CONFIRMED=0
for a in "$@"; do
  if [ "$a" = "--yes" ]; then CONFIRMED=1; fi
done
if [ "${CONFIRM:-0}" = "1" ]; then CONFIRMED=1; fi
if [ "$CONFIRMED" -ne 1 ]; then
  echo "DRY-RUN refuse: poison.sh injects broken messages into dts:profiles." >&2
  echo "Re-run with --yes or CONFIRM=1." >&2
  exit 2
fi
docker compose -f compose.yaml up -d --build ingester
# Readiness-poll вместо фиксированного sleep: Redis PING + CH SELECT 1.
READY=0
for _ in $(seq 1 60); do
  if docker exec haklolarp-stack-redis-1 redis-cli PING >/dev/null 2>&1 \
    && docker exec haklolarp-stack-clickhouse-1 clickhouse-client --query 'SELECT 1' >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [ "$READY" -ne 1 ]; then
  echo "ASSERT_FAIL: stack not ready (redis/CH poll timeout)" >&2
  exit 1
fi
BEFORE=$(docker exec haklolarp-stack-clickhouse-1 clickhouse-client --query 'SELECT count() FROM haklolarp.profiles_raw')
echo "CH_BEFORE=$BEFORE"
docker exec haklolarp-stack-redis-1 redis-cli XADD dts:profiles MAXLEN '~' 20000 '*' kind profile payload '{broken json'
docker exec haklolarp-stack-redis-1 redis-cli XADD dts:profiles MAXLEN '~' 20000 '*' kind alien payload '{}'
# Drain-цикл: ждать pending==0 через miniresp xpending_summary вместо sleep.
DRAINED=0
for _ in $(seq 1 60); do
  PENDING="$(python3 -c 'import sys,os; sys.path.insert(0,os.path.join(os.getcwd(),"lib")); from miniresp import MiniResp,RedisError; r=MiniResp("127.0.0.1",6379,timeout=10.0); total=0
for s in ("dts:profiles","dts:events"):
    try: n,_,_,_=r.xpending_summary(s,"ingesters")
    except RedisError: continue
    total+=n
r.close(); print(total)' 2>/dev/null || echo '?')"
  echo "pending=$PENDING"
  if [ "$PENDING" = "0" ]; then
    DRAINED=1
    break
  fi
  sleep 2
done
if [ "$DRAINED" -ne 1 ]; then
  echo "ASSERT_FAIL: drain timeout, pending did not reach 0" >&2
  exit 1
fi
echo "--- ingester log ---"
INGESTER_LOG="$(docker logs haklolarp-stack-ingester-1 --tail 20)"
echo "$INGESTER_LOG"
echo "$INGESTER_LOG" | grep -q "dead-letter" || {
  echo "ASSERT_FAIL: no dead-letter in ingester log" >&2
  exit 1
}
AFTER=$(docker exec haklolarp-stack-clickhouse-1 clickhouse-client --query 'SELECT count() FROM haklolarp.profiles_raw')
echo "CH_AFTER=$AFTER"
if [ "$AFTER" != "$BEFORE" ]; then
  echo "ASSERT_FAIL: CH_AFTER=$AFTER != CH_BEFORE=$BEFORE" >&2
  exit 1
fi
python3 verify.py streams
echo POISON_DONE
