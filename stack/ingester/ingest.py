"""Ingester LAN-стенда: Redis Stream -> ClickHouse.

Читает группу `ingesters` стрима профилей (XREADGROUP), батчем пишет в CH,
подтверждает XACK. Неподтверждённое остаётся в PEL; раз в минуту XAUTOCLAIM
забирает зависшие записи себе (пережил рестарт с новым именем консьюмера).

Семантика at-least-once: крэш между INSERT и XACK даст дубли в CH
(дедупликации нет — см. CONTRACTS.md). Порядок внутри одного инжестера
сохранён; второго консьюмера пока нет.

Переменные окружения:
  REDIS_HOST/REDIS_PORT, STREAM_PROFILES (default dts:profiles),
  GROUP (default ingesters), CH_URL (default http://clickhouse:8123),
  BATCH (default 50), BLOCK_MS (default 5000),
  CLAIM_IDLE_MS (default 120000), CLAIM_INTERVAL_S (default 60).
"""

import json
import os
import signal
import socket
import sys
import time

sys.path.insert(0, "/opt/minilib")
from chutil import ch_insert, ch_query, ch_insert_native
from miniresp import MiniResp, RedisError

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
STREAM = os.environ.get("STREAM_PROFILES", "dts:profiles")
GROUP = os.environ.get("GROUP", "ingesters")
CH_URL = os.environ.get("CH_URL", "http://clickhouse:8123")
BATCH = int(os.environ.get("BATCH", "50"))
BLOCK_MS = int(os.environ.get("BLOCK_MS", "5000"))
CLAIM_IDLE_MS = int(os.environ.get("CLAIM_IDLE_MS", "120000"))
CLAIM_INTERVAL_S = int(os.environ.get("CLAIM_INTERVAL_S", "60"))
HEARTBEAT = "/tmp/ingester.heartbeat"

STOP = False
CONSUMER = f"{socket.gethostname()}-{os.getpid()}"


def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"{ts} ingester {msg}", flush=True)


def beat():
    with open(HEARTBEAT, "w", encoding="utf-8") as f:
        f.write(str(time.time()))


def on_term(_signum, _frame):
    global STOP
    STOP = True
    log("SIGTERM, finishing batch")


def connect():
    while not STOP:
        try:
            r = MiniResp(REDIS_HOST, REDIS_PORT, timeout=15.0)
            r.ping()
            return r
        except (ConnectionError, OSError, RedisError) as e:
            log(f"redis unavailable: {e}, retry in 5s")
            time.sleep(5)
    raise SystemExit("stopped before connect")


def ensure_group(r):
    r.xgroup_create(STREAM, GROUP, start="0", mkstream=True)


def handle_batch(r, batch):
    """Батч в CH + ACK. Битые записи (не наш продюсер/порча) уходят
    в dead-letter: ACK сразу + лог, без вечных ретраев.
    Ошибка CH — ни одного ACK, весь батч перечитается (at-least-once)."""
    rows = []
    ids = []
    dead = []
    for entry_id, fields in batch:
        try:
            if fields.get("kind") != "profile":
                raise ValueError(f"unknown kind: {fields.get('kind')!r}")
            rows.append(json.loads(fields["payload"]))
            ids.append(entry_id)
        except (ValueError, KeyError, TypeError) as e:
            log(f"dead-letter id={entry_id}: {e}")
            dead.append(entry_id)
    if rows:
        native=[row for row in rows if "replay_id" in row]
        legacy=[row for row in rows if "replay_id" not in row]
        if legacy: ch_insert(CH_URL,"profiles_raw",legacy)
        if native:
            safe=[];seen={}
            for row in native:
                key=(row["replay_id"],row["row_idx"])
                if key not in seen:
                    previous=ch_query(CH_URL,"SELECT payload_sha256 FROM haklolarp.profiles_v2 FINAL WHERE replay_id={replay:String} AND row_idx={row:UInt32} LIMIT 1",{"replay":key[0],"row":key[1]})
                    seen[key]=previous[0]["payload_sha256"] if previous else row["payload_sha256"]
                if seen[key]!=row["payload_sha256"]:
                    ch_insert(CH_URL,"profile_conflicts",[{"replay_id":key[0],"row_idx":key[1],"payload_sha256":row["payload_sha256"],"reason":"Logical profile identity reused with different content"}])
                else: safe.append(row)
            if safe: ch_insert_native(CH_URL,safe)
    done = ids + dead
    if done:
        n = r.xack(STREAM, GROUP, *done)
        if os.environ.get("STREAM_NO_TRIM")=="1": r.execute("XDEL",STREAM,*done)
        log(f"stored={len(rows)} dead={len(dead)} acked={n} last={done[-1]}")


def claim_stale(r):
    cursor = "0-0"
    claimed_total = 0
    while not STOP:
        cursor, batch, _deleted = r.xautoclaim(
            STREAM, GROUP, CONSUMER, CLAIM_IDLE_MS, start=cursor, count=BATCH
        )
        if batch:
            handle_batch(r, batch)
            claimed_total += len(batch)
        if cursor == "0-0":
            break
    if claimed_total:
        log(f"reclaimed={claimed_total}")


def main():
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    log(f"start consumer={CONSUMER} stream={STREAM} group={GROUP}")
    r = connect()
    ensure_group(r)
    last_claim = 0.0
    beat()
    while not STOP:
        try:
            batch = r.xreadgroup(
                GROUP, CONSUMER, STREAM, count=BATCH, block_ms=BLOCK_MS
            )
        except (ConnectionError, OSError, RedisError) as e:
            log(f"read failed: {e}, reconnecting")
            r.close()
            r = connect()
            ensure_group(r)
            continue
        if batch:
            try:
                handle_batch(r, batch)
            except Exception as e:  # noqa: BLE001 - CH down: leave in PEL, retry
                log(f"batch failed, will retry: {e}")
                time.sleep(5)
        now = time.time()
        if now - last_claim >= CLAIM_INTERVAL_S:
            try:
                claim_stale(r)
            except (ConnectionError, OSError, RedisError) as e:
                log(f"claim failed: {e}")
            last_claim = now
        beat()
    log("stopped")


if __name__ == "__main__":
    main()
