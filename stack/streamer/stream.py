"""Mock-streamer: читает CSV записи и шлёт профили в backend /api/infer.

CSV: dts_id,channel_id,timestamp,points[,points_size] — points это JSON-строка
[[x_m, T], ...]. Ничего не чинит молча: битая строка = ошибка с номером строки.

  python3 stream.py --backend http://backend:8000 --csv /data/2.csv \
      --limit 50 --delay 0.1
"""

import argparse
import csv
import json
import sys
import time
import urllib.request

MAX_BODY = 2 * 1024 * 1024
# Профили 1.csv длиннее дефолтного лимита csv (128К).
csv.field_size_limit(sys.maxsize)


def call(method, url, token, obj=None):
    data = json.dumps(obj).encode() if obj is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-DTS-Token", token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--delay", type=float, default=0.1)
    ap.add_argument("--mode", default="measurement_replay")
    a = ap.parse_args()

    base = a.backend.rstrip("/")
    manifest = call("GET", base + "/api/manifest", None)
    token = manifest["mutation_token"]
    sess = call("POST", base + "/api/session", token, {"mode": a.mode})
    sid = sess["session_id"]
    print(f"session {sid} mode={sess['mode']}", flush=True)

    sent, ok, bad = 0, 0, 0
    # Сессия закрывается в finally: убитый mid-run стример иначе течёт
    # сессию до TTL (4 таких — и новые ранны ловят 409).
    try:
        with open(a.csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for key in ("dts_id", "channel_id", "timestamp", "points"):
                if key not in (reader.fieldnames or []):
                    raise SystemExit(f"CSV missing column: {key}")
            for lineno, row in enumerate(reader, start=2):
                if sent >= a.limit:
                    break
                try:
                    points = json.loads(row["points"])
                except json.JSONDecodeError as e:
                    print(f"line {lineno}: bad points JSON: {e}", flush=True)
                    bad += 1
                    continue
                profile = {
                    "dts_id": row["dts_id"],
                    "channel_id": row["channel_id"],
                    "timestamp": row["timestamp"],
                    "points": points,
                }
                try:
                    res = call(
                        "POST",
                        base + "/api/infer",
                        token,
                        {"session_id": sid, "profile": profile},
                    )
                except Exception as e:  # noqa: BLE001 - show backend error, keep going
                    print(f"line {lineno}: backend: {e}", flush=True)
                    bad += 1
                    continue
                sent += 1
                if res.get("queued"):
                    ok += 1
                else:
                    bad += 1
                    print(
                        f"line {lineno}: not queued: {res}",
                        flush=True,
                    )
                if a.delay > 0:
                    time.sleep(a.delay)
    finally:
        try:
            call("POST", base + "/api/session/close", token, {"session_id": sid})
        except Exception as e:  # noqa: BLE001 - close best-effort
            print(f"session close failed: {e}", flush=True)
    print(f"sent={sent} queued={ok} bad={bad}", flush=True)
    if sent == 0:
        raise SystemExit("nothing sent")


if __name__ == "__main__":
    main()
