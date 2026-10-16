#!/bin/bash
# Smoke backend после security-фикса. Запуск в tmux на Xeon:
# tmux new -d -s smoke -c STACK_DIR bash stack/smoke.sh ; cat /tmp/smoke.log
# Недеструктивен: --yes не требуется (принимается и игнорируется с hint).
# Скрипт лежит в stack/ и работает из собственного каталога.
# Лог отделённого прогона: /tmp/smoke.log.
set -euo pipefail
exec > /tmp/smoke.log 2>&1
exec 9>/tmp/smoke.lock
flock -n 9 || { echo ALREADY_RUNNING; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
STACK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "STACK_DIR=$STACK_DIR"
# Недеструктивен: --yes принимать и игнорировать с hint.
for a in "$@"; do
  if [ "$a" = "--yes" ]; then
    echo "hint: smoke.sh is non-destructive; --yes ignored." >&2
  fi
done
docker compose -f compose.yaml up -d --build backend
# Poll /api/manifest вместо фиксированного sleep.
READY=0
for _ in $(seq 1 60); do
  if python3 -c "import json,urllib.request; json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/manifest', timeout=5))" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [ "$READY" -ne 1 ]; then
  echo "ASSERT_FAIL: backend /api/manifest poll timeout" >&2
  exit 1
fi
TOKEN=$(python3 -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/manifest'))['mutation_token'])")
echo "TOKEN_OK"
python3 - "$TOKEN" <<'PYEOF'
import json
import sys
import urllib.request

token = sys.argv[1]
BASE = "http://127.0.0.1:8000"


def call(method, path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("X-DTS-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode()
    except Exception as e:
        body = ""
        try:
            body = e.read().decode()
        except Exception:
            pass
        return getattr(e, "code", "?"), body


code, out = call("POST", "/api/session", [1, 2, 3])
print(f"list-body -> {code} {out}")
assert code == 400, "non-dict body must be 400"
code, out = call("POST", "/api/session", {"mode": "live"})
print(f"session -> {code} {out[:80]}")
assert code == 200, "session create must be 200"
sid = json.loads(out)["session_id"]
prof = {"dts_id": "smoke", "channel_id": "ch1", "timestamp": "2026-01-01T00:00:00+00:00", "points": [[0, 20.0], [1, 21.0]]}
code, out = call("POST", "/api/infer", {"session_id": sid, "profile": prof})
print(f"infer -> {code} {out[:120]}")
assert code == 200, "infer must be 200"
body = json.loads(out)
assert body.get("queued") is True, "infer must be queued"
assert body.get("stream_id"), "infer must return stream_id"
code, out = call("POST", "/api/infer", {"session_id": sid, "profile": {}})
print(f"infer-bad-body -> {code} {out[:80]}")
assert code == 400, "bad profile must be 400"
code, out = call("POST", "/api/infer", {"session_id": "no-such-session", "profile": prof})
print(f"infer-unknown-session -> {code} {out[:80]}")
assert code == 404, "unknown session must be 404"
req = urllib.request.Request(BASE + "/api/session", data=json.dumps({"mode": "live"}).encode(), method="POST")
req.add_header("Content-Type", "application/json")
req.add_header("X-DTS-Token", "bad")
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        code, out = r.status, r.read().decode()
except Exception as e:
    try:
        out = e.read().decode()
    except Exception:
        out = ""
    code = getattr(e, "code", "?")
print(f"bad-token -> {code} {out[:80]}")
assert code == 403, "bad token must be 403"
code, _ = call("POST", "/api/session/close", {"session_id": sid})
print(f"close -> {code}")
assert code == 200, "session close must be 200"
print("SMOKE_OK")
PYEOF
echo SMOKE_DONE
