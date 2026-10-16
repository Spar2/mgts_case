"""Self-test miniresp против живого Redis. Запуск из compose-сети:
  docker run --rm --network haklolarp-stack_default \
    -v ./lib:/lib:ro -e REDIS_HOST=redis python:3.12-slim python3 /lib/test_miniresp.py
Использует отдельный ключ dts:selftest, историю режет MAXLEN ~ 10.
"""

import os
import sys

sys.path.insert(0, "/lib")
from miniresp import MiniResp, RedisError

HOST = os.environ.get("REDIS_HOST", "redis")
KEY = "dts:selftest"
GROUP = "selftest-group"


def check(name, cond):
    print(("ok" if cond else "FAIL") + " " + name)
    if not cond:
        raise SystemExit("selftest failed at " + name)


r = MiniResp(HOST, 6379, timeout=10.0)
check("ping", r.ping())

sid = r.xadd(KEY, {"payload": '{"a": 1}', "kind": "probe"}, maxlen_approx=10)
check("xadd-id", isinstance(sid, str) and "-" in sid)
check("xlen>=1", r.xlen(KEY) >= 1)

check("xgroup-create", r.xgroup_create(KEY, GROUP, start="0") == "OK")
check("xgroup-busy", r.xgroup_create(KEY, GROUP, start="0") == "BUSYGROUP")

got = r.xreadgroup(GROUP, "probe-consumer", KEY, count=10, block_ms=1000)
check("xreadgroup-nonempty", len(got) >= 1)
eid, fields = got[0]
check("fields", fields.get("kind") == "probe")
check("xack", r.xack(KEY, GROUP, eid) == 1)

total, _, _, _ = r.xpending_summary(KEY, GROUP)
check("xpending-zero", total == 0)

cursor, claimed, deleted = r.xautoclaim(KEY, GROUP, "probe-consumer", 60000)
check("xautoclaim-shape", cursor == "0-0" and claimed == [])

groups = r.xinfo_groups(KEY)
check("xinfo-groups", any(g.get("name") == GROUP for g in groups))

try:
    r.execute("NOSUCHCOMMAND")
    check("bad-command-raises", False)
except RedisError:
    check("bad-command-raises", True)

try:
    MiniResp("127.0.0.1", 6399, timeout=2.0).ping()
    check("conn-refused-raises", False)
except ConnectionError:
    check("conn-refused-raises", True)

r.close()
print("SELFTEST_OK")
