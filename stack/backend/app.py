"""Backend-заглушка LAN-стенда: HTTP API по VIEWER_CONTRACT.md (sol-ветка).

Реализует ровно контракт, без ML: валидация входа строгая (DATA_CONTRACT v1),
принятые профили публикует в Redis Stream (dts:profiles), события всегда пустые.
В ClickHouse пишет отдельный ingester. Настоящий StreamingAnalyzer подключат
ml/web позже за тем же API.

Переменные окружения:
  PORT (default 8000), REGISTRY_DIR (default /registry),
  REDIS_HOST (default redis), REDIS_PORT (default 6379),
  STREAM_PROFILES (default dts:profiles), STREAM_MAXLEN (default 20000),
  STREAM_NO_TRIM=1 + STREAM_BACKLOG_MAX (default 1000): без MAXLEN-трима,
  переполнение — 503 backpressure (иначе трим ~MAXLEN),
  MAX_SESSIONS=16 (запас под сессии убитых стримеров),
  SESSION_TTL_S=300 (протухшие чистятся при создании новых).
  GET /readyz — проба Redis (PING + XLEN dts:profiles), вне CONTRACTS v1.
"""

import hmac
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/opt/minilib")
from chutil import epoch_to_ch, sanitize_temps
from miniresp import MiniResp, RedisError

PORT = int(os.environ.get("PORT", "8000"))
REGISTRY_DIR = os.environ.get("REGISTRY_DIR", "/registry")
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
STREAM_PROFILES = os.environ.get("STREAM_PROFILES", "dts:profiles")
STREAM_MAXLEN = int(os.environ.get("STREAM_MAXLEN", "20000"))
MAX_BODY = 2 * 1024 * 1024
MAX_SESSIONS = 16
SESSION_TTL_S = 300
SEGMENT_GAP_S = 300.0
RECORDS = ["1", "2", "3", "4", "5", "6"]

MUTATION_TOKEN = uuid.uuid4().hex


def load_model():
    path = os.path.join(REGISTRY_DIR, "manifest.json")
    with open(path, "r", encoding="utf-8-sig") as f:
        manifest = json.load(f)
    return {
        "name": manifest.get("name", "seed-stub"),
        "model_key": manifest.get("model_key", "unregistered"),
        "manifest": manifest,
    }


MODEL = load_model()
SESSIONS = {}  # session_id -> dict

_REDIS_LOCK = threading.Lock()
_REDIS_CONN = None


def stream_publish(row):
    """Публикация канонического профиля в Redis Stream. Возвращает stream id.

    Соединение переиспользуется между запросами (один TCP на процесс, под
    замком — сокет не потокобезопасен). При обрыве — один реконнект;
    серверные ошибки (напр. OOM при noeviction) уходят наружу сразу.
    """
    global _REDIS_CONN
    payload = json.dumps(row)
    fields = {"kind": "profile", "payload": payload}
    try:
        with _REDIS_LOCK:
            if _REDIS_CONN is None:
                _REDIS_CONN = MiniResp(REDIS_HOST, REDIS_PORT, timeout=10.0)
            try:
                if os.environ.get("STREAM_NO_TRIM")=="1" and _REDIS_CONN.xlen(STREAM_PROFILES)>=int(os.environ.get("STREAM_BACKLOG_MAX","1000")):
                    raise RedisError("Backlog limit reached; backpressure")
                return _REDIS_CONN.xadd(
                    STREAM_PROFILES, fields, maxlen_approx=None if os.environ.get("STREAM_NO_TRIM")=="1" else STREAM_MAXLEN
                )
            except (ConnectionError, OSError):
                try:
                    _REDIS_CONN.close()
                except OSError:
                    pass
                _REDIS_CONN = MiniResp(REDIS_HOST, REDIS_PORT, timeout=10.0)
                if os.environ.get("STREAM_NO_TRIM")=="1" and _REDIS_CONN.xlen(STREAM_PROFILES)>=int(os.environ.get("STREAM_BACKLOG_MAX","1000")):
                    raise RedisError("Backlog limit reached; backpressure")
                return _REDIS_CONN.xadd(
                    STREAM_PROFILES, fields, maxlen_approx=None if os.environ.get("STREAM_NO_TRIM")=="1" else STREAM_MAXLEN
                )
    except (RedisError, ConnectionError, OSError) as e:
        raise RedisError(f"stream publish failed: {e}") from e


def parse_ts(raw):
    """ISO8601 -> (epoch_float, zoned_bool). Строгий, без угадывания зон.
    Паритет с sol dts/data.py: суффикс +HH нормализуется в +HH:00,
    исходная строка хранится отдельно без изменений."""
    if not isinstance(raw, str) or not raw:
        raise ValueError("timestamp must be a non-empty string")
    try:
        dt = datetime.fromisoformat(re.sub(r"([+-]\d{2})$", r"\1:00", raw))
    except ValueError:
        raise ValueError(f"timestamp is not ISO8601: {raw!r}")
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC).timestamp(), False
    return dt.timestamp(), True


def validate_profile(p):
    """Возвращает нормализованный dict или raise ValueError. Состояние не меняет."""
    if not isinstance(p, dict):
        # TRY004: все ошибки входа маппятся в HTTP 400 через ValueError.
        raise ValueError("profile must be an object")  # noqa: TRY004
    for k in ("dts_id", "channel_id", "timestamp", "points"):
        if k not in p:
            raise ValueError(f"missing required key: {k}")
    if not isinstance(p["dts_id"], str) or not p["dts_id"]:
        raise ValueError("dts_id must be a non-empty string")
    if not isinstance(p["channel_id"], str) or not p["channel_id"]:
        raise ValueError("channel_id must be a non-empty string")
    pts = p["points"]
    if not isinstance(pts, list) or not pts:
        raise ValueError("points must be a non-empty list")
    xs, temps, reasons = [], [], []
    prev_x = None
    for i, pair in enumerate(pts):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"points[{i}] must be [x_m, temp]")
        x, t = pair
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(f"points[{i}][0] must be numeric")  # noqa: TRY004
        if not math.isfinite(x) or x < 0:
            raise ValueError(f"points[{i}][0] must be finite and >= 0")
        if prev_x is not None and not x > prev_x:
            raise ValueError(f"x_m must strictly increase at index {i}")
        prev_x = float(x)
        xs.append(float(x))
        if t is None:
            temps.append(float("nan"))
            reasons.append(2)
        elif isinstance(t, bool) or not isinstance(t, (int, float)):
            raise ValueError(f"points[{i}][1] must be numeric or null")
        elif t == -300:
            temps.append(-300.0)
            reasons.append(1)
        elif not math.isfinite(t):
            temps.append(float("nan"))
            reasons.append(2)
        else:
            temps.append(float(t))
            reasons.append(0)
    mask = p.get("valid_mask")
    if mask is not None:
        if not isinstance(mask, list) or len(mask) != len(pts):
            raise ValueError("valid_mask length must match points")
        for i, m in enumerate(mask):
            if m not in (0, 1):
                raise ValueError("valid_mask values must be 0/1")
            if m == 0 and reasons[i] == 0:
                reasons[i] = 8
    epoch, zoned = parse_ts(p["timestamp"])
    return {
        "dts_id": p["dts_id"],
        "channel_id": p["channel_id"],
        "ts_raw": p["timestamp"],
        "epoch": epoch,
        "zoned": zoned,
        "x_m": xs,
        "temp_c": temps,
        "invalid_reason": reasons,
    }


def sweep_sessions():
    now = time.time()
    for sid in [s for s, v in SESSIONS.items() if now - v["last"] > SESSION_TTL_S]:
        del SESSIONS[sid]


class Handler(BaseHTTPRequestHandler):
    server_version = "haklo-stub/1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        ctype = self.headers.get("Content-Type", "")
        if ctype.split(";")[0].strip() != "application/json":
            raise ValueError("Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("bad Content-Length")
        if length < 1 or length > MAX_BODY:
            raise ValueError(f"body length 1..{MAX_BODY} required")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"malformed JSON: {e}")

    def _check_token(self):
        presented = self.headers.get("X-DTS-Token") or ""
        if not hmac.compare_digest(presented, MUTATION_TOKEN):
            self._json(403, {"error": "bad or missing X-DTS-Token"})
            return False
        return True

    def do_GET(self):
        if self.path == '/readyz':
            try:
                connection = MiniResp(REDIS_HOST, REDIS_PORT, timeout=2.0)
                try:
                    connection.execute('PING')
                    backlog = connection.xlen(STREAM_PROFILES)
                finally:
                    connection.close()
                return self._json(200, {'redis': True, 'backlog_profiles': backlog})
            except (RedisError, ConnectionError, OSError):
                return self._json(503, {'redis': False})
        if self.path == "/healthz":
            return self._json(200, {"ok": True})
        if self.path == "/api/manifest":
            return self._json(
                200,
                {
                    "records": RECORDS,
                    "model": {"name": MODEL["name"], "model_key": MODEL["model_key"]},
                    "mutation_token": MUTATION_TOKEN,
                    "fields": ["native", "clean", "baseline_adaptive", "residual"],
                },
            )
        if self.path.startswith("/api/snapshot"):
            return self._json(
                200,
                {
                    "events": [],
                    "closed_events": [],
                    "note": "stub: no analyzer attached yet",
                },
            )
        self._json(404, {"error": "unknown route"})

    def do_POST(self):
        try:
            body = self._read_json()
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        if not isinstance(body, dict):
            return self._json(400, {"error": "body must be a JSON object"})
        if self.path == "/api/session":
            if not self._check_token():
                return
            sweep_sessions()
            mode = body.get("mode", "live")
            if mode not in ("live", "measurement_replay", "synthetic_replay"):
                return self._json(400, {"error": "bad mode"})
            if len(SESSIONS) >= MAX_SESSIONS:
                return self._json(409, {"error": "too many sessions"})
            sid = uuid.uuid4().hex
            SESSIONS[sid] = {
                "mode": mode,
                "last": time.time(),
                "profile_index": 0,
                "_lock": threading.Lock(),
                "segment_id": 0,
                "prev_epoch": None,
                "zoned": None,
                "schema_version": body.get("schema_version",1),
                "replay_id": body.get("replay_id",sid),
                "source_sha256": body.get("source_sha256", ""),
            }
            return self._json(
                200, {"session_id": sid, "model_key": MODEL["model_key"], "mode": mode}
            )
        if self.path == "/api/session/close":
            if not self._check_token():
                return
            SESSIONS.pop(body.get("session_id", ""), None)
            return self._json(200, {"closed": True})
        if self.path == "/api/infer":
            if not self._check_token():
                return
            sid = body.get("session_id", "")
            sess = SESSIONS.get(sid)
            if sess is None:
                return self._json(404, {"error": "unknown session"})
            with sess["_lock"]:
                try:
                    prof = validate_profile(body.get("profile"))
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
                if sess["zoned"] is None:
                    pass
                elif sess["zoned"] != prof["zoned"]:
                    return self._json(
                        400, {"error": "zoned/naive time mixing in one session"}
                    )
                if sess["prev_epoch"] is not None:
                    if not prof["epoch"] > sess["prev_epoch"]:
                        return self._json(400, {"error": "time must strictly increase"})
                idx = sess["profile_index"]
                segment=sess["segment_id"]
                if sess["prev_epoch"] is not None and prof["epoch"]-sess["prev_epoch"]>SEGMENT_GAP_S: segment+=1
                row={
                    "dts_id":prof["dts_id"],"channel_id":prof["channel_id"],"ts_raw":prof["ts_raw"],
                    "ts":epoch_to_ch(prof["epoch"]),"row_idx":idx,"segment_id":segment,"x_m":prof["x_m"],
                    "temp_c":sanitize_temps(prof["temp_c"]),"invalid_reason":prof["invalid_reason"]}
                if sess["schema_version"]==2:
                    if not re.fullmatch(r"[0-9a-f]{32}",sess["replay_id"]) or not re.fullmatch(r"[0-9a-f]{64}",sess["source_sha256"]): return self._json(400,{"error":"Invalid source provenance"})
                    source_row=body.get("source_row")
                    source_segment=body.get("source_segment")
                    if isinstance(source_segment,bool) or not isinstance(source_segment,int) or source_segment<0: return self._json(400,{"error":"Invalid native segment"})
                    native=body.get("native_invalid_reason")
                    elapsed=body.get("elapsed_s")
                    if isinstance(source_row,bool) or not isinstance(source_row,int) or source_row<0 or not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or not isinstance(native,list) or len(native)!=len(prof["x_m"]) or any(isinstance(v,bool) or v not in (0,1,2,4,8) for v in native): return self._json(400,{"error":"Invalid native provenance"})
                    if any(flag==0 and reason!=0 for flag,reason in zip(native,prof["invalid_reason"])): return self._json(400,{"error":"Invalid native valid mask"})
                    row.update(segment_id=source_segment,replay_id=sess["replay_id"],source_sha256=sess["source_sha256"],row_idx=source_row,elapsed_s=elapsed,time_kind="zoned" if prof["zoned"] else "local_unzoned",ts=epoch_to_ch(prof["epoch"]) if prof["zoned"] else None,invalid_reason=native)
                    row["payload_sha256"]=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
                try:
                    stream_id = stream_publish(row)
                except RedisError as e:
                    return self._json(503, {"error": str(e)})
                sess.update(prev_epoch=prof["epoch"],last=time.time(),profile_index=idx+1,segment_id=segment,zoned=prof["zoned"])
                out = {
                    "events": [],
                    "closed_events": [],
                    "model_key": MODEL["model_key"],
                    "mode": sess["mode"],
                    "segment_id": sess["segment_id"],
                    "profile_index": idx,
                    "registered_sensor": False,
                    "queued": True,
                    "stream_id": stream_id,
                }
                return self._json(200, out)
        self._json(404, {"error": "unknown route"})


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"stub backend on {PORT}, model={MODEL['model_key']}", flush=True)
    srv.serve_forever()
