"""Минимальный RESP2-клиент для Redis 7: только команды стримов.

Без зависимостей (stdlib: socket). Покрывает ровно то, что нужно стенду:
PING, XADD, XGROUP CREATE, XREADGROUP, XACK, XPENDING, XAUTOCLAIM,
XLEN, XINFO GROUPS. Команды Redis 8 (MAXCOUNT/MAXSIZE) не используются.

Ошибки сервера -> RedisError. Обрыв сокета -> ConnectionError (переподключение
на вызывающей стороне: закрыть клиента, создать нового).
"""

import socket


class RedisError(Exception):
    pass


class MiniResp:
    def __init__(self, host="redis", port=6379, timeout=10.0):
        self._addr = (host, port)
        self._timeout = timeout
        self._sock = socket.create_connection(self._addr, timeout=timeout)
        self._buf = b""

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass

    # --- wire ---

    def _send(self, *args):
        parts = [f"*{len(args)}\r\n".encode()]
        for a in args:
            if isinstance(a, str):
                a = a.encode("utf-8")
            elif isinstance(a, int):
                a = str(a).encode()
            parts.append(f"${len(a)}\r\n".encode() + a + b"\r\n")
        try:
            self._sock.sendall(b"".join(parts))
        except OSError as e:
            raise ConnectionError(f"redis send failed: {e}") from e

    def _recv_line(self):
        while b"\r\n" not in self._buf:
            try:
                chunk = self._sock.recv(65536)
            except OSError as e:
                raise ConnectionError(f"redis recv failed: {e}") from e
            if not chunk:
                raise ConnectionError("redis closed connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\r\n", 1)
        return line

    def _recv_bytes(self, n):
        while len(self._buf) < n + 2:
            try:
                chunk = self._sock.recv(65536)
            except OSError as e:
                raise ConnectionError(f"redis recv failed: {e}") from e
            if not chunk:
                raise ConnectionError("redis closed connection")
            self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n + 2 :]
        return data

    def _parse(self):
        line = self._recv_line()
        if not line:
            raise RedisError("empty reply")
        kind, payload = line[:1], line[1:]
        if kind == b"+":
            return payload.decode("utf-8")
        if kind == b"-":
            raise RedisError(payload.decode("utf-8", "replace"))
        if kind == b":":
            return int(payload)
        if kind == b"$":
            n = int(payload)
            if n == -1:
                return None
            return self._recv_bytes(n).decode("utf-8")
        if kind == b"*":
            n = int(payload)
            if n == -1:
                return None
            return [self._parse() for _ in range(n)]
        raise RedisError(f"unknown reply type: {line!r}")

    def execute(self, *args):
        self._send(*args)
        return self._parse()

    # --- streams ---

    def ping(self):
        return self.execute("PING") == "PONG"

    def xadd(self, key, fields, maxlen_approx=None):
        cmd = ["XADD", key]
        if maxlen_approx is not None:
            cmd += ["MAXLEN", "~", maxlen_approx]
        cmd += ["*"]
        for k, v in fields.items():
            cmd += [k, v]
        return self.execute(*cmd)

    def xgroup_create(self, key, group, start="0", mkstream=True):
        cmd = ["XGROUP", "CREATE", key, group, start]
        if mkstream:
            cmd.append("MKSTREAM")
        try:
            return self.execute(*cmd)
        except RedisError as e:
            if "BUSYGROUP" in str(e):
                return "BUSYGROUP"
            raise

    def xreadgroup(self, group, consumer, key, count=50, block_ms=5000):
        rep = self.execute(
            "XREADGROUP",
            "GROUP",
            group,
            consumer,
            "COUNT",
            count,
            "BLOCK",
            block_ms,
            "STREAMS",
            key,
            ">",
        )
        if rep is None:
            return []
        out = []
        for _, entries in rep:
            for entry_id, flat in entries:
                fields = dict(zip(flat[::2], flat[1::2]))
                out.append((entry_id, fields))
        return out

    def xack(self, key, group, *ids):
        if not ids:
            return 0
        return self.execute("XACK", key, group, *ids)

    def xpending_summary(self, key, group):
        rep = self.execute("XPENDING", key, group)
        if rep is None or rep[0] == 0:
            return (0, None, None, [])
        total, first, last, consumers = rep
        return (total, first, last, [(c[0], c[1]) for c in consumers])

    def xautoclaim(self, key, group, consumer, min_idle_ms, start="0-0", count=10):
        rep = self.execute(
            "XAUTOCLAIM",
            key,
            group,
            consumer,
            min_idle_ms,
            start,
            "COUNT",
            count,
        )
        cursor, entries = rep[0], rep[1]
        deleted = rep[2] if len(rep) > 2 else []
        out = []
        for entry_id, flat in entries:
            out.append((entry_id, dict(zip(flat[::2], flat[1::2]))))
        return (cursor, out, deleted)

    def xlen(self, key):
        return self.execute("XLEN", key)

    def xinfo_groups(self, key):
        rep = self.execute("XINFO", "GROUPS", key)
        groups = []
        for flat in rep or []:
            groups.append(dict(zip(flat[::2], flat[1::2])))
        return groups
