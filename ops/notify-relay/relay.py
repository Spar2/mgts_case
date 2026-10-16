#!/usr/bin/env python3
"""notify-relay — stdlib-only HTTP relay: POST /relay -> Discord webhook.

Secrets come ONLY from env (host file /etc/notify-relay/relay.env, mode 0600).
Nothing secret is stored in the repo or in the image.

Required env:
  RELAY_TOKEN          Bearer token, expected as ``Authorization: Bearer <token>``.
  DISCORD_WEBHOOK_URL  Full ``https://discord.com/api/webhooks/...`` URL.

Optional env:
  RELAY_HOST       Bind address (default "0.0.0.0").
  RELAY_PORT       Bind port (default "8080").
  RELAY_MAX_BYTES  Max JSON body bytes (default "65536").
  RELAY_ALLOW_HTTP Set to "1" ONLY for local dry-runs against a stub webhook;
                   production requires https:// (default).

HTTP codes:
  GET  /healthz -> 200 {"ok": true}
  POST /relay   -> 401 (bad/missing bearer), 411 (no length), 413 (too large),
                   400 (bad JSON / not an object / empty),
                   500 (webhook not configured), 502 (webhook forward failed),
                   200 {"ok": true, "discord": <status>} on success.
  anything else -> 404; wrong method on a known path -> 405.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FORWARD_TIMEOUT = 10  # seconds, per spec
LOG = logging.getLogger("notify-relay")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _bearer_ok(header: str | None, expected: str) -> bool:
    """Constant-time bearer check (hmac.compare_digest, per spec)."""
    if not header or not expected:
        return False
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return False
    return hmac.compare_digest(token.strip(), expected)


def _webhook_ok(url: str) -> bool:
    """https required; http only with explicit RELAY_ALLOW_HTTP=1 (dry-run)."""
    if url.startswith("https://"):
        return True
    return url.startswith("http://") and _env("RELAY_ALLOW_HTTP") == "1"


def _forward_to_discord(webhook_url: str, payload: dict) -> int:
    """POST payload as JSON to the Discord webhook. Returns upstream status."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "notify-relay/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=FORWARD_TIMEOUT) as resp:
        return int(resp.status or 200)


class Handler(BaseHTTPRequestHandler):
    server_version = "notify-relay/1.0"

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        LOG.info("%s %s", self.address_string(), fmt % args)

    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: D102
        if self.path.split("?", 1)[0] == "/healthz":
            self._send_json(HTTPStatus.OK, {"ok": True})
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: D102
        if self.path.split("?", 1)[0] != "/relay":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        expected = _env("RELAY_TOKEN")
        if not _bearer_ok(self.headers.get("Authorization"), expected):
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send_json(HTTPStatus.LENGTH_REQUIRED, {"error": "length_required"})
            return
        try:
            length = int(raw_length)
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad_length"})
            return
        try:
            max_bytes = int(_env("RELAY_MAX_BYTES", "65536"))
        except ValueError:
            max_bytes = 65536
        if length < 0 or length > max_bytes:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "too_large"})
            return

        try:
            raw = self.rfile.read(length) if length else b""
        except (ConnectionError, ValueError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "read_failed"})
            return
        try:
            payload = json.loads(raw.decode("utf-8") or "null")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad_json"})
            return
        if not isinstance(payload, dict) or not payload:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "object_required"})
            return

        webhook_url = _env("DISCORD_WEBHOOK_URL")
        if not _webhook_ok(webhook_url):
            LOG.error("DISCORD_WEBHOOK_URL is not configured")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "webhook_misconfigured"})
            return

        try:
            upstream = _forward_to_discord(webhook_url, payload)
        except urllib.error.HTTPError as exc:
            LOG.warning("discord HTTP %s", exc.code)
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "webhook_failed", "discord": exc.code})
            return
        except Exception as exc:  # timeout / DNS / refused — never leak the URL
            LOG.warning("discord forward failed: %r", exc)
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "webhook_failed"})
            return

        if 200 <= upstream < 300:
            self._send_json(HTTPStatus.OK, {"ok": True, "discord": upstream})
        else:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "webhook_failed", "discord": upstream})

    # Known paths accept only their documented method.
    def do_PUT(self) -> None:  # noqa: D102
        self._send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method_not_allowed"})

    do_DELETE = do_PUT
    do_PATCH = do_PUT


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not _env("RELAY_TOKEN") or not _webhook_ok(_env("DISCORD_WEBHOOK_URL")):
        LOG.error("RELAY_TOKEN and DISCORD_WEBHOOK_URL (https://...) must be set in env")
        return 2
    host = _env("RELAY_HOST", "0.0.0.0")
    try:
        port = int(_env("RELAY_PORT", "8080"))
    except ValueError:
        LOG.error("RELAY_PORT must be an integer")
        return 2
    server = ThreadingHTTPServer((host, port), Handler)
    LOG.info("listening on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
