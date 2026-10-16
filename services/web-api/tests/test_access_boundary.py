"""Exercise the complete anonymous route boundary and revoked stream access."""

import asyncio
import re
import time

import pytest
from fastapi.testclient import TestClient
from psycopg import OperationalError
from starlette.requests import Request
from test_platform import login, native_fixture, pg  # noqa: F401
from workbench import accounts
from workbench.accounts import add_user
from workbench.api import create_app
from workbench.platform import principal

PASSWORD = "QA-only-password-123456"


def test_every_registered_private_api_denies_anonymous(pg, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("DTS_TEST_ALLOW_ANONYMOUS", "false")
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    app = create_app(root / "records", queue, tmp_path / "runtime")
    exceptions = {
        "/api/v1/health",
        "/api/v2/health/live",
        "/api/v2/auth/login",
        "/api/v2/auth/activate",
        "/api/v2/auth/recover",
    }
    checked = 0
    with TestClient(app) as client:
        for route in app.routes:
            if not route.path.startswith("/api/") or route.path in exceptions:
                continue
            path = re.sub(r"\{[^}]+\}", "qa-denied", route.path)
            for method in route.methods:
                response = client.request(method, path, follow_redirects=False)
                assert response.status_code == 401, (method, path, response.status_code)
                assert response.headers["Cache-Control"] == "no-store"
                checked += 1
        assert checked >= 80
        assert (
            client.get(
                "/api/v2/auth/guard", headers={"X-Original-URI": "//attacker.invalid"}
            ).headers["X-Login-Redirect"]
            == "/login?next=%2Fworks"
        )


@pytest.mark.parametrize("supplied", [b"\xff", b"incorrect-token"])
def test_invalid_csrf_returns_denial_not_server_error(pg, tmp_path, supplied):  # noqa: F811
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "member", PASSWORD, "Member", "analyst")
    with TestClient(
        create_app(root / "records", queue, tmp_path / "runtime")
    ) as client:
        login(client, "member")
        response = client.post(
            "/api/v1/bookmarks", headers=[(b"x-workbench-token", supplied)], json={}
        )
        assert response.status_code == 403
        assert client.get("/api/v1/bookmarks").json() == []


@pytest.mark.parametrize(
    "host", ["attacker.invalid", "[invalid", "testserver:invalid", "evil@testserver"]
)
def test_invalid_host_is_rejected_before_database(pg, tmp_path, monkeypatch, host):  # noqa: F811
    from workbench import api

    root, queue, _ = native_fixture.__wrapped__(tmp_path)

    def forbidden(*_):
        raise AssertionError("Invalid host reached session storage")

    monkeypatch.setattr(api, "authenticate", forbidden)
    with TestClient(
        create_app(root / "records", queue, tmp_path / "runtime")
    ) as client:
        client.cookies.set("dts_session", "opaque-test-cookie")
        assert client.get("/api/v1/records", headers={"Host": host}).status_code == 403


def test_cross_site_auth_and_unsafe_asset_types_are_blocked(pg, tmp_path, monkeypatch):  # noqa: F811
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("QA shell")
    for name in ("app.js", "private.html", "app.js.map", "config.json"):
        (assets / name).write_text("QA fixture")
    monkeypatch.setenv("DTS_WEB_DIST", str(dist))
    with TestClient(
        create_app(root / "records", queue, tmp_path / "runtime")
    ) as client:
        assert client.get("/assets/app.js").status_code == 200
        for path in (
            "/assets/private.html",
            "/assets/app.js.map",
            "/assets/config.json",
            "/assets/.hidden.js",
        ):
            assert client.get(path).status_code == 404
        for path in (
            "/api/v2/auth/login",
            "/api/v2/auth/activate",
            "/api/v2/auth/recover",
        ):
            assert (
                client.post(
                    path, headers={"Sec-Fetch-Site": "cross-site"}, json={}
                ).status_code
                == 403
            )
            assert (
                client.post(
                    path, headers={"Content-Type": "text/plain"}, content="{}"
                ).status_code
                == 415
            )
        page = client.get("/login")
        assert page.headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in page.headers["Content-Security-Policy"]


def test_relogin_revokes_previous_browser_session_and_replay_is_owned(pg, tmp_path):  # noqa: F811
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    for name in ("alice", "bob"):
        add_user(pg, name, PASSWORD, name, "analyst")
    with TestClient(
        create_app(root / "records", queue, tmp_path / "runtime")
    ) as client:
        headers = login(client, "alice")
        previous = client.cookies.get("dts_session")
        key = client.post(
            "/api/v1/replays", json={"record_id": "5"}, headers=headers
        ).json()["id"]
        headers = login(client, "bob")
        for path in (f"/api/v1/replays/{key}", f"/api/v1/replays/{key}/events"):
            assert client.get(path).status_code == 403
        assert (
            client.post(
                f"/api/v1/replays/{key}/commands",
                json={"action": "play"},
                headers=headers,
            ).status_code
            == 403
        )
        assert (
            client.delete(f"/api/v1/replays/{key}", headers=headers).status_code == 403
        )
        current = client.cookies.get("dts_session")
        client.cookies.clear()
        client.cookies.set("dts_session", previous)
        assert client.get("/api/v2/auth/me").status_code == 401
        client.cookies.clear()
        client.cookies.set("dts_session", current)
        assert client.get("/api/v2/auth/me").json()["username"] == "bob"


@pytest.mark.parametrize(
    "path", ["/api/v1/replays/{key}/events", "/api/v2/stream/{key}/events"]
)
@pytest.mark.parametrize("revocation", ["logout", "idle", "disable", "storage-outage"])
def test_open_stream_stops_after_access_loss(
    pg, tmp_path, monkeypatch, path, revocation  # noqa: F811
):
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    user = add_user(pg, "stream-member", PASSWORD, "Stream member", "analyst")
    app = create_app(root / "records", queue, tmp_path / "runtime")
    with TestClient(app) as client:
        headers = login(client, "stream-member")
        expected = client.get("/api/v2/auth/me").json()
        if path.startswith("/api/v1"):
            key = client.post(
                "/api/v1/replays", json={"record_id": "5"}, headers=headers
            ).json()["id"]
        else:
            key = "qa-stream"
            pg.put(
                "stream",
                key,
                {
                    "id": key,
                    "project_id": "local",
                    "status": "running",
                    "snapshot_version": 1,
                },
            )
        route = next(r for r in app.routes if r.path == path)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": path.replace("{key}", key),
                "query_string": b"",
                "scheme": "http",
                "server": ("testserver", 80),
                "headers": [
                    (
                        b"cookie",
                        ("dts_session=" + client.cookies.get("dts_session")).encode(),
                    )
                ],
            },
            receive=receive,
        )
        token = principal.set(expected)

        async def exercise():
            response = await route.endpoint(key, request)
            first = await anext(response.body_iterator)
            assert "event:" in first
            if revocation == "storage-outage":

                def unavailable(*_):
                    raise OperationalError("QA storage unavailable")

                monkeypatch.setattr(accounts, "authenticate", unavailable)
            else:
                with pg.connect() as db:
                    if revocation == "logout":
                        db.execute(
                            "DELETE FROM sessions WHERE user_id=%s", (user["id"],)
                        )
                    elif revocation == "idle":
                        db.execute(
                            "UPDATE sessions SET last_seen=%s WHERE user_id=%s",
                            (time.time() - 1801, user["id"]),
                        )
                    else:
                        db.execute(
                            "UPDATE users SET enabled=false WHERE id=%s", (user["id"],)
                        )
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(anext(response.body_iterator), timeout=3)

        try:
            asyncio.run(exercise())
        finally:
            principal.reset(token)
