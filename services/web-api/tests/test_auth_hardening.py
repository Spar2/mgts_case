"""Auth and durable ownership regressions against disposable PostgreSQL schemas."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient
from test_platform import login, native_fixture, pg  # noqa: F401
from workbench import accounts
from workbench.accounts import add_user
from workbench.api import create_app
from workbench.platform import principal

PASSWORD = "QA-only-password-123456"


@pytest.mark.parametrize("change", ["password", "disable"])
def test_login_cannot_issue_session_from_stale_credentials(pg, tmp_path, monkeypatch, change):  # noqa: F811
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    user = add_user(pg, "member", PASSWORD, "Member", "expert")
    verified, resume = Event(), Event()
    original = accounts.passwords

    class PausedVerification:
        def __getattr__(self, name):
            return getattr(original, name)

        def verify(self, hashed, password):
            original.verify(hashed, password)
            verified.set()
            assert resume.wait(10), "Concurrent security update did not finish"

    app = create_app(root / "records", queue, tmp_path / "runtime")
    with TestClient(app) as client:
        monkeypatch.setattr(accounts, "passwords", PausedVerification())
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(client.post, "/api/v2/auth/login", json={"username": "member", "password": PASSWORD})
            try:
                assert verified.wait(10)
                with pg.connect() as db:
                    if change == "password":
                        db.execute("UPDATE users SET password_hash=%s WHERE id=%s", (original.hash("replacement-password-123"), user["id"]))
                    else:
                        db.execute("UPDATE users SET enabled=false WHERE id=%s", (user["id"],))
                    db.execute("DELETE FROM sessions WHERE user_id=%s", (user["id"],))
            finally:
                resume.set()
            response = pending.result(timeout=10)
        assert response.status_code == 401
        assert "dts_session" not in response.cookies
        with pg.connect() as db:
            assert db.execute("SELECT count(*) FROM sessions WHERE user_id=%s", (user["id"],)).fetchone()[0] == 0


def test_document_collision_cannot_move_data_between_projects(pg):  # noqa: F811
    pg.put("qa-document", "shared-id", {"project_id": "other", "value": "preserve"})
    token = principal.set({"id": "technical-actor", "project_id": "local"})
    try:
        with pytest.raises(PermissionError):
            pg.put("qa-document", "shared-id", {"value": "replace"})
        with pytest.raises(PermissionError):
            pg.delete("qa-document", "shared-id")
        with pytest.raises(PermissionError):
            pg.get("qa-document", "shared-id")
        assert pg.list("qa-document") == []
        pg.put("qa-document", "own-id", {"value": "first"})
        pg.put("qa-document", "own-id", {"value": "second"})
        assert pg.get("qa-document", "own-id")["value"] == "second"
    finally:
        principal.reset(token)
    assert pg.get("qa-document", "shared-id")["value"] == "preserve"


def test_session_and_work_ownership_survive_application_restart(pg, tmp_path):  # noqa: F811
    root, queue, record = native_fixture.__wrapped__(tmp_path)
    user = add_user(pg, "owner", PASSWORD, "Owner", "analyst")
    runtime = tmp_path / "runtime"
    with TestClient(create_app(root / "records", queue, runtime)) as client:
        headers = login(client, "owner")
        cookie = client.cookies.get("dts_session")
        response = client.post("/api/v2/investigations", headers={**headers, "Idempotency-Key": "restart-proof"}, json={
            "title": "Durable work",
            "context": {
                "source_sha256": record.manifest["source_sha256"],
                "region": {"record_id": "5", "row_start": 0, "row_end": 3, "column_start": 0, "column_end": 3},
                "profile": 1, "column": 1,
            },
        })
        assert response.status_code == 201, response.text
        saved = response.json()
        assert saved["account_id"] == user["id"] and saved["project_id"] == "local"
        with pg.connect() as db:
            digest = db.execute("SELECT digest FROM sessions WHERE user_id=%s", (user["id"],)).fetchone()[0]
            assert digest == hashlib.sha256(cookie.encode()).hexdigest() and digest != cookie
            assert db.execute("SELECT payload->>'account_id' FROM documents WHERE kind='investigation' AND id=%s", (saved["id"],)).fetchone()[0] == user["id"]
    with TestClient(create_app(root / "records", queue, runtime)) as restarted:
        restarted.cookies.set("dts_session", cookie)
        assert restarted.get("/api/v2/auth/me").json()["id"] == user["id"]
        assert restarted.get("/api/v2/investigations/" + saved["id"]).json() == saved
        assert restarted.get("/api/v2/investigations/" + saved["id"] + "/history").json()["items"][0]["author_name"] == "Owner"


def test_auth_refusals_are_not_cached_and_cross_origin_login_is_rejected(pg, tmp_path):  # noqa: F811
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "member", PASSWORD, "Member", "expert")
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime")) as client:
        denied = client.get("/api/v1/records")
        assert denied.status_code == 401 and denied.headers["Cache-Control"] == "no-store"
        denied = client.post("/api/v2/auth/login", headers={"Origin": "https://untrusted.invalid"}, json={"username": "member", "password": PASSWORD})
        assert denied.status_code == 403 and denied.headers["Cache-Control"] == "no-store"
        with pg.connect() as db:
            assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0


def test_secure_cookie_and_absolute_expiry(pg, tmp_path, monkeypatch):  # noqa: F811
    import time

    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    user = add_user(pg, "member", PASSWORD, "Member", "expert")
    monkeypatch.setenv("DTS_ENV", "production")
    monkeypatch.setenv("DTS_SECURE_COOKIES", "true")
    monkeypatch.setenv("DTS_ALLOWED_HOSTS", "testserver")
    monkeypatch.setenv("DTS_ALLOWED_ORIGINS", "https://testserver")
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime"), base_url="https://testserver") as client:
        response = client.post("/api/v2/auth/login", json={"username": "member", "password": PASSWORD})
        assert response.status_code == 200
        cookie = response.headers["set-cookie"]
        assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=strict" in cookie
        assert "Max-Age=43200" in cookie and "Path=/" in cookie
        assert response.headers["Cache-Control"] == "no-store"
        assert client.get("/api/v2/auth/me").status_code == 200
        with pg.connect() as db:
            db.execute("UPDATE sessions SET expires=%s,last_seen=%s WHERE user_id=%s", (time.time() - 1, time.time(), user["id"]))
        assert client.get("/api/v2/auth/me").status_code == 401
        assert client.get("/api/v1/records").status_code == 401
