import pytest
from fastapi.testclient import TestClient
from test_platform import native_fixture, pg, login
from workbench.accounts import add_user
from workbench.api import create_app
from workbench.platform import configured


def test_production_requires_database(monkeypatch, tmp_path):
    monkeypatch.setenv("DTS_ENV", "production")
    monkeypatch.delenv("DTS_DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="requires PostgreSQL"):
        configured(tmp_path)


def test_database_outage_keeps_login_and_fails_all_data(monkeypatch, tmp_path):
    from test_public_demo import published
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    monkeypatch.setenv("DTS_ENV", "production")
    monkeypatch.setenv("DTS_SECURE_COOKIES", "true")
    monkeypatch.setenv("DTS_ALLOWED_HOSTS", "testserver")
    monkeypatch.setenv("DTS_ALLOWED_ORIGINS", "https://testserver")
    monkeypatch.setenv("DTS_DATABASE_URL", "postgresql://unused:unused@127.0.0.1:1/unavailable")
    monkeypatch.setenv("DTS_PUBLIC_PACKAGE", str(published()))
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("PUBLIC_LANDING_SHELL", encoding="utf-8")
    monkeypatch.setenv("DTS_WEB_DIST", str(dist))
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime"), base_url="https://testserver") as client:
        landing = client.get("/")
        assert landing.status_code == 200 and landing.text == "PUBLIC_LANDING_SHELL"
        assert landing.headers["Cache-Control"] == "no-store"
        assert client.get("/api/public/v1/package").status_code == 401
        assert client.post("/api/v2/auth/login", json={"username": "unknown", "password": "unavailable-password"}).status_code == 503
        client.cookies.set("dts_session", "opaque-unavailable-session")
        assert client.get("/api/v1/records").status_code == 503


def test_bad_csrf_does_not_extend_idle_session(pg, tmp_path):
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    user = add_user(pg, "user", "QA-only-password-123456", "User", "expert")
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime")) as client:
        login(client, "user")
        with pg.connect() as db:
            before = db.execute("SELECT last_seen FROM sessions WHERE user_id=%s", (user["id"],)).fetchone()[0]
        assert client.post("/api/v2/auth/activity", json={}).status_code == 403
        with pg.connect() as db:
            assert db.execute("SELECT last_seen FROM sessions WHERE user_id=%s", (user["id"],)).fetchone()[0] == before
