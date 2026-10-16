"""Closed deployment regression tests using real isolated PostgreSQL."""
import pytest
from fastapi.testclient import TestClient
from test_platform import (
    login,
    native_fixture,
    pg,  # noqa: F401 - pytest fixture registration
)
from workbench.accounts import add_user
from workbench.api import create_app
from workbench.platform import configured


@pytest.mark.parametrize("environment", [None, "development", "production", "test"])
def test_no_database_never_silently_opens_product(monkeypatch, tmp_path, environment):
    monkeypatch.delenv("DTS_DATABASE_URL", raising=False)
    monkeypatch.setenv("DTS_TEST_ALLOW_ANONYMOUS", "false")
    if environment is None:
        monkeypatch.delenv("DTS_ENV", raising=False)
    else:
        monkeypatch.setenv("DTS_ENV", environment)
    with pytest.raises(ValueError, match="requires PostgreSQL"):
        configured(tmp_path)


def test_production_rejects_test_bypass(monkeypatch, tmp_path):
    monkeypatch.setenv("DTS_ENV", "production")
    monkeypatch.setenv("DTS_TEST_ALLOW_ANONYMOUS", "true")
    monkeypatch.delenv("DTS_DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="requires PostgreSQL"):
        configured(tmp_path)


def test_account_actions_never_use_anonymous_fixture_identity(tmp_path, monkeypatch):
    """Even the explicit legacy QA bypass cannot impersonate an account."""
    monkeypatch.delenv("DTS_DATABASE_URL", raising=False)
    monkeypatch.delenv("DTS_PUBLIC_PACKAGE", raising=False)
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime")) as client:
        headers = {"X-Workbench-Token": client.get("/api/v1/session").json()["mutation_token"]}
        assert client.get("/api/v2/auth/sessions").status_code == 401
        assert client.post("/api/v2/auth/sessions/others/revoke", json={}, headers=headers).status_code == 401
        assert client.post("/api/v2/auth/password", headers=headers, json={
            "current_password": "QA-only-password-123456", "password": "replacement-password-123456",
        }).status_code == 401


def test_anonymous_pages_apis_and_downloads_are_closed(pg, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("DTS_TEST_ALLOW_ANONYMOUS", "false")
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("LOGIN_SHELL", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("static", encoding="utf-8")
    monkeypatch.setenv("DTS_WEB_DIST", str(dist))
    add_user(pg, "member", "QA-only-password-123456", "Member", "analyst")
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime")) as client:
        for path in ["/try", "/request-access", "/research", "/works", "/monitoring", "/laboratory", "/cases", "/demo", "/account"]:
            response = client.get(path, follow_redirects=False)
            assert response.status_code == 303, (path, response.status_code)
            assert response.headers["location"].startswith("/login?next=")
        for path in ["/api/v1/records", "/api/v1/session", "/api/v1/queue", "/api/v2/investigations", "/api/v2/result-runs", "/api/v2/accounts", "/api/v2/jobs", "/api/v2/cases/missing/bundle", "/api/public/v1/package", "/openapi.json"]:
            response = client.get(path, follow_redirects=False)
            assert response.status_code in (401, 303), (path, response.status_code)
            assert "LOGIN_SHELL" not in response.text
        for path in ["/api/public/v1/report", "/api/public/v1/access-requests"]:
            assert client.post(path, json={}).status_code == 401
        health = client.get("/api/v1/health").json()
        assert health["auth_enabled"] and health["access_mode"] == "closed-team"
        assert "records" not in health
        for path in ["/", "/?next=%2Fresearch", "/login", "/activate", "/recover", "/assets/app.js"]:
            assert client.get(path).status_code == 200
        response = client.get("/api/v2/auth/guard", headers={"X-Original-URI": "/research?record=5&row=2"})
        assert response.status_code == 401
        assert response.headers["X-Login-Redirect"] == "/login?next=%2Fresearch%3Frecord%3D5%26row%3D2"
        headers = login(client, "member")
        assert client.get("/api/v2/auth/guard").status_code == 204
        assert client.get("/api/v1/records").status_code == 200
        assert client.get("/works", follow_redirects=False).status_code == 200
        assert client.post("/api/v2/auth/logout", json={}, headers=headers).status_code == 200
        assert client.get("/api/v1/records").status_code == 401


def test_viewer_cannot_create_persistent_passport(pg, tmp_path):  # noqa: F811
    root, queue, record = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "viewer", "QA-only-password-123456", "Viewer", "viewer")
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime")) as client:
        headers = login(client, "viewer")
        response = client.post("/api/v2/passports", headers=headers, json={
            "region": {"record_id": "5", "row_start": 0, "row_end": 3, "column_start": 0, "column_end": 3},
            "profile": 0, "source_sha256": record.manifest["source_sha256"], "method": "identity", "title": "QA", "scale": [0, 30],
        })
        assert response.status_code == 403, response.text
        with pg.connect() as db:
            assert db.execute("SELECT count(*) FROM documents WHERE kind='passport'").fetchone()[0] == 0


def test_only_named_published_landing_files_are_anonymous(pg, tmp_path, monkeypatch):
    from workbench.access_policy import LANDING_ASSETS

    monkeypatch.setenv("DTS_TEST_ALLOW_ANONYMOUS", "false")
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    dist = tmp_path / "dist"
    (dist / "landing").mkdir(parents=True)
    (dist / "index.html").write_text("LANDING", encoding="utf-8")
    for path in LANDING_ASSETS:
        (dist / path.lstrip("/")).write_bytes(b"published")
    (dist / "landing/private.json").write_bytes(b"must stay private")
    monkeypatch.setenv("DTS_WEB_DIST", str(dist))
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime")) as client:
        for path in LANDING_ASSETS:
            response = client.get(path)
            assert response.status_code == 200 and response.content == b"published"
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert client.post(path, json={}).status_code == 403
        for path in ["/landing/private.json", "/landing/visual-data.json.bak", "/landing/.env"]:
            response = client.get(path, follow_redirects=False)
            assert response.status_code == 303 and "must stay private" not in response.text
