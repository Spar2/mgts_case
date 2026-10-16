import json

import pytest
from fastapi.testclient import TestClient
from test_platform import native_fixture, pg  # noqa: F401 (pg fixture used in tests below)
from test_public_demo import published
from workbench.api import create_app
from workbench.public_demo import load_package


def test_bq_unset_package_returns_503(tmp_path, monkeypatch):
    monkeypatch.delenv("DTS_PUBLIC_PACKAGE", raising=False)
    monkeypatch.delenv("DTS_DATABASE_URL", raising=False)
    monkeypatch.delenv("DTS_ENV", raising=False)
    assert load_package() is None
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime")) as client:
        package = client.get("/api/public/v1/package")
        assert package.status_code == 503
        assert "not configured" in package.json()["detail"]
        report = client.post("/api/public/v1/report", json={"notes": "hi"})
        assert report.status_code == 503


def test_bq_empty_package_returns_503(tmp_path, monkeypatch):
    monkeypatch.setenv("DTS_PUBLIC_PACKAGE", "")
    monkeypatch.delenv("DTS_DATABASE_URL", raising=False)
    monkeypatch.delenv("DTS_ENV", raising=False)
    assert load_package() is None
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime")) as client:
        package = client.get("/api/public/v1/package")
        assert package.status_code == 503
        assert "not configured" in package.json()["detail"]
        report = client.post("/api/public/v1/report", json={"notes": "hi"})
        assert report.status_code == 503


def test_bq_corrupt_package_fail_fast(tmp_path, monkeypatch):
    root = published()
    data = (root / "package.json").read_bytes()
    manifest = json.loads((root / "manifest.json").read_bytes())
    corrupted = tmp_path / "corrupt-pkg"
    corrupted.mkdir()
    (corrupted / "package.json").write_bytes(data + b" ")
    (corrupted / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("DTS_PUBLIC_PACKAGE", str(corrupted))
    with pytest.raises(ValueError, match="integrity"):
        load_package()
    native_root, queue, _ = native_fixture.__wrapped__(tmp_path)
    app = create_app(native_root / "records", queue, tmp_path / "runtime")
    with pytest.raises(ValueError, match="integrity"):
        with TestClient(app):
            pass


def test_bq_anonymous_with_platform_private_401(pg, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("DTS_PUBLIC_PACKAGE", str(published()))
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime")) as client:
        private = client.get("/api/v1/records")
        assert private.status_code == 401
        assert "Sign in first" in private.json()["detail"]
        assert client.get("/api/public/v1/package").status_code == 200


def test_bq_anonymous_without_platform_public_200_private_503(tmp_path, monkeypatch):
    monkeypatch.delenv("DTS_DATABASE_URL", raising=False)
    monkeypatch.delenv("DTS_ENV", raising=False)
    monkeypatch.setenv("DTS_PUBLIC_PACKAGE", str(published()))
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime")) as client:
        assert client.get("/api/public/v1/package").status_code == 200
        private = client.get("/api/v1/records")
        assert private.status_code == 503
        assert "Account storage" in private.json()["detail"]


def test_bq_report_does_not_touch_journal_or_reviews(pg, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("DTS_PUBLIC_PACKAGE", str(published()))
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    with TestClient(create_app(root / "records", queue, tmp_path / "runtime")) as client:
        report = client.post(
            "/api/public/v1/report", json={"notes": "<script>alert(1)</script>"}
        )
        assert report.status_code == 200
        assert "<script>" not in report.text
        assert client.app.state.journal.annotations() == {}
        with pg.connect() as db:
            assert db.execute("SELECT count(*) FROM reviews").fetchone()[0] == 0
            assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
            assert db.execute("SELECT count(*) FROM access_requests").fetchone()[0] == 0
            assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
