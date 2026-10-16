"""Direct links must open the SPA; arbitrary fallback paths remain unavailable."""

from fastapi.testclient import TestClient
from test_platform import login
from test_platform import pg as pg  # noqa: PLC0414 - pytest fixture
from test_workbench import fixture as native_fixture
from workbench.accounts import add_user
from workbench.api import create_app


def test_case_direct_links(pg, tmp_path, monkeypatch):
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("DTS case interface", encoding="utf-8")
    monkeypatch.setenv("DTS_WEB_DIST", str(dist))
    add_user(pg, "route-user", "QA-only-password-123456", "route-user", "viewer")
    with TestClient(create_app(root / "records", queue, root / "runtime")) as client:
        login(client, "route-user")
        for path in ("/cases", "/cases/" + "a" * 32):
            response = client.get(path)
            assert response.status_code == 200 and response.text == "DTS case interface"
        for path in ("/cases/not-an-id", "/api/unrecognized", "/private/file"):
            assert client.get(path).status_code == 404
