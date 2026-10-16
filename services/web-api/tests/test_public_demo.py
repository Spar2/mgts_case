import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from test_platform import pg, native_fixture
from workbench.api import create_app
from workbench.public_demo import load_package


def published():
    return Path(__file__).resolve().parents[1] / "public/demo-v1"


def test_public_isolated_from_archive_and_journal(pg, tmp_path, monkeypatch):
    monkeypatch.setenv("DTS_PUBLIC_PACKAGE", str(published()))
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    with TestClient(
        create_app(root / "records", queue, tmp_path / "runtime")
    ) as client:
        package = client.get("/api/public/v1/package").json()
        assert len(package["raw_c"]) == 51
        assert all(len(row) == 86 for row in package["raw_c"])
        assert package["x_m"][0] == 798 and package["x_m"][-1] == 883
        for path in (
            "/api/v1/records",
            "/api/v1/queue",
            "/api/v2/accounts",
            "/api/v2/jobs",
            "/api/v2/auth/sessions",
            "/api/v2/access-requests",
            "/api/v2/metrics",
        ):
            assert client.get(path).status_code == 401
        assert client.get("/api/public/v1/records/5").status_code == 401
        assert (
            client.post(
                "/api/public/v1/report", json={"answer": "present", "record_id": "1"}
            ).status_code
            == 422
        )
        report = client.post(
            "/api/public/v1/report", json={"notes": "<script>alert(1)</script>"}
        )
        assert report.status_code == 200 and "<script>" not in report.text
        assert "798.0 м" in report.text and "884.0 м" not in report.text
        assert (
            client.post(
                "/api/public/v1/access-requests",
                json={
                    "name": "Visitor",
                    "contact": "known-channel",
                    "reason": "Research",
                },
            ).status_code
            == 202
        )
        with pg.connect() as db:
            assert db.execute("SELECT count(*) FROM reviews").fetchone()[0] == 0
            assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
            assert db.execute("SELECT count(*) FROM access_requests").fetchone()[0] == 1
            assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 0


def test_package_integrity(monkeypatch, tmp_path):
    root = published()
    data = (root / "package.json").read_bytes()
    manifest = json.loads((root / "manifest.json").read_bytes())
    assert hashlib.sha256(data).hexdigest() == manifest["sha256"]
    (tmp_path / "package.json").write_bytes(data + b" ")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("DTS_PUBLIC_PACKAGE", str(tmp_path))
    import pytest

    with pytest.raises(ValueError, match="integrity"):
        load_package()


def test_published_values_match_real_source(monkeypatch, tmp_path):
    import os
    import pytest
    import numpy as np
    from workbench.store import Store

    if not os.environ.get("DTS_REAL_RECORDS"):
        pytest.skip("Requires original prepared real records")
    monkeypatch.setenv("DTS_PUBLIC_PACKAGE", str(published()))
    package = load_package()
    record = Store(os.environ["DTS_REAL_RECORDS"], tmp_path / "public-qa-cache").get(
        "5"
    )
    original = record.raw_c[1419:1470, 798:884]
    mask = record.invalid_reason[1419:1470, 798:884]
    assert package["invalid_reason"] == mask.tolist()
    assert package["x_m"] == record.x_m[798:884].tolist()
    assert package["time_s"] == record.elapsed_s[1419:1470].tolist()
    assert package["timestamps"] == [r["timestamp"] for r in record.profiles[1419:1470]]
    for row, native, reasons in zip(package["raw_c"], original, mask):
        for value, expected, reason in zip(row, native, reasons):
            assert value == float(expected) if reason == 0 else value is None
    assert np.isfinite(original[mask == 0]).all()
