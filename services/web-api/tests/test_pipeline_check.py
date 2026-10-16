"""Compile-only pipeline syntax check. Sources are never executed or returned."""

import pytest
from fastapi.testclient import TestClient
from test_platform import login
from test_platform import pg as platform_pg
from test_workbench import fixture as native_fixture
from workbench.accounts import add_user
from workbench.api import create_app
from workbench.pipeline_check import STAGE_ORDER

pg = platform_pg

CLEAN_MARKER = "CLEAN_MARKER_9f3c"
ANOMALY_MARKER = "ANOMALY_MARKER_9f3c"
SOURCE_MARKER = "SOURCE_MARKER_9f3c"


@pytest.fixture
def app(pg, tmp_path):
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "check-viewer", "QA-only-password-123456", "Check viewer", "viewer")
    return create_app(root / "records", queue, root / "runtime")


def _stages(sources):
    return [{"id": stage_id, "source": sources[stage_id]} for stage_id in STAGE_ORDER]


def test_valid_five_stages(app):
    with TestClient(app) as client:
        headers = login(client, "check-viewer")
        response = client.post(
            "/api/v2/pipeline-check",
            headers=headers,
            json={"stages": _stages({stage_id: "value = 1\n" for stage_id in STAGE_ORDER})},
        )
        assert response.status_code == 200, response.text
        assert response.json() == {"ok": True}
        assert SOURCE_MARKER not in response.text


def test_clean_syntax_error_precedes_broken_anomaly(app):
    clean = "def clean():\nreturn '" + CLEAN_MARKER + "'\n"
    anomaly = ANOMALY_MARKER + " = (\n"
    try:
        compile(clean, "<pipeline>", "exec")
    except SyntaxError as exc:
        expected_line = 1 if exc.lineno is None else exc.lineno
        expected_message = exc.msg
    else:
        raise AssertionError("clean fixture must be a syntax error")
    with TestClient(app) as client:
        headers = login(client, "check-viewer")
        response = client.post(
            "/api/v2/pipeline-check",
            headers=headers,
            json={
                "stages": [
                    {"id": "anomaly", "source": anomaly},
                    {"id": "vision", "source": "value = 1\n"},
                    {"id": "clean", "source": clean},
                    {"id": "classify", "source": "value = 2\n"},
                    {"id": "preprocess", "source": "value = 3\n"},
                ]
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ok"] is False
        assert body["stage_id"] == "clean"
        assert body["line"] == expected_line
        assert isinstance(body["line"], int)
        assert body["message"] == expected_message
        assert set(body) == {"ok", "stage_id", "line", "message"}
        assert ANOMALY_MARKER not in response.text
        assert CLEAN_MARKER not in response.text
        assert anomaly not in response.text
        assert clean not in response.text


def test_empty_source_skipped(app, monkeypatch):
    seen = []
    real_compile = compile

    def spy(source, filename, mode):
        seen.append((source, filename, mode))
        return real_compile(source, filename, mode)

    monkeypatch.setattr("workbench.pipeline_check.compile", spy, raising=False)
    anomaly = "anomaly_value = 1\n"
    vision = "vision_value = 2\n"
    with TestClient(app) as client:
        headers = login(client, "check-viewer")
        response = client.post(
            "/api/v2/pipeline-check",
            headers=headers,
            json={
                "stages": _stages(
                    {
                        "preprocess": " \n\t",
                        "clean": "",
                        "anomaly": anomaly,
                        "classify": "\r\n",
                        "vision": vision,
                    }
                )
            },
        )
        assert response.status_code == 200, response.text
        assert response.json() == {"ok": True}
        assert seen == [(anomaly, "<pipeline>", "exec"), (vision, "<pipeline>", "exec")]
        assert anomaly not in response.text
        assert vision not in response.text


def test_missing_session_is_401(app):
    with TestClient(app) as client:
        response = client.post(
            "/api/v2/pipeline-check",
            json={"stages": _stages({stage_id: "value = 1\n" for stage_id in STAGE_ORDER})},
        )
        assert response.status_code == 401
        assert SOURCE_MARKER not in response.text


def test_unknown_stage_id_is_422(app):
    stages = _stages({stage_id: SOURCE_MARKER + "\n" for stage_id in STAGE_ORDER})
    stages.append({"id": "export", "source": SOURCE_MARKER + "\n"})
    with TestClient(app) as client:
        headers = login(client, "check-viewer")
        response = client.post(
            "/api/v2/pipeline-check",
            headers=headers,
            json={"stages": stages},
        )
        assert response.status_code == 422
        assert SOURCE_MARKER not in response.text
