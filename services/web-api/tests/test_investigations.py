"""Work documents use real isolated PostgreSQL; fixtures are technical, not labels."""

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
from test_platform import pg as platform_pg, login
from test_workbench import fixture as native_fixture
from workbench.accounts import add_user
from workbench.api import create_app

pg = platform_pg


def setup(pg, tmp_path):
    root, queue, native = native_fixture.__wrapped__(tmp_path)
    for name, role in (("work-admin", "admin"), ("work-viewer", "viewer")):
        add_user(pg, name, "QA-only-password-123456", name, role)
    app = create_app(root / "records", queue, root / "runtime")
    body = {
        "title": "Technical work",
        "context": {
            "source_sha256": native.manifest["source_sha256"],
            "region": {
                "record_id": "5",
                "row_start": 0,
                "row_end": 3,
                "column_start": 0,
                "column_end": 3,
            },
            "profile": 1,
            "column": 1,
        },
    }
    return app, body


def test_work_idempotency_revision_history_and_viewer_access(pg, tmp_path):
    app, body = setup(pg, tmp_path)
    with TestClient(app) as client:
        headers = {**login(client, "work-admin"), "Idempotency-Key": "work-create-1"}
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(
                pool.map(
                    lambda _: client.post(
                        "/api/v2/investigations", json=body, headers=headers
                    ),
                    range(2),
                )
            )
        assert all(r.status_code == 201 for r in responses), [r.text for r in responses]
        work = responses[0].json()
        assert len({r.json()["id"] for r in responses}) == 1
        route = "/api/v2/investigations/" + work["id"]
        changed = client.patch(
            route,
            json={"expected_revision": 1, "notes": "Observed shape; cause unknown"},
            headers=headers,
        )
        assert changed.status_code == 200 and changed.json()["revision"] == 2
        assert (
            client.patch(
                route, json={"expected_revision": 1, "notes": "stale"}, headers=headers
            ).status_code
            == 409
        )
        assert (
            client.post("/api/v2/investigations", json=body, headers=headers).json()
            == work
        )
        assert (
            client.post(
                "/api/v2/investigations",
                json={**body, "title": "different"},
                headers=headers,
            ).status_code
            == 409
        )
        events = client.get(route + "/history").json()["items"]
        assert [e["revision"] for e in events] == [2, 1]
        assert all(e["author_name"] == "work-admin" for e in events)
        viewer = login(client, "work-viewer")
        assert client.get(route).status_code == 200
        assert (
            client.patch(
                route,
                json={"expected_revision": 2, "notes": "forbidden"},
                headers=viewer,
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/v2/investigations",
                json=body,
                headers={**viewer, "Idempotency-Key": "viewer-create"},
            ).status_code
            == 403
        )


def test_work_source_and_results_cannot_be_substituted(pg, tmp_path):
    app, body = setup(pg, tmp_path)
    with TestClient(app) as client:
        headers = {**login(client, "work-admin"), "Idempotency-Key": "work-source-1"}
        assert (
            client.post(
                "/api/v2/investigations",
                json={
                    **body,
                    "context": {**body["context"], "source_sha256": "0" * 64},
                },
                headers=headers,
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/api/v2/investigations",
                json={**body, "context": {**body["context"], "column": 4}},
                headers=headers,
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v2/investigations",
                json={**body, "context": {**body["context"], "run_ids": ["missing"]}},
                headers=headers,
            ).status_code
            == 409
        )
        work = client.post("/api/v2/investigations", json=body, headers=headers).json()
        assert (
            client.patch(
                "/api/v2/investigations/" + work["id"],
                json={
                    "expected_revision": 1,
                    "context": {**body["context"], "source_sha256": "f" * 64},
                },
                headers=headers,
            ).status_code
            == 409
        )
        client.app.state.platform.put(
            "investigation",
            "foreign",
            {**work, "id": "foreign", "project_id": "foreign"},
        )
        assert client.get("/api/v2/investigations/foreign").status_code == 403
        assert all(
            w["id"] != "foreign" for w in client.get("/api/v2/investigations").json()
        )


def test_case_link_preserves_snapshot_and_monitoring_requires_confirmed_range(
    pg, tmp_path
):
    app, body = setup(pg, tmp_path)
    with TestClient(app) as client:
        headers = {**login(client, "work-admin"), "Idempotency-Key": "work-link-1"}
        case = client.post(
            "/api/v2/cases",
            json={
                "title": "QA immutable case",
                "region": body["context"]["region"],
                "profile": 1,
                "source_sha256": body["context"]["source_sha256"],
            },
            headers=headers,
        )
        assert case.status_code == 201, case.text
        case = case.json()
        bundle = client.get("/api/v2/cases/" + case["id"] + "/bundle").content
        work = client.post(
            "/api/v2/investigations",
            json={**body, "case_ids": [case["id"]]},
            headers=headers,
        ).json()
        response = client.patch(
            "/api/v2/investigations/" + work["id"],
            json={
                "expected_revision": 1,
                "context": {
                    **body["context"],
                    "region": {
                        **body["context"]["region"],
                        "row_start": 1,
                        "row_end": 2,
                    },
                },
            },
            headers=headers,
        )
        assert response.status_code == 200 and response.json()["case_ids"] == [
            case["id"]
        ]
        assert client.get("/api/v2/cases/" + case["id"] + "/bundle").content == bundle
        other = client.post(
            "/api/v2/investigations",
            json=body,
            headers={**headers, "Idempotency-Key": "work-link-other"},
        ).json()
        link = client.post(
            "/api/v2/investigations/" + other["id"] + "/cases",
            json={"case_id": case["id"]},
            headers=headers,
        )
        assert link.status_code == 200 and link.json()["case_ids"] == [case["id"]]
        again = client.post(
            "/api/v2/investigations/" + other["id"] + "/cases",
            json={"case_id": case["id"]},
            headers=headers,
        )
        assert again.json()["revision"] == link.json()["revision"]
        app.state.platform.put(
            "stream",
            "confirmed",
            {
                "id": "confirmed",
                "project_id": "local",
                "record_id": "5",
                "source_sha256": body["context"]["source_sha256"],
                "verified_persisted": 1,
            },
        )
        proposal = {**body, "origin": "monitoring", "stream_id": "confirmed"}
        assert (
            client.post(
                "/api/v2/investigations",
                json=proposal,
                headers={**headers, "Idempotency-Key": "stream-create-1"},
            ).status_code
            == 409
        )
        proposal["context"] = {
            **body["context"],
            "region": {**body["context"]["region"], "row_end": 2},
        }
        assert (
            client.post(
                "/api/v2/investigations",
                json=proposal,
                headers={**headers, "Idempotency-Key": "stream-create-1"},
            ).status_code
            == 201
        )


def test_review_request_freezes_context_and_survives_package_cache_restart(
    pg, tmp_path
):
    app, body = setup(pg, tmp_path)
    add_user(pg, "work-expert", "QA-only-password-123456", "work-expert", "expert")
    with pg.connect() as db:
        expert_id = db.execute(
            "SELECT id FROM users WHERE username='work-expert'"
        ).fetchone()[0]
    with TestClient(app) as client:
        headers = {
            **login(client, "work-admin"),
            "Idempotency-Key": "work-review-source",
        }
        work = client.post("/api/v2/investigations", json=body, headers=headers).json()
        url = "/api/v2/investigations/" + work["id"]
        request_body = {"expected_revision": 1, "assigned_to": expert_id}
        request_headers = {**headers, "Idempotency-Key": "work-review-request"}
        response = client.post(
            url + "/review-requests", json=request_body, headers=request_headers
        )
        assert response.status_code == 201, response.text
        review = response.json()
        assert (
            client.post(
                url + "/review-requests", json=request_body, headers=request_headers
            ).json()
            == review
        )
        assert (
            client.patch(
                url,
                json={
                    "expected_revision": 2,
                    "context": {
                        **body["context"],
                        "region": {
                            **body["context"]["region"],
                            "row_start": 1,
                            "row_end": 2,
                        },
                    },
                },
                headers=headers,
            ).status_code
            == 200
        )
        assert (
            client.get(url + "/review-requests").json()[0]["context"]["region"][
                "row_start"
            ]
            == 0
        )
        # Clear lazy in-memory registrations: the durable package must resolve again.
        app.state.packages = {
            k: v for k, v in app.state.packages.items() if k == "general"
        }
        expert_headers = login(client, "work-expert")
        params = {
            "package": review["package"],
            "reviewer": "work-expert",
            "independent": "true",
        }
        queue = client.get("/api/v1/queue", params=params)
        assert queue.status_code == 200, queue.text
        assert queue.json()["cards"][0]["row_bounds"] == [0, 3]
        assert (
            client.get(
                "/api/v1/queue/" + review["review_id"] + "/methods", params=params
            ).status_code
            == 403
        )
        answer = {
            "queue_sha256": review["queue_sha256"],
            "review_id": review["review_id"],
            "expected_revision": 0,
            "reviewer": "work-expert",
            "reviewer_type": "expert",
            "temperature_episode": "unknown",
            "phenomenon": "unknown",
            "notes": "",
        }
        saved = client.post(
            "/api/v1/reviews",
            params=params,
            json=answer,
            headers={**expert_headers, "Idempotency-Key": "work-expert-answer"},
        )
        assert saved.status_code == 200, saved.text
        assert saved.json()["training_target"] is None
        assert (
            client.get(
                "/api/v1/queue/" + review["review_id"] + "/methods", params=params
            ).status_code
            == 200
        )
        inbox = client.get("/api/v2/review-inbox").json()["items"]
        assert inbox[0]["status"] == "answered" and "context" not in inbox[0]
        assert (
            client.post(
                "/api/v2/review-progress",
                json={
                    "package": review["package"],
                    "review_id": review["review_id"],
                    "state": "skipped",
                },
                headers=expert_headers,
            ).status_code
            == 200
        )
        assert (
            client.get(
                "/api/v2/review-progress", params={"package": review["package"]}
            ).json()["items"][review["review_id"]]
            == "skipped"
        )
        # An explicit skip does not manufacture a different answer or training target.
        assert (
            client.get("/api/v1/queue", params=params).json()["annotations"][
                review["review_id"]
            ]["temperature_episode"]
            == "unknown"
        )
        frozen = {
            "title": "QA expert window",
            "notes": "No accident claim",
            "region": body["context"]["region"],
            "profile": 1,
            "source_sha256": body["context"]["source_sha256"],
            "review": {
                "package": review["package"],
                "review_id": review["review_id"],
                "queue_sha256": review["queue_sha256"],
                "expected_revision": 1,
            },
        }
        case = client.post(
            "/api/v2/cases",
            json=frozen,
            headers={**expert_headers, "Idempotency-Key": "expert-frozen-case"},
        )
        assert case.status_code == 201, case.text
        case = case.json()
        assert case["result"]["evidence"]["expert_answer"]["revision"] == 1
        link = client.post(
            url + "/cases", json={"case_id": case["id"]}, headers=expert_headers
        )
        assert link.status_code == 200, link.text
        archive = client.get("/api/v2/cases/" + case["id"] + "/bundle").content
        changed = {**answer, "expected_revision": 1, "notes": "New revision only"}
        assert (
            client.post(
                "/api/v1/reviews",
                params=params,
                json=changed,
                headers={**expert_headers, "Idempotency-Key": "work-expert-answer-v2"},
            ).status_code
            == 200
        )
        assert client.get("/api/v2/cases/" + case["id"] + "/bundle").content == archive
        status = client.get(url + "/review-requests").json()[0]
        assert status["answer_revision"] == 2
