"""Real PostgreSQL/Redis lifecycle acceptance in disposable namespaces."""

import time
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
from test_platform import pg, login, native_fixture
from workbench.accounts import add_user
from workbench.api import create_app

PASSWORD = "QA-only-password-123456"


def test_invitation_atomic_and_recovery(pg, tmp_path):
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    admin = add_user(pg, "admin", PASSWORD, "Admin")
    with TestClient(
        create_app(root / "records", queue, tmp_path / "runtime")
    ) as client:
        headers = login(client, "admin")
        assert (
            client.post(
                "/api/v2/invitations", json={"role": "admin"}, headers=headers
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/api/v2/invitations", json={"project_id": "other"}, headers=headers
            ).status_code
            == 403
        )
        invitation = client.post("/api/v2/invitations", json={}, headers=headers).json()
        token = invitation["link"].split("#")[1]
        body = {
            "token": token,
            "username": "member",
            "display_name": "Member",
            "password": PASSWORD,
        }
        assert (
            client.post(
                "/api/v2/auth/activate", json={**body, "role": "admin"}
            ).status_code
            == 422
        )
        with ThreadPoolExecutor(2) as pool:
            results = list(
                pool.map(
                    lambda _: (
                        client.post("/api/v2/auth/activate", json=body).status_code
                    ),
                    range(2),
                )
            )
        assert sorted(results) == [200, 410]
        member_headers = login(client, "member")
        user = client.get("/api/v2/auth/me").json()
        assert user["role"] == "viewer"
        assert client.get("/api/v2/auth/sessions").json()[0]["current"]
        assert (
            client.post(
                "/api/v2/auth/password",
                json={
                    "current_password": PASSWORD,
                    "password": "updated-member-password",
                },
            ).status_code
            == 403
        )
        headers = login(client, "admin")
        reset = client.post(
            f"/api/v2/accounts/{user['id']}/recovery", json={}, headers=headers
        ).json()
        replacement = {
            "token": reset["link"].split("#")[1],
            "password": "updated-member-password",
        }
        assert client.post("/api/v2/auth/recover", json=replacement).status_code == 200
        assert client.post("/api/v2/auth/recover", json=replacement).status_code == 410
        with pg.connect() as db:
            assert (
                db.execute(
                    "SELECT count(*) FROM sessions WHERE user_id=%s", (user["id"],)
                ).fetchone()[0]
                == 0
            )
            assert db.execute(
                "SELECT password_hash FROM users WHERE id=%s", (admin["id"],)
            ).fetchone()
        assert (
            client.post(
                "/api/v2/auth/login",
                json={"username": "member", "password": "updated-member-password"},
            ).status_code
            == 200
        )
        with pg.connect() as db:
            db.execute(
                "UPDATE sessions SET last_seen=%s WHERE user_id=%s",
                (time.time() - 1801, user["id"]),
            )
        assert client.get("/api/v2/auth/me").status_code == 401


def test_invitation_revocation_expiry_collision(pg, tmp_path):
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    add_user(pg, "admin", PASSWORD, "Admin")
    with TestClient(
        create_app(root / "records", queue, tmp_path / "runtime")
    ) as client:
        headers = login(client, "admin")

        def issue():
            return client.post(
                "/api/v2/invitations", json={"role": "expert"}, headers=headers
            ).json()

        invitation = issue()
        body = {
            "token": invitation["link"].split("#")[1],
            "username": "admin",
            "display_name": "Name",
            "password": PASSWORD,
        }
        assert client.post("/api/v2/auth/activate", json=body).status_code == 409
        assert (
            client.post(
                "/api/v2/auth/activate", json={**body, "username": "expert"}
            ).status_code
            == 200
        )
        invitation = issue()
        assert (
            client.post(
                f"/api/v2/invitations/{invitation['id']}/revoke",
                json={},
                headers=headers,
            ).status_code
            == 200
        )
        body["token"] = invitation["link"].split("#")[1]
        assert client.post("/api/v2/auth/activate", json=body).status_code == 410

        invitation = issue()
        with pg.connect() as db:
            db.execute(
                "UPDATE invitations SET expires=0 WHERE id=%s", (invitation["id"],)
            )
        body["token"] = invitation["link"].split("#")[1]
        assert client.post("/api/v2/auth/activate", json=body).status_code == 410


def test_password_change_and_session_revocation(pg, tmp_path):
    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    member = add_user(pg, "member", PASSWORD, "Member", "expert")
    app = create_app(root / "records", queue, tmp_path / "runtime")
    with TestClient(app) as client:
        headers = login(client, "member")
        old_cookie = client.cookies.get("dts_session")
        # A second device retains a distinct session. Logging in again in the
        # same cookie jar now rotates and revokes that jar's previous session.
        client.cookies.clear()
        headers = login(client, "member")
        assert len(client.get("/api/v2/auth/sessions").json()) == 2
        assert (
            client.post(
                "/api/v2/auth/sessions/others/revoke", json={}, headers=headers
            ).status_code
            == 200
        )
        assert len(client.get("/api/v2/auth/sessions").json()) == 1
        original_cookie = client.cookies.get("dts_session")
        client.cookies.set("dts_session", old_cookie)
        assert client.get("/api/v2/auth/me").status_code == 401
        client.cookies.clear()
        client.cookies.set("dts_session", original_cookie)
        assert (
            client.post(
                "/api/v2/auth/password",
                json={
                    "current_password": "incorrect-value",
                    "password": "new-member-password",
                },
                headers=headers,
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/api/v2/auth/password",
                json={"current_password": PASSWORD, "password": "new-member-password"},
                headers=headers,
            ).status_code
            == 200
        )
        assert client.get("/api/v2/auth/me").status_code == 401
        assert (
            client.post(
                "/api/v2/auth/login", json={"username": "member", "password": PASSWORD}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/api/v2/auth/login",
                json={"username": "member", "password": "new-member-password"},
            ).status_code
            == 200
        )
        with pg.connect() as db:
            assert (
                db.execute("SELECT id FROM users WHERE username='member'").fetchone()[0]
                == member["id"]
            )


def test_backup_preserves_roles_and_revokes_links(pg, tmp_path):
    from workbench.maintenance import backup, restore
    from workbench.platform import Platform
    from psycopg import sql
    import psycopg
    import os
    import uuid
    from urllib.parse import quote

    root, queue, _ = native_fixture.__wrapped__(tmp_path)
    user = add_user(pg, "admin", PASSWORD, "Admin")
    pg.put(
        "qa-proof",
        "artifact",
        {"result": {"purpose": "backup integrity acceptance only"}},
    )
    with TestClient(
        create_app(root / "records", queue, tmp_path / "runtime")
    ) as client:
        headers = login(client, "admin")
        client.post("/api/v2/invitations", json={}, headers=headers).raise_for_status()
        client.post(
            f"/api/v2/accounts/{user['id']}/recovery", json={}, headers=headers
        ).raise_for_status()
        client.post(
            "/api/public/v1/access-requests",
            json={"name": "QA", "contact": "channel", "reason": "QA only"},
        ).raise_for_status()
    snapshot = tmp_path / "snapshot"
    backup(pg, snapshot)
    base = os.environ["DTS_TEST_PG_URL"]
    schema = "qa_restore_" + uuid.uuid4().hex
    with psycopg.connect(base, autocommit=True) as db:
        db.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    url = (
        base
        + ("&" if "?" in base else "?")
        + "options="
        + quote("-c search_path=" + schema)
    )
    try:
        restored = Platform(url, tmp_path / "restored")
        restore(restored, snapshot)
        with restored.connect() as db:
            assert db.execute("SELECT user_id,role FROM memberships").fetchone() == (
                user["id"],
                "admin",
            )
            assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
            for table in ("invitations", "password_resets"):
                assert (
                    db.execute(
                        "SELECT count(*) FROM " + table + " WHERE revoked"
                    ).fetchone()[0]
                    == 1
                )
            assert db.execute("SELECT count(*) FROM access_requests").fetchone()[0] == 1
        assert (
            restored.get("qa-proof", "artifact")["result"]
            == pg.get("qa-proof", "artifact")["result"]
        )
        import pytest

        with pytest.raises(ValueError, match="empty target"):
            restore(restored, snapshot)
    finally:
        with psycopg.connect(base, autocommit=True) as db:
            db.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
