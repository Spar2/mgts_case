"""Single-use access links and account security; tokens are never persisted raw."""

import hashlib
import secrets
import time
import uuid
from typing import Literal

from fastapi import HTTPException, Request
from psycopg.errors import UniqueViolation
from pydantic import BaseModel, ConfigDict, Field

from .platform import principal


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Invite(StrictBody):
    project_id: str = "local"
    role: Literal["viewer", "expert", "analyst", "admin"] = "viewer"
    current_password: str | None = Field(default=None, max_length=512)


class Redeem(StrictBody):
    token: str = Field(min_length=40, max_length=128)
    username: str = Field(min_length=1, max_length=100)
    display_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=12, max_length=512)


class ChangePassword(StrictBody):
    current_password: str = Field(max_length=512)
    password: str = Field(min_length=12, max_length=512)


class ResetPassword(StrictBody):
    token: str = Field(min_length=40, max_length=128)
    password: str = Field(min_length=12, max_length=512)


def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def admin():
    user = principal.get()
    if not user or user["role"] != "admin":
        raise HTTPException(403, "Administrator required")
    return user


def install_lifecycle(app, passwords):
    from argon2.exceptions import VerificationError

    def verify(db, key, password):
        row = db.execute(
            "SELECT password_hash FROM users WHERE id=%s AND enabled", (key,)
        ).fetchone()
        try:
            if not row or not password:
                raise HTTPException(401, "Current password required")
            passwords.verify(row[0], password)
        except VerificationError:
            raise HTTPException(401, "Current password incorrect") from None

    @app.post("/api/v2/invitations")
    def create_invitation(body: Invite):
        user = admin()
        if body.project_id != user["project_id"]:
            raise HTTPException(403, "Wrong project")
        key, token, now = uuid.uuid4().hex, secrets.token_urlsafe(32), time.time()
        with app.state.platform.connect() as db:
            if body.role == "admin":
                verify(db, user["id"], body.current_password)
            db.execute(
                "INSERT INTO invitations(id,digest,project_id,role,created_by,created,expires) VALUES(%s,%s,%s,%s,%s,%s,%s)",
                (
                    key,
                    digest(token),
                    body.project_id,
                    body.role,
                    user["id"],
                    now,
                    now + 86400,
                ),
            )
        app.state.platform.audit("invitation.created", key, user["project_id"])
        return {"id": key, "link": "/activate#" + token, "expires": now + 86400}

    @app.get("/api/v2/invitations")
    def invitations():
        user = admin()
        with app.state.platform.connect() as db:
            return [
                dict(
                    zip(
                        ("id", "role", "created", "expires", "consumed", "revoked"), row
                    )
                )
                for row in db.execute(
                    "SELECT id,role,created,expires,consumed,revoked FROM invitations WHERE project_id=%s ORDER BY created DESC LIMIT 200",
                    (user["project_id"],),
                )
            ]

    @app.post("/api/v2/invitations/{key}/revoke")
    def revoke_invitation(key: str):
        user = admin()
        with app.state.platform.connect() as db:
            row = db.execute(
                "UPDATE invitations SET revoked=true WHERE id=%s AND project_id=%s RETURNING id",
                (key, user["project_id"]),
            ).fetchone()
            if not row:
                raise HTTPException(404, "Invitation not found")
        app.state.platform.audit("invitation.revoked", key, user["project_id"])
        return {"revoked": key}

    @app.post("/api/v2/auth/activate")
    def activate(body: Redeem, request: Request):
        app.state.limiter.check("activate", request.client.host, 10)
        if not app.state.platform:
            raise HTTPException(503, "Account storage unavailable")
        username, name = body.username.strip().lower(), body.display_name.strip()
        if not username or not name:
            raise HTTPException(422, "Login and name cannot be blank")
        password_hash = passwords.hash(body.password)
        key, now = uuid.uuid4().hex, time.time()
        try:
            with app.state.platform.connect() as db:
                row = db.execute(
                    "SELECT id,project_id,role FROM invitations WHERE digest=%s AND NOT revoked AND consumed IS NULL AND expires>%s FOR UPDATE",
                    (digest(body.token), now),
                ).fetchone()
                if not row:
                    raise HTTPException(
                        410, "Invitation expired, revoked or already used"
                    )
                db.execute(
                    "INSERT INTO users(id,username,display_name,password_hash) VALUES(%s,%s,%s,%s)",
                    (key, username, name, password_hash),
                )
                db.execute(
                    "INSERT INTO memberships VALUES(%s,%s,%s)", (row[1], key, row[2])
                )
                db.execute(
                    "UPDATE invitations SET consumed=%s WHERE id=%s", (now, row[0])
                )
                db.execute(
                    "INSERT INTO audit(project_id,actor,action,resource,at) VALUES(%s,%s,'invitation.accepted',%s,%s)",
                    (row[1], key, row[0], now),
                )
        except UniqueViolation:
            raise HTTPException(
                409, "Этот логин уже занят. Приглашение не потрачено."
            ) from None
        return {"created": True, "sign_in_required": True}

    @app.post("/api/v2/auth/password")
    def change_password(body: ChangePassword):
        user = principal.get()
        if not user:
            raise HTTPException(401, "Sign in first")
        hashed = passwords.hash(body.password)
        with app.state.platform.connect() as db:
            db.execute("SELECT id FROM users WHERE id=%s FOR UPDATE", (user["id"],))
            verify(db, user["id"], body.current_password)
            db.execute(
                "UPDATE users SET password_hash=%s WHERE id=%s", (hashed, user["id"])
            )
            db.execute("DELETE FROM sessions WHERE user_id=%s", (user["id"],))
            db.execute(
                "UPDATE password_resets SET revoked=true WHERE user_id=%s",
                (user["id"],),
            )
        app.state.platform.audit("password.changed", user["id"], user["project_id"])
        return {"sign_in_required": True}

    @app.post("/api/v2/accounts/{key}/recovery")
    def recovery(key: str):
        user = admin()
        token, link_id, now = secrets.token_urlsafe(32), uuid.uuid4().hex, time.time()
        with app.state.platform.connect() as db:
            scopes = db.execute(
                "SELECT project_id FROM memberships WHERE user_id=%s", (key,)
            ).fetchall()
            if any(scope[0] != user["project_id"] for scope in scopes):
                raise HTTPException(403, "Cannot recover an account in another project")
            target = db.execute(
                "SELECT u.id FROM users u JOIN memberships m ON m.user_id=u.id WHERE u.id=%s AND u.enabled AND m.project_id=%s FOR UPDATE OF u",
                (key, user["project_id"]),
            ).fetchone()
            if not target:
                raise HTTPException(404, "Account not found")
            db.execute(
                "INSERT INTO password_resets(id,digest,user_id,created_by,created,expires) VALUES(%s,%s,%s,%s,%s,%s)",
                (link_id, digest(token), key, user["id"], now, now + 1800),
            )
        app.state.platform.audit("recovery.issued", key, user["project_id"])
        return {"link": "/recover#" + token, "expires": now + 1800}

    @app.post("/api/v2/auth/recover")
    def reset_password(body: ResetPassword, request: Request):
        app.state.limiter.check("recover", request.client.host, 10)
        if not app.state.platform:
            raise HTTPException(503, "Account storage unavailable")
        hashed, now = passwords.hash(body.password), time.time()
        with app.state.platform.connect() as db:
            # Lock the user first: distinct outstanding links cannot race one another.
            target = db.execute(
                "SELECT user_id FROM password_resets WHERE digest=%s",
                (digest(body.token),),
            ).fetchone()
            if not target:
                raise HTTPException(410, "Recovery link unavailable")
            if not db.execute(
                "SELECT id FROM users WHERE id=%s AND enabled FOR UPDATE", (target[0],)
            ).fetchone():
                raise HTTPException(410, "Recovery link unavailable")
            row = db.execute(
                "SELECT id FROM password_resets WHERE digest=%s AND NOT revoked AND consumed IS NULL AND expires>%s FOR UPDATE",
                (digest(body.token), now),
            ).fetchone()
            if not row:
                raise HTTPException(410, "Recovery link unavailable")
            db.execute(
                "UPDATE users SET password_hash=%s WHERE id=%s", (hashed, target[0])
            )
            db.execute(
                "UPDATE password_resets SET revoked=true WHERE user_id=%s", (target[0],)
            )
            db.execute(
                "UPDATE password_resets SET consumed=%s WHERE id=%s", (now, row[0])
            )
            db.execute("DELETE FROM sessions WHERE user_id=%s", (target[0],))
        app.state.platform.audit("password.recovered", target[0], actor=target[0])
        return {"sign_in_required": True}

    @app.get("/api/v2/auth/sessions")
    def sessions(request: Request):
        user, now = principal.get(), time.time()
        if not user:
            raise HTTPException(401, "Sign in first")
        current = digest(request.cookies.get("dts_session", ""))
        with app.state.platform.connect() as db:
            return [
                {
                    "id": row[0],
                    "created": row[1],
                    "last_seen": row[2],
                    "expires": row[3],
                    "label": row[4],
                    "current": row[0] == current,
                }
                for row in db.execute(
                    "SELECT digest,created,last_seen,expires,label FROM sessions WHERE user_id=%s AND expires>%s AND last_seen>%s ORDER BY created DESC",
                    (user["id"], now, now - 1800),
                )
            ]

    @app.post("/api/v2/auth/activity")
    def activity():
        return {"active": True}

    @app.post("/api/v2/auth/sessions/{key}/revoke")
    def revoke_session(key: str, request: Request):
        user = principal.get()
        if not user:
            raise HTTPException(401, "Sign in first")
        current = digest(request.cookies.get("dts_session", ""))
        with app.state.platform.connect() as db:
            if key == "others":
                db.execute(
                    "DELETE FROM sessions WHERE user_id=%s AND digest<>%s",
                    (user["id"], current),
                )
            else:
                db.execute(
                    "DELETE FROM sessions WHERE user_id=%s AND digest=%s",
                    (user["id"], key),
                )
        return {"revoked": True}
