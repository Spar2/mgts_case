"""Local accounts with revocable opaque sessions and project-scoped permissions."""

import argparse
import asyncio
import getpass
import hashlib
import os
import secrets
import time
import uuid
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from psycopg import OperationalError
from psycopg.errors import UniqueViolation
from pydantic import BaseModel, ConfigDict, Field

from .account_lifecycle import install_lifecycle
from .platform import Platform, principal

passwords = PasswordHasher()


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=12, max_length=512)


class NewUser(Credentials):
    display_name: str = Field(min_length=1, max_length=100)
    role: str = "viewer"
    project_id: str = "local"
    current_password: str | None = Field(default=None, max_length=512)


def add_user(platform, username, password, display_name, role="admin", project="local"):
    username, display_name = username.strip().lower(), display_name.strip()
    if (
        not username
        or not display_name
        or len(username) > 100
        or len(display_name) > 100
    ):
        raise ValueError(
            "Nonempty username and display name, at most 100 characters, required"
        )
    if len(password) < 12 or role not in ("viewer", "expert", "analyst", "admin"):
        raise ValueError(
            "Password must contain at least 12 characters and role must be valid"
        )
    key = uuid.uuid4().hex
    with platform.connect() as db:
        db.execute(
            "INSERT INTO users(id,username,display_name,password_hash) VALUES(%s,%s,%s,%s)",
            (
                key,
                username.strip().lower(),
                display_name.strip(),
                passwords.hash(password),
            ),
        )
        db.execute("INSERT INTO memberships VALUES(%s,%s,%s)", (project, key, role))
    return {"id": key, "username": username, "display_name": display_name, "role": role}


def authenticate(platform, request):
    token = request.cookies.get("dts_session", "")
    if not token or len(token) > 256:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    project = request.headers.get("x-dts-project", "local")
    with platform.connect() as db:
        row = db.execute(
            "SELECT u.id,u.username,u.display_name,m.role,s.csrf,s.expires,s.last_seen FROM sessions s JOIN users u ON u.id=s.user_id JOIN memberships m ON m.user_id=u.id WHERE s.digest=%s AND m.project_id=%s AND u.enabled AND s.expires>%s AND s.last_seen>%s FOR UPDATE OF s",
            (digest, project, time.time(), time.time() - 1800),
        ).fetchone()
    return (
        dict(
            zip(
                (
                    "id",
                    "username",
                    "display_name",
                    "role",
                    "csrf",
                    "expires",
                    "last_seen",
                ),
                row,
            ),
            project_id=project,
        )
        if row
        else None
    )


def touch_session(platform, request):
    token = request.cookies.get("dts_session", "")
    with platform.connect() as db:
        db.execute(
            "UPDATE sessions SET last_seen=%s WHERE digest=%s",
            (time.time(), hashlib.sha256(token.encode()).hexdigest()),
        )


async def session_is_current(platform, request, expected):
    """Streams must stop on revocation, expiry, scope changes or storage failure."""
    try:
        active = await asyncio.to_thread(authenticate, platform, request)
    except OperationalError:
        return False
    return bool(
        active
        and expected
        and all(active[k] == expected[k] for k in ("id", "project_id", "role"))
    )


def install_accounts(app):
    install_lifecycle(app, passwords)
    dummy = passwords.hash(secrets.token_urlsafe(30))

    @app.post("/api/v2/auth/login")
    def login(body: Credentials, request: Request):
        platform = app.state.platform
        if not platform:
            raise HTTPException(503, "Configure PostgreSQL to use accounts")
        app.state.limiter.check(
            "login", request.client.host, 30, body.username.strip().lower()
        )
        now = time.time()
        with platform.connect() as db:
            user = db.execute(
                "SELECT id,display_name,password_hash,enabled FROM users WHERE username=%s",
                (body.username.strip().lower(),),
            ).fetchone()
        try:
            passwords.verify(user[2] if user else dummy, body.password)
        except VerificationError:
            raise HTTPException(401, "Incorrect username or password") from None
        if not user or not user[3]:
            raise HTTPException(401, "Incorrect username or password")
        token = secrets.token_urlsafe(48)
        csrf = secrets.token_urlsafe(32)
        with platform.connect() as db:
            # Serialize issuance with password replacement and account disabling.
            # Argon2 verification stays outside the lock; an intervening change
            # invalidates the verified snapshot instead of issuing a stale session.
            current = db.execute(
                "SELECT password_hash,enabled FROM users WHERE id=%s FOR UPDATE",
                (user[0],),
            ).fetchone()
            if (
                not current
                or not current[1]
                or not secrets.compare_digest(current[0], user[2])
            ):
                raise HTTPException(401, "Incorrect username or password")
            if passwords.check_needs_rehash(current[0]):
                db.execute(
                    "UPDATE users SET password_hash=%s WHERE id=%s",
                    (passwords.hash(body.password), user[0]),
                )
            now = time.time()
            previous = request.cookies.get("dts_session", "")
            if previous:
                db.execute(
                    "DELETE FROM sessions WHERE digest=%s",
                    (hashlib.sha256(previous.encode()).hexdigest(),),
                )
            db.execute("DELETE FROM sessions WHERE expires<%s", (now,))
            db.execute(
                "INSERT INTO sessions(digest,user_id,csrf,expires) VALUES(%s,%s,%s,%s)",
                (
                    hashlib.sha256(token.encode()).hexdigest(),
                    user[0],
                    csrf,
                    now + 43200,
                ),
            )
        platform.audit("login", user[0], actor=user[0])
        response = JSONResponse({"display_name": user[1], "csrf": csrf})
        response.set_cookie(
            "dts_session",
            token,
            httponly=True,
            samesite="strict",
            secure=os.environ.get("DTS_SECURE_COOKIES") == "true",
            max_age=43200,
            path="/",
        )
        return response

    @app.get("/api/v2/auth/me")
    def me():
        user = principal.get()
        if not user:
            raise HTTPException(401, "Sign in first")
        return user

    @app.post("/api/v2/auth/logout")
    def logout(request: Request):
        with app.state.platform.connect() as db:
            db.execute(
                "DELETE FROM sessions WHERE digest=%s",
                (
                    hashlib.sha256(
                        request.cookies.get("dts_session", "").encode()
                    ).hexdigest(),
                ),
            )
        response = JSONResponse({"signed_out": True})
        response.delete_cookie("dts_session", path="/")
        return response

    @app.post("/api/v2/accounts")
    def create_user(body: NewUser):
        if principal.get()["role"] != "admin":
            raise HTTPException(403, "Administrator required")
        if body.project_id != principal.get()["project_id"]:
            raise HTTPException(403, "Wrong project")
        if body.role not in ("viewer", "expert", "analyst", "admin"):
            raise HTTPException(422, "Invalid role")
        if body.role == "admin":
            with app.state.platform.connect() as db:
                stored = db.execute(
                    "SELECT password_hash FROM users WHERE id=%s",
                    (principal.get()["id"],),
                ).fetchone()
            try:
                passwords.verify(stored[0], body.current_password or "")
            except VerificationError:
                raise HTTPException(
                    401, "Current administrator password required"
                ) from None
        try:
            user = add_user(
                app.state.platform,
                body.username,
                body.password,
                body.display_name,
                body.role,
                body.project_id,
            )
        except UniqueViolation:
            raise HTTPException(409, "Этот логин уже занят. Выберите другой.") from None
        app.state.platform.audit("user.created", user["id"])
        return user

    @app.get("/api/v2/accounts")
    def users():
        if principal.get()["role"] != "admin":
            raise HTTPException(403, "Administrator required")
        with app.state.platform.connect() as db:
            return [
                dict(
                    zip(
                        (
                            "id",
                            "username",
                            "display_name",
                            "role",
                            "project_id",
                            "enabled",
                        ),
                        r,
                    )
                )
                for r in db.execute(
                    "SELECT u.id,u.username,u.display_name,m.role,m.project_id,u.enabled FROM users u JOIN memberships m ON u.id=m.user_id WHERE m.project_id=%s ORDER BY username",
                    (principal.get()["project_id"],),
                ).fetchall()
            ]

    @app.post("/api/v2/accounts/{key}/disable")
    def disable(key: str):
        if principal.get()["role"] != "admin" or key == principal.get()["id"]:
            raise HTTPException(403, "Cannot disable this account")
        with app.state.platform.connect() as db:
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                ("accounts:" + principal.get()["project_id"],),
            )
            memberships = db.execute(
                "SELECT project_id FROM memberships WHERE user_id=%s", (key,)
            ).fetchall()
            if not memberships or any(
                r[0] != principal.get()["project_id"] for r in memberships
            ):
                raise HTTPException(403, "Cannot disable an account in another project")
            remaining = db.execute(
                "SELECT count(*) FROM users u JOIN memberships m ON u.id=m.user_id WHERE m.project_id=%s AND m.role='admin' AND u.enabled AND u.id<>%s",
                (principal.get()["project_id"], key),
            ).fetchone()[0]
            if remaining == 0:
                raise HTTPException(
                    409, "Проект должен иметь действующего администратора"
                )
            db.execute("UPDATE users SET enabled=false WHERE id=%s", (key,))
            db.execute("DELETE FROM sessions WHERE user_id=%s", (key,))
        app.state.platform.audit("user.disabled", key)
        return {"disabled": key}

    @app.post("/api/v2/accounts/{key}/enable")
    def enable(key: str):
        if principal.get()["role"] != "admin":
            raise HTTPException(403, "Administrator required")
        with app.state.platform.connect() as db:
            memberships = db.execute(
                "SELECT project_id FROM memberships WHERE user_id=%s", (key,)
            ).fetchall()
            if not memberships or any(
                r[0] != principal.get()["project_id"] for r in memberships
            ):
                raise HTTPException(403, "Cannot change an account in another project")
            db.execute("UPDATE users SET enabled=true WHERE id=%s", (key,))
        app.state.platform.audit("user.enabled", key, principal.get()["project_id"])
        return {"enabled": key}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("username")
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Reset an existing administrator without changing review authors",
    )
    args = parser.parse_args()
    platform = Platform(
        os.environ["DTS_DATABASE_URL"], Path(os.environ.get("DTS_RUNTIME", ".runtime"))
    )
    password = getpass.getpass("New administrator password (12+ characters): ")
    if getpass.getpass("Repeat password: ") != password:
        raise ValueError("Passwords differ")
    if args.recover:
        with platform.connect() as db:
            user = db.execute(
                "SELECT u.id FROM users u JOIN memberships m ON m.user_id=u.id WHERE u.username=%s AND m.role='admin' FOR UPDATE OF u",
                (args.username.strip().lower(),),
            ).fetchone()
            if not user or len(password) < 12:
                raise ValueError(
                    "Existing administrator and 12-character password required"
                )
            db.execute(
                "UPDATE users SET password_hash=%s,enabled=true WHERE id=%s",
                (passwords.hash(password), user[0]),
            )
            db.execute("DELETE FROM sessions WHERE user_id=%s", (user[0],))
            db.execute(
                "UPDATE password_resets SET revoked=true WHERE user_id=%s", (user[0],)
            )
        platform.audit("admin.operator_recovery", user[0], actor="operator")
        print("Administrator recovered; sessions revoked and authorship preserved.")
    else:
        add_user(platform, args.username, password, args.name)
        print("Administrator created. Sign in at the local workbench.")


if __name__ == "__main__":
    main()
