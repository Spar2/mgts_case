"""Start an isolated PostgreSQL QA server, or delete only its generated schema.

Uses prepared data read-only. Credentials and connection URL stay in .runtime.
Requires DTS_TEST_PG_URL pointing at a dedicated local/CI PostgreSQL service.
"""

import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import psycopg
from psycopg import sql

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "services/web-api"))
from workbench.accounts import add_user
from workbench.platform import Platform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("serve", "drop"))
    parser.add_argument(
        "--state", type=Path, default=BASE / ".runtime/auth-qa/state.json"
    )
    parser.add_argument("--records", type=Path)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    statefile = args.state.resolve()
    if not statefile.is_relative_to(BASE / ".runtime"):
        parser.error("QA state must stay inside this checkout .runtime")
    root = statefile.parent
    if args.action == "drop":
        state = json.loads(statefile.read_text(encoding="utf-8"))
        schema = state["schema"]
        if (
            not schema.startswith("qa_browser_")
            or len(schema) != len("qa_browser_") + 32
        ):
            raise ValueError("Refusing to remove an unrelated schema")
        with psycopg.connect(state["base_url"], autocommit=True) as db:
            db.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )
        print("Isolated browser schema removed")
        return
    if statefile.exists():
        parser.error("Choose a new private QA state path")
    if not args.records or not args.queue:
        parser.error("--records and --queue are required")
    root.mkdir(parents=True, exist_ok=True)
    base_url = os.environ["DTS_TEST_PG_URL"]
    schema = "qa_browser_" + uuid.uuid4().hex
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query))
    query["options"] = "-c search_path=" + schema
    url = urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query, quote_via=quote),
            parts.fragment,
        )
    )
    with psycopg.connect(base_url, autocommit=True) as db:
        db.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    runtime = root / "runtime"
    platform = Platform(url, runtime)
    credentials = {"username": "qa-admin", "password": secrets.token_urlsafe(24)}
    add_user(
        platform,
        credentials["username"],
        credentials["password"],
        "QA ONLY — технический аккаунт",
        "admin",
    )
    statefile.write_text(
        json.dumps(
            {
                "base_url": base_url,
                "url": url,
                "schema": schema,
                "credentials": credentials,
            }
        ),
        encoding="utf-8",
    )
    # The cookie exists only in the isolated browser context. No secrets in CLI text.
    import httpx

    environment = {
        **os.environ,
        "PYTHONPATH": str(BASE / "services/web-api"),
        "DTS_DATABASE_URL": url,
        "DTS_RECORDS": str(args.records.resolve()),
        "DTS_QUEUE": str(args.queue.resolve()),
        "DTS_RUNTIME": str(runtime),
        "DTS_WEB_DIST": str(BASE / "apps/web/dist"),
        "DTS_QA_MODE": "isolated",
        "DTS_ALLOWED_ORIGINS": f"http://127.0.0.1:{args.port},http://localhost:{args.port}",
    }
    if args.pack:
        environment["DTS_RESEARCH_PACK"] = str(args.pack.resolve())
    else:
        environment.pop("DTS_RESEARCH_PACK", None)
    with (root / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "workbench.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--no-access-log",
            ],
            cwd=BASE,
            env=environment,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        (root / "server.pid").write_text(str(process.pid))
        signal.signal(signal.SIGTERM, lambda *_: process.terminate())
        try:
            import time

            with httpx.Client(
                base_url=f"http://127.0.0.1:{args.port}", trust_env=False, timeout=30
            ) as client:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            "Isolated server exited; inspect private server.log"
                        )
                    try:
                        health = client.get("/api/v1/health")
                        if health.is_success and health.json().get("qa_mode"):
                            break
                    except httpx.RequestError:
                        pass
                    time.sleep(0.5)
                else:
                    raise RuntimeError("Isolated server startup timed out")
                client.post("/api/v2/auth/login", json=credentials).raise_for_status()
                (root / "browser-auth.json").write_text(
                    json.dumps(
                        {
                            "cookies": [
                                {
                                    "name": c.name,
                                    "value": c.value,
                                    "domain": "127.0.0.1",
                                    "path": "/",
                                    "expires": -1,
                                    "httpOnly": True,
                                    "secure": False,
                                    "sameSite": "Strict",
                                }
                                for c in client.cookies.jar
                            ],
                            "origins": [],
                        }
                    ),
                    encoding="utf-8",
                )
            print(f"Isolated authenticated QA ready on {args.port}", flush=True)
            process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)


if __name__ == "__main__":
    main()
