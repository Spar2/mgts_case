from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException
from test_platform import pg
from workbench.platform import principal
from workbench.rate_limits import RateLimiter


def test_shared_login_limits_and_outage(monkeypatch, pg):
    first, second = RateLimiter(), RateLimiter()
    for _ in range(5):
        first.check("login", "qa-ip", 30, "name")
    with pytest.raises(HTTPException) as denied:
        second.check("login", "qa-ip", 30, "name")
    assert (
        denied.value.status_code == 429 and int(denied.value.headers["Retry-After"]) > 0
    )
    for i in range(24):
        second.check("login", "qa-ip", 30, str(i))
    with pytest.raises(HTTPException) as denied:
        first.check("login", "qa-ip", 30, "new-name")
    assert denied.value.status_code == 429
    monkeypatch.setenv("DTS_REDIS_URL", "redis://127.0.0.1:1/0")
    with pytest.raises(HTTPException) as outage:
        RateLimiter().check("login", "ip", 30, "user")
    assert outage.value.status_code == 503


def test_atomic_queue_capacity(pg):
    def submit(i):
        try:
            return pg.enqueue("prepare_views", {"index": i})
        except HTTPException as error:
            return error.status_code

    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(submit, range(16)))
    assert sum(isinstance(r, dict) for r in results) == 10
    assert results.count(429) == 6
    assert pg.claim("one") and pg.claim("two")
    assert pg.claim("three") is None
    from workbench.accounts import add_user

    user = add_user(pg, "analyst", "QA-only-password-123456", "Analyst", "analyst")
    scope = principal.set({"id": user["id"], "project_id": "local"})
    try:
        job = pg.enqueue("prepare_views", {"index": 100}, key="user-action")
        assert (
            pg.enqueue("prepare_views", {"index": 100}, key="user-action")["id"]
            == job["id"]
        )
        with pytest.raises(HTTPException) as denied:
            pg.enqueue("prepare_views", {"index": 101})
        assert denied.value.status_code == 429
    finally:
        principal.reset(scope)
