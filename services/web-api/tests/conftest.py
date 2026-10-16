"""Explicit opt-in for legacy isolated fixtures, never inherited by deployments."""
import pytest


@pytest.fixture(autouse=True)
def isolated_test_mode(monkeypatch):
    monkeypatch.setenv("DTS_ENV", "test")
    monkeypatch.setenv("DTS_TEST_ALLOW_ANONYMOUS", "true")
