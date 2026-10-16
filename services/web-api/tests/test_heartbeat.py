from types import SimpleNamespace

import pytest
from workbench import heartbeat


def test_reader_contention_is_bounded_and_atomic(tmp_path, monkeypatch):
    path = tmp_path / "worker.json"
    path.write_text("old", encoding="utf-8")
    original = heartbeat.os.replace
    attempts = []

    def replace(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            assert path.read_text(encoding="utf-8") == "old"
            raise PermissionError("reader")
        original(source, destination)

    monkeypatch.setattr(heartbeat, "os", SimpleNamespace(name="nt", replace=replace))
    monkeypatch.setattr(heartbeat.time, "sleep", lambda _: None)
    heartbeat.publish(path, "new")
    assert len(attempts) == 3
    assert path.read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("system,expected", [("nt", 5), ("posix", 1)])
def test_permanent_failure_propagates_and_cleans_up(
    tmp_path, monkeypatch, system, expected
):
    attempts = []

    def replace(*_):
        attempts.append(1)
        raise PermissionError("denied")

    monkeypatch.setattr(heartbeat, "os", SimpleNamespace(name=system, replace=replace))
    monkeypatch.setattr(heartbeat.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        heartbeat.publish(tmp_path / "worker.json", "new")
    assert len(attempts) == expected
    assert not list(tmp_path.glob("*.tmp"))
