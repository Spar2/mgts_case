"""Dual-access контракт: HOST<->ORIGIN и loopback-only compose.

Чистый stdlib: pytest-раннер подхватывает файл как обычный модуль
(только `assert`, без фикстур/сети/docker).
"""

import os
import re
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
ALLOW = {"compose.yaml"}  # staging stub, wildcard-исключение pr-gate


def test_public_host_origin_consistent():
    host = os.environ.get("DTS_PUBLIC_HOST", "").strip()
    origin = os.environ.get("DTS_PUBLIC_ORIGIN", "").strip()
    if not host or not origin:
        return
    assert "://" not in host and " " not in host, host
    u = urlsplit(origin)
    assert u.scheme == "https", origin
    assert (u.hostname or "").lower() == host.lower(), (origin, host)


def test_compose_no_wildcard_ports():
    files = sorted(ROOT.glob("compose*.yaml")) + sorted(ROOT.glob("compose*.yml"))
    assert files, "no compose*.yaml at repo root"
    bad = []
    for f in files:
        if f.name in ALLOW:
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if re.search(r"!\w", line):
                continue  # compose merge-теги (!override/!reset) — не биндинг
            s = line.strip()
            inline = "ports:" in line and "[" in line and ":" in line
            listed = re.match(r'-\s*"[^"]*:\d+[^"]*"\s*(#.*)?$', s)
            hostip = "host_ip:" in line
            if (inline or listed or hostip) and (
                "127.0.0.1" not in line or "0.0.0.0" in line
            ):
                bad.append(f"{f.name}:{s}")
    assert not bad, f"non-loopback/wildcard ports: {bad}"
