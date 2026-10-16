#!/usr/bin/env python3
"""Xeon self-hosted CI poller (stdlib only, no secrets inside).

Watches origin/main + origin/web in a bare mirror; on new SHA runs
/opt/ci/run/run.sh <branch> <sha> and records state. Idempotent no-op
when nothing changed. SSH via read-only deploy key (path below must exist).
"""
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

MIRROR = Path("/opt/ci/repos/haklolarp.git")
ORIGIN = "git@github.com:werd0y45/haklolarp.git"
KEY = "/home/hakloboss/.ssh/haklolarp_deploy"
KNOWN = "/home/hakloboss/.ssh/known_hosts"
STATE = Path("/opt/ci/state.json")
RUNSH = "/opt/ci/run/run.sh"
BRANCHES = ("main", "web")
SSH = f"ssh -i {KEY} -o StrictHostKeyChecking=yes -o UserKnownHostsFile={KNOWN}"


def sh(*args):
    env = dict(os.environ)
    env["GIT_SSH_COMMAND"] = SSH
    return subprocess.run(
        args, check=True, capture_output=True, text=True, env=env
    ).stdout.strip()


def main() -> None:
    if not (MIRROR / "HEAD").exists():
        MIRROR.parent.mkdir(parents=True, exist_ok=True)
        sh("git", "clone", "--bare", ORIGIN, str(MIRROR))
        # Bare clone maps heads/* -> heads/*; we address origin/* below.
        sh(
            "git", "-C", str(MIRROR), "config",
            "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*",
        )
    sh("git", "-C", str(MIRROR), "fetch", "origin", *BRANCHES)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for br in BRANCHES:
        sha = sh("git", "-C", str(MIRROR), "rev-parse", f"origin/{br}")
        if state.get(br, {}).get("sha") == sha:
            continue
        rc = subprocess.run([RUNSH, br, sha]).returncode
        # Record always (even red): CI processed this sha; status lives in
        # journal + ntfy. Raise after recording so the timer shows failure.
        state[br] = {
            "sha": sha,
            "rc": rc,
            "at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state, indent=2) + "\n")
        if rc:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
