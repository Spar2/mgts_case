#!/usr/bin/env bash
# Xeon self-hosted CI run: worktree gates + deploy staging (always) + web (green only).
# Exit code = deploy status; gates status reported separately (main has lint debt).
# Logs: caller (journal). Worktree always cleaned (trap).
set -euo pipefail
BRANCH="${1:?usage: run.sh <branch> <sha>}"
SHA="${2:?usage: run.sh <branch> <sha>}"
SHA7="${SHA:0:7}"
GIT="git --git-dir=/opt/ci/repos/haklolarp.git"
WT="/opt/ci/run/wt-${SHA}"
VENV=/opt/ci/.venv
NTFY_ENV=/opt/ci/ntfy.env
RC=0
GRC=0
hit() { [ "${1:-0}" -gt "$RC" ] && RC="$1"; return 0; }
ghit() { [ "${1:-0}" -gt "$GRC" ] && GRC="$1"; return 0; }
cleanup() {
  $GIT worktree remove --force "$WT" >/dev/null 2>&1 || true
  $GIT worktree prune >/dev/null 2>&1 || true
}
notify() {
  [ -f "$NTFY_ENV" ] || return 0
  # shellcheck disable=SC1091
  . "$NTFY_ENV" # single line: TOPIC=<name>
  [ -n "${TOPIC:-}" ] || return 0
  curl -s -m 10 -d "branch=$BRANCH sha=$SHA7 deploy=$RC gates=$GRC" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1 || true
}
trap 'notify; cleanup' EXIT
$GIT worktree add "$WT" "$SHA"
set +e
if [ ! -x "$VENV/bin/python" ]; then python3 -m venv "$VENV"; hit $?; fi
PY="$VENV/bin/python"
cd "$WT"; hit $?
# NOTE: pip runs AFTER cd — requirements-dev.txt resolves inside the worktree.
# Always (re)install: venv may predate a requirements change; pip is quick when satisfied.
"$PY" -m pip install -q -r requirements-dev.txt; ghit $?
"$PY" -m ruff check .; ghit $?
"$PY" -m yamllint .github/workflows compose.yaml compose.web.yaml; ghit $?
if [ -d tests ]; then "$PY" -m pytest -q -p no:cacheprovider; ghit $?; else echo "skip: no tests/"; fi
docker compose -f compose.yaml config -q; ghit $?
docker compose -f compose.web.yaml config -q; ghit $?
echo "gates-rc=$GRC (main lint debt tolerated; web deploy needs 0)"
cd /opt/haklolarp-staging; hit $?
git pull --ff-only; hit $?
docker compose up -d --remove-orphans --wait; hit $?
curl -fsS -m 10 http://127.0.0.1:8080/ >/dev/null; hit $?
if [ "$BRANCH" = web ] && [ "$GRC" -eq 0 ]; then
  cd /opt/haklolarp-web; hit $?
  git pull --ff-only origin web; hit $?
  docker compose -f compose.web.yaml -f compose.platform.yaml -f compose.production.yaml up -d --remove-orphans --wait; hit $?
  curl -fsS -m 10 http://127.0.0.1:8081/api/v1/health >/dev/null; hit $?
else
  echo "skip web deploy (branch=$BRANCH gates-rc=$GRC)"
fi
exit "$RC"
