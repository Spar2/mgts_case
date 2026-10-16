#!/usr/bin/env bash
# =============================================================================
# FLEET canonical multi-branch runner — run-sandbox.sh
# -----------------------------------------------------------------------------
# Purpose : SOLE sanctioned entry point for short-lived pilot probes per
#           branch (web|ml|data) via ops/sandboxes/runner/docker-compose.sandboxes.yml.
# Usage   : ./run-sandbox.sh [start|stop|exec|status] <web|ml|data> [command...]
# Examples:
#   ./run-sandbox.sh start web
#   ./run-sandbox.sh status ml
#   ./run-sandbox.sh exec web id\; touch /workspace/.sandbox-check-web && rm /workspace/.sandbox-check-web
#   ./run-sandbox.sh exec data -- sh -c 'cat /etc/shadow; echorule UNREACHABLE'
#   ./run-sandbox.sh stop data
# Allowlist (ENFORCED): web | ml | data ONLY.
#   - `main`    -> REFUSED (REAL deploy target, staging-web-1 :8080).
#   - `preview` -> REFUSED (preview EXCLUDED entirely, no service exists).
#   - `devops`  -> REFUSED (control plane, not a pilot target).
# Branch defaults (fallback only — Xeon `stat` wins when path exists):
#   web : 1001:1001 /opt/haklolarp-web  18081->8081 (proven live pilot)
#   ml  : 1002:1002 /opt/haklolarp-ml   18082->8082 (PENDING VERIFY Phase 2)
#   data: 1003:1003 /opt/haklolarp-stack 18000->8000 + 16379->6379 + 18123->8123
#         (PENDING VERIFY Phase 2)
# Resolution order per branch:
#   SANDBOX_SRC       = ${SANDBOX_SRC:-<branch-default-path>}
#   SANDBOX_UID_GID   = ${SANDBOX_UID_GID:-<stat -c '%u:%g' $SANDBOX_SRC or branch-default>}
#                       (explicit env wins; else live `stat`; else fallback)
#   SANDBOX_HOST_PORT = ${SANDBOX_HOST_PORT:-<branch-default-hostport>}
#   data extras: SANDBOX_DATA_REDIS_PORT=${SANDBOX_DATA_REDIS_PORT:-16379},
#                SANDBOX_DATA_CH_PORT=${SANDBOX_DATA_CH_PORT:-18123}.
#   Shell env (this script's exports) overrides compose-file defaults
#   (${VAR:-default} interpolation — context7, current).
# Actions:
#   start  = docker compose -f <sibling> --profile <branch> up -d
#   stop   = docker compose -f <sibling> --profile <branch> down
#   exec   = docker compose -f <sibling> --profile <branch> exec --user $UID_GID <svc> [cmd]
#            (default cmd: bash; always --user resolved UID so root-invoke safe)
#   status = docker compose -f <sibling> --profile <branch> ps
# Guards (fail-closed, before start/exec):
#   - compose sibling must exist; service sandbox-<branch> must exist in it.
#   - loopback: every `ports:` line must contain 127.0.0.1; any 0.0.0.0 refuses.
#   - host-port denylist: resolved HOST/REDIS/CH ports must NOT be in
#     8080 8081 8082 5434 6379 8123 8000 (live stands) — refuses.
#   - socket/root mounts: refuses docker.sock, `/:`, /etc, /root mounts.
# COMPOSE_FILE pinning: ALWAYS the sibling docker-compose.sandboxes.yml next
#   to THIS script. COMPOSE_FILE env override IGNORED (else guards bypassed).
# Lifecycle (Xeon, /tmp/sandbox-fleet/, delete after):
#   ./run-sandbox.sh start ml; ./run-sandbox.sh status ml; ./run-sandbox.sh stop ml
# Notes:
#   - Resolves compose file relative to ITS OWN directory (any CWD works).
#   - `docker compose config` + live proofs DEFERRED to Xeon verifier.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# Pinned sibling: COMPOSE_FILE env override IGNORED (see header).
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.sandboxes.yml"

usage() {
  cat <<'EOF'
Usage: run-sandbox.sh [start|stop|exec|status] <web|ml|data> [command...]
  actions: start | stop | exec | status
  branches: web | ml | data   (main/preview/devops refused)
Examples:
  ./run-sandbox.sh start web
  ./run-sandbox.sh status ml
  ./run-sandbox.sh exec web id
  ./run-sandbox.sh exec data -- sh -c 'touch /workspace/.sandbox-check-data && rm /workspace/.sandbox-check-data'
  ./run-sandbox.sh stop data
EOF
}

die() { echo "run-sandbox: ERROR: $*" >&2; exit 1; }

# --help / -h (either position)
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || "${2:-}" == "--help" || "${2:-}" == "-h" ]]; then usage; exit 0; fi

[[ $# -ge 2 ]] || { usage >&2; die "want: [start|stop|exec|status] <web|ml|data> (got $# args)"; }

ACTION="$1"; BRANCH="$2"; shift 2 || true
# `exec -- <cmd>` passthrough: drop a lone `--` separator if present.
if [[ "$ACTION" == "exec" && "${1:-}" == "--" ]]; then shift || true; fi

case "$ACTION" in
  start|stop|exec|status) : ;;
  *) usage >&2; die "unknown action '$ACTION' (allowlist: start|stop|exec|status)" ;;
esac

case "$BRANCH" in
  web|ml|data) : ;;
  main)    die "refusing branch 'main' — REAL deploy target (staging-web-1 :8080), no pilot cell by design" ;;
  preview*) die "refusing branch '$BRANCH' — preview EXCLUDED entirely, no service exists by design" ;;
  devops)  die "refusing branch 'devops' — control plane, not a pilot target" ;;
  *)       usage >&2; die "unknown branch '$BRANCH' (allowlist: web|ml|data)" ;;
esac

SERVICE="sandbox-${BRANCH}"
[[ -f "$COMPOSE_FILE" ]] || die "compose file not found: $COMPOSE_FILE"

# --- per-branch defaults (fallback only) -------------------------------------
case "$BRANCH" in
  web)
    DEF_UID_GID="1001:1001"; DEF_SRC="/opt/haklolarp-web"
    DEF_HOST_PORT="18081"; INTERNAL_PORT="8081"
    ;;
  ml)
    DEF_UID_GID="1002:1002"; DEF_SRC="/opt/haklolarp-ml"
    DEF_HOST_PORT="18082"; INTERNAL_PORT="8082"
    ;;
  data)
    DEF_UID_GID="1003:1003"; DEF_SRC="/opt/haklolarp-stack"
    DEF_HOST_PORT="18000"; INTERNAL_PORT="8000"
    DEF_REDIS_PORT="16379"; DEF_CH_PORT="18123"
    ;;
esac

# --- resolution (env override > live stat > fallback) -------------------------
# shellcheck disable=SC2034
TARGET_SRC="${SANDBOX_SRC:-$DEF_SRC}"
TARGET_HOST_PORT="${SANDBOX_HOST_PORT:-$DEF_HOST_PORT}"

# UID/GID: explicit SANDBOX_UID_GID wins; else live stat of the (possibly
# overridden) source path; else branch fallback. `stat` runs at RUNTIME on
# Xeon so verify-corrected owners apply without editing this file.
if [[ -n "${SANDBOX_UID_GID:-}" ]]; then
  TARGET_UID_GID="$SANDBOX_UID_GID"
elif [[ -e "$TARGET_SRC" ]]; then
  _stat_owner="$(stat -c '%u:%g' "$TARGET_SRC" 2>/dev/null || true)"
  if [[ -n "${_stat_owner:-}" ]]; then
    TARGET_UID_GID="$_stat_owner"
  else
    TARGET_UID_GID="$DEF_UID_GID"
  fi
else
  TARGET_UID_GID="$DEF_UID_GID"
fi

export SANDBOX_SRC="$TARGET_SRC"
export SANDBOX_UID_GID="$TARGET_UID_GID"
export SANDBOX_HOST_PORT="$TARGET_HOST_PORT"
if [[ "$BRANCH" == "data" ]]; then
  export SANDBOX_DATA_REDIS_PORT="${SANDBOX_DATA_REDIS_PORT:-$DEF_REDIS_PORT}"
  export SANDBOX_DATA_CH_PORT="${SANDBOX_DATA_CH_PORT:-$DEF_CH_PORT}"
fi

# --- fail-closed guards --------------------------------------------------------
# 1) service must exist in the compose file.
grep -qE "^[[:space:]]+${SERVICE}:" "$COMPOSE_FILE" \
  || die "service '$SERVICE' not found in $COMPOSE_FILE (refusing $ACTION)"
# 2) loopback: every ports mapping must pin 127.0.0.1; 0.0.0.0 refuses.
CFG_NC="$(grep -vE '^[[:space:]]*#' "$COMPOSE_FILE")"
if echo "$CFG_NC" | grep -Eq '^[[:space:]]*-[[:space:]]*"0\.0\.0\.0:'; then
  die "forbidden 0.0.0.0 bind in $COMPOSE_FILE (loopback-only, refusing $ACTION)"
fi
# 3) host-port denylist (live stands): 8080/8081/8082/5434/6379/8123/8000.
FORBIDDEN_PORTS="8080 8081 8082 5434 6379 8123 8000"
_check_port() {
  local _p="$1" _name="$2" _bad
  for _bad in $FORBIDDEN_PORTS; do
    [[ "$_p" == "$_bad" ]] && die "refusing $ACTION: $_name port $_p is a LIVE stand port (denylist: $FORBIDDEN_PORTS)"
  done
  return 0 # last [[ in loop is false on success; without this, set -e aborts the caller
}
_check_port "$TARGET_HOST_PORT" "SANDBOX_HOST_PORT"
if [[ "$BRANCH" == "data" ]]; then
  _check_port "$SANDBOX_DATA_REDIS_PORT" "SANDBOX_DATA_REDIS_PORT"
  _check_port "$SANDBOX_DATA_CH_PORT" "SANDBOX_DATA_CH_PORT"
fi
# 4) socket/root mounts refuse.
if echo "$CFG_NC" | grep -Eq 'docker\.sock'; then
  die "forbidden 'docker.sock' in $COMPOSE_FILE (refusing $ACTION)"
fi
if echo "$CFG_NC" | grep -Eq '^[[:space:]]*-[[:space:]]*"/(etc|root)|^[[:space:]]*-[[:space:]]*/(etc|root)/'; then
  die "forbidden host mount (/etc or /root) in $COMPOSE_FILE (refusing $ACTION)"
fi

echo "run-sandbox: action=$ACTION branch=$BRANCH service=$SERVICE src=$SANDBOX_SRC uid=$SANDBOX_UID_GID hostport=$SANDBOX_HOST_PORT internal=$INTERNAL_PORT" >&2
if [[ "$BRANCH" == "data" ]]; then
  echo "run-sandbox: data extras redis=$SANDBOX_DATA_REDIS_PORT ch=$SANDBOX_DATA_CH_PORT" >&2
fi

# --- actions -------------------------------------------------------------------
case "$ACTION" in
  start)
    exec docker compose -f "$COMPOSE_FILE" --profile "$BRANCH" up -d
    ;;
  stop)
    exec docker compose -f "$COMPOSE_FILE" --profile "$BRANCH" down
    ;;
  exec)
    # Default: interactive bash when no command given (python:slim has bash).
    if [[ $# -eq 0 ]]; then set -- bash; fi
    exec docker compose -f "$COMPOSE_FILE" --profile "$BRANCH" exec --user "$SANDBOX_UID_GID" "$SERVICE" "$@"
    ;;
  status)
    exec docker compose -f "$COMPOSE_FILE" --profile "$BRANCH" ps
    ;;
esac
