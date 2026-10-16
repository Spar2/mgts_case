#!/usr/bin/env bash
# =============================================================================
# AGENT EXEC SANDBOX — run-subagent.sh
# -----------------------------------------------------------------------------
# Purpose : SOLE sanctioned entry point for running subagent commands inside a
#           branch exec cell. Wraps:
#             docker compose -f docker-compose.agent.yml \
#               exec --user agentuser sandbox-<branch>-exec <command>
# Allowlist (ENFORCED below): web | ml | data ONLY.
#   - `main`    -> REFUSED (REAL deploy target, staging-web-1 :8080).
#   - `preview` -> REFUSED (preview EXCLUDED entirely, no service exists).
#   - `devops`  -> REFUSED (control plane, not an exec target).
#   - Anything else -> usage error.
# COMPOSE_FILE pinning: the compose file is ALWAYS the sibling
#   docker-compose.agent.yml next to THIS script. A COMPOSE_FILE env override
#   is IGNORED by design — otherwise a caller could point the guards at an
#   unreviewed file (or a reviewed file at unguarded content). All guards run
#   against the pinned sibling path.
# Pre-exec config guards (fail-closed): before every exec this script greps the
#   compose file for escape hatches and aborts if ANY are found:
#   - /var/run/docker.sock mount, host-root mounts (`:/:`, `:/etc`, `:/root`,
#     ` /:/`, "/etc", "/root"), long-syntax `source:` (blocks
#     `source: /etc` bind tricks), env interpolation `$` (blocks
#     `- ${X:-/etc}:/...` indirection — NO $ expansion allowed at all).
#   - `network_mode:` anything except `none` (catches quoted/unquoted host,
#     service:, container:, bridge, True/1/empty — every non-none refuses).
#   - `privileged:` any truthy value (true/True/TRUE/1/yes/on quoted or not —
#     only an explicit false/0/no/off is tolerated; this file ships none).
#   - Never-allow keys: `cap_add:`, `devices:`, `sysctls:`, `extends:`,
#     `include:`, `env_file:`, `volumes_from:` (any indent, incl. top-level).
#   - Any `ports:` stanza (exec cells must have ZERO ingress).
#   - Missing hardening: `user:`, `cap_drop: ALL`, `no-new-privileges`,
#     `mem_limit`, `cpus:`, `pids_limit`, `read_only:`.
#   - UID/GID sync gate: compose `user:` must equal Dockerfile ARG UID/GID;
#     existing host workspace dirs must be owned by that id (else chown or
#     rebuild — see docker-compose header).
# Usage examples:
#   ./run-subagent.sh web  git -C /workspace status --short
#   ./run-subagent.sh ml   python3 -c "import sys; print(sys.version)"
#   ./run-subagent.sh data ls /workspace
#   ./run-subagent.sh web  bash            # interactive shell in web cell
#   ./run-subagent.sh --help               # this help
# Lifecycle (Xeon):
#   docker compose -f docker-compose.agent.yml up -d --build   # start cells
#   ./run-subagent.sh web <cmd>                                # exec (this)
#   docker compose -f docker-compose.agent.yml down            # stop cells
# Notes:
#   - Script resolves the compose file relative to ITS OWN directory, so it
#     works from any CWD as long as docker-compose.agent.yml sits next to it.
#   - Default command with no <command> given: `bash` (interactive shell).
#   - Always forces `--user agentuser` so even a root-invoked script cannot
#     land the agent as root inside the cell.
#   - `docker compose config` / runtime escape tests are DEFERRED to the Xeon
#     verifier (authoring host has no docker); the grep guards here are the
#     static backstop, not a substitute for `config` + live tests.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# Pinned sibling: COMPOSE_FILE env override IGNORED (see header).
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.agent.yml"
ALLOWED="web ml data"

usage() {
  cat <<'EOF'
Usage: run-subagent.sh <branch> [<command>...]
  branch: web | ml | data   (main/preview/devops refused)
Examples:
  ./run-subagent.sh web  git -C /workspace status --short
  ./run-subagent.sh ml   python3 -c "import sys; print(sys.version)"
  ./run-subagent.sh data ls /workspace
  ./run-subagent.sh web  bash
EOF
}

die() { echo "run-subagent: ERROR: $*" >&2; exit 1; }

# --help / -h
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi

[[ $# -ge 1 ]] || { usage >&2; die "missing <branch> (want: web|ml|data)"; }

BRANCH="$1"; shift || true

# --- allowlist + explicit refusals -------------------------------------------
case "$BRANCH" in
  web|ml|data) : ;;
  main)    die "refusing branch 'main' — REAL deploy target (staging-web-1), no sandbox cell exists by design" ;;
  preview*) die "refusing branch '$BRANCH' — preview is EXCLUDED entirely, no service exists by design" ;;
  devops)  die "refusing branch 'devops' — control plane, not an exec target" ;;
  *)       usage >&2; die "unknown branch '$BRANCH' (allowlist: $ALLOWED)" ;;
esac

SERVICE="sandbox-${BRANCH}-exec"

[[ -f "$COMPOSE_FILE" ]] || die "compose file not found: $COMPOSE_FILE"

# --- fail-closed config guards (static grep backstop) -------------------------
# NOTE: all greps below run against the compose file with `#` comment lines
# stripped (grep -vE '^[[:space:]]*#'), so this script's own documentation
# comments that NAME the forbidden strings cannot false-positive. Inline
# trailing comments (value # comment) are tolerated by the allow-patterns.
# 1) service must exist in the compose file
grep -qE "^[[:space:]]+${SERVICE}:" "$COMPOSE_FILE" \
  || die "service '$SERVICE' not found in $COMPOSE_FILE (refusing exec)"

# Non-comment config body used for every guard below.
CFG_NC="$(grep -vE '^[[:space:]]*#' "$COMPOSE_FILE")"

# 2a) network_mode: ONLY 'none' allowed (quoted/unquoted). Any other value
#     (host, service:, container:, bridge, True, 1, empty) refuses.
if echo "$CFG_NC" | grep -Eq '^[[:space:]]*network_mode[[:space:]]*:'; then
  if echo "$CFG_NC" | grep -E '^[[:space:]]*network_mode[[:space:]]*:' \
    | grep -Ev '^[[:space:]]*network_mode[[:space:]]*:[[:space:]]*["'\'']?none["'\'']?[[:space:]]*(#.*)?$' | grep -q .; then
    die "forbidden network_mode (only 'none' allowed) in $COMPOSE_FILE (refusing exec)"
  fi
fi
# 2b) privileged: only explicit false-variants tolerated; any truthy/missing
#     value (true/True/TRUE/1/yes/on quoted or not) refuses.
if echo "$CFG_NC" | grep -Eq '^[[:space:]]*privileged[[:space:]]*:'; then
  if echo "$CFG_NC" | grep -E '^[[:space:]]*privileged[[:space:]]*:' \
    | grep -Ev '^[[:space:]]*privileged[[:space:]]*:[[:space:]]*["'\'']?(false|False|FALSE|0|no|No|NO|off|Off|OFF)["'\'']?[[:space:]]*(#.*)?$' | grep -q .; then
    die "forbidden 'privileged:' truthy value in $COMPOSE_FILE (refusing exec)"
  fi
fi
# 2c) never-allow keys: long-syntax escape hatches + privilege escalation.
for pat in '^[[:space:]]*cap_add[[:space:]]*:' '^[[:space:]]*devices[[:space:]]*:' '^[[:space:]]*sysctls[[:space:]]*:' '^[[:space:]]*extends[[:space:]]*:' '^[[:space:]]*include[[:space:]]*:' '^[[:space:]]*env_file[[:space:]]*:' '^[[:space:]]*volumes_from[[:space:]]*:' '^[[:space:]]*source[[:space:]]*:'; do
  if echo "$CFG_NC" | grep -Eq "$pat"; then
    die "forbidden pattern '$pat' found in $COMPOSE_FILE (refusing exec)"
  fi
done
# 2d) docker.sock anywhere refuses.
if echo "$CFG_NC" | grep -Eq 'docker\.sock'; then
  die "forbidden pattern 'docker.sock' found in $COMPOSE_FILE (refusing exec)"
fi
# 2e) env indirection: ANY '$' in non-comment config refuses (blocks
#     '- ${X:-/etc}:/...', '$HOST:/...', '$$', etc. — no expansion allowed).
if echo "$CFG_NC" | grep -qF '$'; then
  die "forbidden env interpolation '\$' in $COMPOSE_FILE (refusing exec — no \$ expansion allowed)"
fi
# host-root mounts: crude but effective line match for source-side /,/etc,/root
if echo "$CFG_NC" | grep -Eq '^[[:space:]]*-[[:space:]]*(/var/run/docker\.sock|/:[^:]|/:$|/etc|/root)[/:"]'; then
  die "forbidden host mount (docker.sock / /etc /root) in $COMPOSE_FILE (refusing exec)"
fi
if echo "$CFG_NC" | grep -Eq '^[[:space:]]*-[[:space:]]*"/(etc|root)|^[[:space:]]*-[[:space:]]*/(etc|root)/'; then
  die "forbidden host mount (/etc or /root) in $COMPOSE_FILE (refusing exec)"
fi
# 3) zero ingress: no ports: stanza allowed in exec compose
if echo "$CFG_NC" | grep -Eq '^[[:space:]]+ports:'; then
  die "'ports:' found in $COMPOSE_FILE — exec cells must publish NO ports (refusing exec)"
fi
# 4) hardening must be present
for pat in 'user:[[:space:]]*"10000:10000"' 'cap_drop:' 'no-new-privileges' 'mem_limit:' 'cpus:' 'pids_limit:' 'read_only:'; do
  echo "$CFG_NC" | grep -Eq "$pat" \
    || die "required hardening '$pat' missing in $COMPOSE_FILE (refusing exec)"
done
# 5) forbidden services must NOT exist (defense in depth with validator greps)
for bad in 'sandbox-main-exec' 'sandbox-preview-exec' 'sandbox-devops-exec'; do
  if echo "$CFG_NC" | grep -Eq "^[[:space:]]+.*${bad}:"; then
    die "forbidden service '${bad}' present in $COMPOSE_FILE (refusing exec)"
  fi
done
# 6) UID/GID sync gate: compose user: must equal Dockerfile ARG UID/GID.
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.agent-sandbox"
if [[ -f "$DOCKERFILE" ]]; then
  DOCK_NC="$(grep -vE '^[[:space:]]*#' "$DOCKERFILE")"
  UID_DEF="$(echo "$DOCK_NC" | grep -E '^[[:space:]]*ARG UID=' | sed -E 's/.*ARG UID=([0-9]+).*/\1/' | head -n 1 || true)"
  GID_DEF="$(echo "$DOCK_NC" | grep -E '^[[:space:]]*ARG GID=' | sed -E 's/.*ARG GID=([0-9]+).*/\1/' | head -n 1 || true)"
  if [[ -n "${UID_DEF:-}" && -n "${GID_DEF:-}" ]]; then
    echo "$CFG_NC" | grep -Eq "user:[[:space:]]*\"${UID_DEF}:${GID_DEF}\"" \
      || die "UID/GID sync gate: compose user: must be \"${UID_DEF}:${GID_DEF}\" to match Dockerfile ARG UID/GID (rebuild with --build-arg UID/GID to sync, keep both files in sync)"
  fi
  # Host workspace ownership check (skipped when path absent, e.g. authoring
  # host; enforced on Xeon where /opt/haklolarp-* exists).
  for hostdir in /opt/haklolarp-web /opt/haklolarp-ml /opt/haklolarp-data; do
    if [[ -e "$hostdir" ]]; then
      owner="$(stat -c '%u:%g' "$hostdir" 2>/dev/null || true)"
      if [[ -n "${owner:-}" && -n "${UID_DEF:-}" && -n "${GID_DEF:-}" && "$owner" != "${UID_DEF}:${GID_DEF}" ]]; then
        die "workspace $hostdir owned by $owner, want ${UID_DEF}:${GID_DEF} — run: chown -R ${UID_DEF}:${GID_DEF} $hostdir OR rebuild with --build-arg UID=$(id -u) --build-arg GID=$(id -g) and sync compose user: (refusing exec)"
      fi
    fi
  done
fi

# --- exec ----------------------------------------------------------------------
# Default: interactive bash when no command given.
if [[ $# -eq 0 ]]; then set -- bash; fi

exec docker compose -f "$COMPOSE_FILE" exec --user agentuser "$SERVICE" "$@"
