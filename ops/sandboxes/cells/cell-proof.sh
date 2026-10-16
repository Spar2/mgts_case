#!/usr/bin/env bash
# ops/sandboxes/cells/cell-proof.sh — READ-ONLY proof for branch workload cells.
# To be run ON XEON by the verifier. This script NEVER starts/stops/modifies
# anything: no `compose up/down`, no `docker rm/restart`, no writes, no pushes.
# Allowed operations ONLY: `ss`, `docker ps/inspect/logs`, read-only `docker exec`
# (health reads + negative visibility checks that are ORDERED TO FAIL), and
# read-only grep over the compose file itself (§S static asserts, no docker needed).
#
# Usage: bash ops/sandboxes/cells/cell-proof.sh [compose-project-name]
#   Optional arg (or $CELLS_PROJECT) narrows container discovery when several
#   compose projects exist on the host. Without it, service-label discovery is used.
#
# Exit: 0 = no FAIL and PASS >= MIN_PASS; 1 = at least one FAIL, or a hollow
#   all-SKIP run (PASS below MIN_PASS). The MIN_PASS floor exists so that a run
#   against an empty host (cells not running) can NEVER exit 0 on SKIP alone.
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.cells.yml"
# 15 static passes (§S) always run; MIN_PASS=20 additionally requires a handful
# of LIVE passes, so SKIP-only runs fail. Bump only with justification.
MIN_PASS=20
PROJECT="${1:-${CELLS_PROJECT:-}}"
PASS=0; FAIL=0; SKIP=0; WARN=0

pass() { PASS=$((PASS+1)); echo "PASS: $*"; }
fail() { FAIL=$((FAIL+1)); echo "FAIL: $*"; }
skip() { SKIP=$((SKIP+1)); echo "SKIP: $*"; }
warn() { WARN=$((WARN+1)); echo "WARN: $*"; }

need() { command -v "$1" >/dev/null 2>&1 || { echo "FATAL: missing required tool: $1"; exit 1; }; }
need docker; need ss

# Find the (single) RUNNING container id for a compose service, optionally scoped.
find_cid() {
  local svc="$1" ids
  if [ -n "$PROJECT" ]; then
    ids=$(docker ps -q --filter "label=com.docker.compose.project=$PROJECT" --filter "label=com.docker.compose.service=$svc" 2>/dev/null)
  else
    ids=$(docker ps -q --filter "label=com.docker.compose.service=$svc" 2>/dev/null)
  fi
  local n
  n=$(printf '%s' "$ids" | grep -c . || true)
  if [ "$n" -eq 0 ]; then return 1; fi
  if [ "$n" -gt 1 ]; then warn "service $svc matches $n containers; checking the first" >&2; fi
  printf '%s' "$ids" | head -n 1
  return 0
}

# Find ANY (running or exited) container id for a service — migrate receipt.
find_cid_any() {
  local svc="$1" ids
  if [ -n "$PROJECT" ]; then
    ids=$(docker ps -a -q --filter "label=com.docker.compose.project=$PROJECT" --filter "label=com.docker.compose.service=$svc" 2>/dev/null)
  else
    ids=$(docker ps -a -q --filter "label=com.docker.compose.service=$svc" 2>/dev/null)
  fi
  local n
  n=$(printf '%s' "$ids" | grep -c . || true)
  if [ "$n" -eq 0 ]; then return 1; fi
  printf '%s' "$ids" | head -n 1
  return 0
}

echo "=== S. static YAML asserts (no docker needed; comments stripped) ==="
if [ ! -f "$COMPOSE_FILE" ]; then
  fail "compose file missing: $COMPOSE_FILE"
else
  # Strip full-line and trailing comments so documentation mentions of banned
  # patterns (e.g. the §B1 rationale) cannot trip the asserts; bans apply to
  # EFFECTIVE YAML only. (Naive strip: '#' inside quoted strings would also be
  # cut — none of the asserted patterns rely on such strings; ports/user/image
  # values contain no '#'.)
  STRIPPED="$(sed -e 's/#.*$//' "$COMPOSE_FILE")"
  # S1: no container_name anywhere (project-safe `-p <name>` discipline).
  if printf '%s\n' "$STRIPPED" | grep -qE '^[[:space:]]*container_name:'; then
    fail "S1 static: container_name present (breaks -p <project> isolation)"
  else
    pass "S1 static: no container_name"
  fi
  # S2: no 0.0.0.0 in effective YAML (all published ports loopback-only).
  if printf '%s\n' "$STRIPPED" | grep -qF '0.0.0.0'; then
    fail "S2 static: 0.0.0.0 present (some port/matrix binds beyond loopback)"
  else
    pass "S2 static: no 0.0.0.0"
  fi
  # S3: no host-gateway round-trip (circular dependency, see compose §B1).
  if printf '%s\n' "$STRIPPED" | grep -qE 'host-gateway|host\.docker\.internal'; then
    fail "S3 static: host-gateway/host.docker.internal present (reintroduces §B1 circle)"
  else
    pass "S3 static: no host-gateway/host.docker.internal"
  fi
  # S4: every published port line binds 127.0.0.1 (shape: "- "127.0.0.1:…:…"",
  # host part may be an env template — only the literal 127.0.0.1 prefix counts).
  n_ports=$(printf '%s\n' "$STRIPPED" | grep -cE '^[[:space:]]*-[[:space:]]*"[^"]*:[^"]*"' || true)
  n_loopback=$(printf '%s\n' "$STRIPPED" | grep -cE '^[[:space:]]*-[[:space:]]*"127\.0\.0\.1:' || true)
  if [ "$n_ports" -eq 0 ]; then
    fail "S4 static: no published-port lines found (parser drift?)"
  elif [ "$n_ports" = "$n_loopback" ]; then
    pass "S4 static: all $n_ports published ports bind 127.0.0.1"
  else
    fail "S4 static: $n_loopback/$n_ports port lines loopback-bound (want all)"
  fi
  # S5: cells-back is internal:true (east-west only).
  if printf '%s\n' "$STRIPPED" | grep -qE 'internal:[[:space:]]*true'; then
    pass "S5 static: cells-back internal:true present"
  else
    fail "S5 static: no internal:true (cells-back must be east-west only)"
  fi
  # S6: no main/preview markers in effective YAML (no staging-web-1, no :8080
  # host port, no 9000+ preview range, no preview service).
  if printf '%s\n' "$STRIPPED" | grep -qE 'staging-web-1|preview|:8080([^0-9]|$)|:900[0-9]|:"900'; then
    fail "S6 static: main/preview marker leaked into cells file"
  else
    pass "S6 static: no main/preview markers"
  fi
  # S7-S9: per-service confinement coverage — 8 services (incl. migrate), each
  # must carry cap_drop ALL, no-new-privileges, read_only:true (no yaml
  # anchors by design, so counts must match exactly). Service headers are
  # counted only ABOVE the top-level `networks:` line: the `volumes:` block
  # reuses cell-* names (cell-data-redis, cell-web-records, ...) and a naive
  # whole-file grep would count those too (14 instead of 8).
  n_services=$(printf '%s\n' "$STRIPPED" | awk '/^networks:/{stop=1} !stop' | grep -cE '^  cell-[a-z-]+:' || true)
  n_capdrop=$(printf '%s\n' "$STRIPPED" | grep -cE '^[[:space:]]+cap_drop:' || true)
  n_all=$(printf '%s\n' "$STRIPPED" | grep -cE '^[[:space:]]+-[[:space:]]+ALL' || true)
  n_nonpriv=$(printf '%s\n' "$STRIPPED" | grep -c 'no-new-privileges:true' || true)
  n_ronly=$(printf '%s\n' "$STRIPPED" | grep -cE '^[[:space:]]+read_only:[[:space:]]+true' || true)
  [ "$n_services" -eq 8 ] \
    && pass "S7a static: 8 cell services (incl. cell-data-migrate)" \
    || fail "S7a static: $n_services cell services (want 8)"
  { [ "$n_capdrop" -eq 8 ] && [ "$n_all" -eq 8 ]; } \
    && pass "S7b static: cap_drop ALL on all 8 services" \
    || fail "S7b static: cap_drop blocks=$n_capdrop ALL=$n_all (want 8/8)"
  [ "$n_nonpriv" -eq 8 ] \
    && pass "S8 static: no-new-privileges on all 8 services" \
    || fail "S8 static: no-new-privileges on $n_nonpriv/8 services"
  [ "$n_ronly" -eq 8 ] \
    && pass "S9 static: read_only:true on all 8 services" \
    || fail "S9 static: read_only:true on $n_ronly/8 services"
  # S10: expected non-root users present (101 nginx, 10001 web-api, 65532 ml/data).
  for u in '"101"' '"10001"' '"65532"'; do
    if printf '%s\n' "$STRIPPED" | grep -qE "^[[:space:]]*user:[[:space:]]*$u"; then
      pass "S10 static: user: $u present"
    else
      fail "S10 static: user: $u missing"
    fi
  done
  # S11: migrate gate present — profile-gated one-shot, restart "no".
  if printf '%s\n' "$STRIPPED" | grep -qE '^  cell-data-migrate:' \
     && printf '%s\n' "$STRIPPED" | grep -qE '"migrate"' \
     && printf '%s\n' "$STRIPPED" | grep -qE 'restart:[[:space:]]*"no"'; then
    pass "S11 static: cell-data-migrate (profiles:[migrate], restart:no) present"
  else
    fail "S11 static: migrate gate missing (need cell-data-migrate + profile + restart:no)"
  fi
  # S12: CELL_*_SRC defaults are smuggling-safe — each ${CELL_...:-default}
  # default must be repo-relative (../..), a ./ path, or a bare named volume
  # (no leading /). Absolute host paths must come from env, never baked in.
  bad_defaults=$(printf '%s\n' "$STRIPPED" | grep -oE '\$\{CELL_[A-Z_]+:-[^}]*\}' | grep -E ':-/' || true)
  if [ -z "$bad_defaults" ]; then
    pass "S12 static: no absolute CELL_*_SRC default baked in (env-only)"
  else
    fail "S12 static: absolute CELL_*_SRC default baked in:$bad_defaults"
  fi
fi

echo "=== A. main exclusion guard (staging-web-1 must keep running) ==="
# Name drift: the compose project prefixes the container, so the live name is
# `haklolarp-staging-web-1`, not bare `staging-web-1`. Match by substring via
# `docker ps --filter name=...` (covers both), with exact-inspect fallback for
# stopped containers. Without this the guard SKIPped on main hosts (false SKIP).
MAIN_NAME="$(docker ps --filter "name=staging-web-1" --format '{{.Names}}' 2>/dev/null | head -n 1 || true)"
if [ -z "$MAIN_NAME" ]; then
  if docker inspect staging-web-1 >/dev/null 2>&1; then
    MAIN_NAME="staging-web-1"
  elif docker inspect haklolarp-staging-web-1 >/dev/null 2>&1; then
    MAIN_NAME="haklolarp-staging-web-1"
  fi
fi
if [ -n "$MAIN_NAME" ]; then
  if [ "$(docker inspect --format '{{.State.Status}}' "$MAIN_NAME" 2>/dev/null)" = "running" ]; then
    pass "main container $MAIN_NAME is running (untouched)"
  else
    fail "$MAIN_NAME exists but is NOT running — main may have been disturbed"
  fi
else
  skip "staging-web-1 absent on this host (not a main host?) — nothing to guard"
fi

echo "=== B. socket checks (ss -ltnp): ALL cells MUST be 127.0.0.1-only ==="
# args: label port
check_port() {
  local label="$1" port="$2" rows local_addrs holder
  rows=$(ss -ltnp "sport = :$port" 2>/dev/null | tail -n +2 || true)
  if [ -z "$rows" ]; then skip "$label: nothing listening on :$port (cell not running?)"; return; fi
  # Parse the LOCAL address field ONLY (awk $4): the PEER column of a listener
  # is always 0.0.0.0:* / [::]:*, so grepping the full line false-positives on
  # every loopback listener. FAIL only on 0.0.0.0:PORT or [::]:PORT in $4.
  # docker on dual-stack hosts may publish [::]:port alongside 127.0.0.1 — that
  # IS an exposure and FAILs here (binds must be 127.0.0.1 exactly).
  local_addrs=$(printf '%s\n' "$rows" | awk '{print $4}')
  if printf '%s\n' "$local_addrs" | grep -qE "0\.0\.0\.0:$port([[:space:]]|$)|\\[::\\]:$port([[:space:]]|$)"; then
    holder=$(printf '%s\n' "$rows" | grep -oE 'users:\(.*\)' | head -n 1 || true)
    [ -z "$holder" ] && holder="occupied by non-cell listener"
    fail "$label :$port exposed beyond loopback (local address: $(printf '%s' "$local_addrs" | tr '\n' ' '); $holder):$rows"
  elif printf '%s\n' "$local_addrs" | grep -qE "127\.0\.0\.1:$port([[:space:]]|$)"; then
    pass "$label :$port loopback-only"
  else
    holder=$(printf '%s\n' "$rows" | grep -oE 'users:\(.*\)' | head -n 1 || true)
    [ -z "$holder" ] && holder="occupied by non-cell listener"
    fail "$label :$port unexpected bind (local address: $(printf '%s' "$local_addrs" | tr '\n' ' '); $holder):$rows"
  fi
}
check_port "cell-web " "${CELL_WEB_PORT:-8081}"
check_port "cell-ml   " "${CELL_ML_PORT:-8082}"
check_port "cell-redis" "${CELL_REDIS_PORT:-6379}"
check_port "cell-ch   " "${CELL_CH_HTTP_PORT:-8123}"
check_port "cell-data " "${CELL_DATA_API_PORT:-8000}"

echo "=== C. confinement asserts (docker inspect: Mounts AND Binds) ==="
# CELL_*_SRC allowlist: only these Sources may back a bind. Extend by editing
# THIS script (commit), never via env — env is the smuggling vector.
# Allowed: bare named volumes (no /), repo-relative checkouts under the known
# branch roots, docker's own volume dir (Mounts Source rendering of (a)).
allowlisted() {
  local src="$1"
  case "$src" in
    *staging*|*preview*) return 1;; # neighbor/main checkouts are NEVER valid sources
  esac
  case "$src" in
    /|/etc|/etc/*|/root|/root/*|/proc|/proc/*|/sys|/sys/*|/var/run/*|*docker.sock) return 1;;
  esac
  case "$src" in
    */*) ;; # has a slash: must match an allowed root below
    *) return 0;; # bare named volume (cell-web-records, ...)
  esac
  case "$src" in
    /opt/haklolarp-*|/opt/cells-*|/srv/haklolarp-*|/srv/cells-*|/home/*|/var/lib/docker/volumes/*) return 0;;
    *) return 1;;
  esac
}
# args: service expected-user(empty means documented root-exclusion)
check_confine() {
  local svc="$1" exp_user="$2" cid
  if ! cid=$(find_cid "$svc"); then skip "$svc not running — confinement asserts deferred"; return; fi
  local user capdrop ronp sec mounts binds src bad
  user=$(docker inspect --format '{{.Config.User}}' "$cid" 2>/dev/null)
  capdrop=$(docker inspect --format '{{.HostConfig.CapDrop}}' "$cid" 2>/dev/null)
  ronp=$(docker inspect --format '{{.HostConfig.ReadonlyRootfs}}' "$cid" 2>/dev/null)
  sec=$(docker inspect --format '{{.HostConfig.SecurityOpt}}' "$cid" 2>/dev/null)
  if [ -n "$exp_user" ]; then
    [ "$user" = "$exp_user" ] && pass "$svc User=$user" || fail "$svc User='$user', want '$exp_user'"
  else
    [ -z "$user" ] && warn "$svc runs as root (DOCUMENTED exclusion: redis/clickhouse entrypoints)" \
                   || fail "$svc User='$user', want root-exclusion(empty)+README entry"
  fi
  case "$capdrop" in *ALL*) pass "$svc CapDrop=$capdrop";; *) fail "$svc CapDrop='$capdrop', want ALL";; esac
  [ "$ronp" = "true" ] && pass "$svc ReadonlyRootfs=true" || fail "$svc ReadonlyRootfs='$ronp', want true"
  case "$sec" in *no-new-privileges*) pass "$svc SecurityOpt has no-new-privileges";; *) fail "$svc SecurityOpt='$sec'";; esac
  # Forbidden mounts: inspect BOTH Mounts (resolved) and Binds (raw spec) —
  # a smuggled bind hidden behind variable indirection shows in at least one.
  mounts=$(docker inspect --format '{{range .Mounts}}{{println .Source .Destination}}{{end}}' "$cid" 2>/dev/null)
  binds=$(docker inspect --format '{{range .HostConfig.Binds}}{{println .}}{{end}}' "$cid" 2>/dev/null)
  if printf '%s\n' "$mounts" "$binds" | grep -qE '(^|[[:space:]])/(etc|root)([[:space:]/:]|$)|docker\.sock|(^|[[:space:]])/([[:space:]:]|$)'; then
    fail "$svc forbidden mount present (Mounts:$mounts Binds:$binds)"
  else
    pass "$svc mounts clean (no / /etc /root docker.sock in Mounts+Binds)"
  fi
  # Allowlist every bind/bind-source: non-allowlisted Source = env-mount smuggling.
  bad=""
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    src="${line%%:*}"; src="${src%% *}"
    if ! allowlisted "$src"; then bad="$bad [$src]"; fi
  done <<EOF
$mounts
$binds
EOF
  # Also check Mounts Sources column explicitly (first field of each Mounts line).
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    src="${line%% *}"
    case "$src" in
      /var/lib/docker/volumes/*) continue;; # named-volume rendering, allowed
    esac
    if ! allowlisted "$src"; then
      case "$bad" in *"[$src]"*) ;; *) bad="$bad [$src]";; esac
    fi
  done <<EOF
$mounts
EOF
  if [ -n "$bad" ]; then
    fail "$svc non-allowlisted mount Source:$bad (see SOURCE-ROOT ALLOWLIST)"
  else
    pass "$svc mount Sources allowlisted"
  fi
}
check_confine cell-web "101"
check_confine cell-web-api "10001"
check_confine cell-ml "65532"
check_confine cell-data-redis ""
check_confine cell-data-clickhouse ""
check_confine cell-data-backend "65532"
check_confine cell-data-ingester "65532"

echo "=== D. neighbor-visibility negative tests (ORDERED TO FAIL) ==="
# Proof idea: host/neighbor paths must be unreachable from inside a cell.
# Each `exec` MUST fail; success = leak = FAIL. The /proc/1/root probe is the
# strong one: it resolves through the HOST pid-1 root symlink, so it catches
# breakout-class misconfigurations (privileged/hostPid/host-mount leaks), not
# just absent bind mounts.
if cid=$(find_cid cell-web); then
  if docker exec "$cid" cat /opt/haklolarp-staging/NOPE 2>/dev/null; then
    fail "cell-web could read /opt/haklolarp-staging — host path leaked into cell"
  else
    pass "cell-web cannot read /opt/haklolarp-staging (negative test failed as ordered)"
  fi
  if docker exec "$cid" sh -c 'ls /opt/haklolarp-web 2>/dev/null'; then
    fail "cell-web sees /opt/haklolarp-web — neighbor checkout visible"
  else
    pass "cell-web cannot see /opt/haklolarp-web (negative test failed as ordered)"
  fi
  if docker exec "$cid" cat /proc/1/root/etc/shadow 2>/dev/null; then
    fail "cell-web read /proc/1/root/etc/shadow — BREAKOUT-class leak (privileged/hostPid/host-root mount?)"
  else
    pass "cell-web cannot read /proc/1/root/etc/shadow (breakout probe failed as ordered)"
  fi
else
  skip "cell-web not running — neighbor-visibility exec deferred (needs a live container)"
fi

echo "=== E. migrate receipt gate (health-only on empty volumes is NOT proof) ==="
# backend /healthz is 200 unconditionally (upstream app.py), so a green backend
# on FRESH/EMPTY volumes means nothing. Require EXACTLY ONE of:
#   (1) an exited-0 cell-data-migrate container (run WITHOUT --rm so it persists), or
#   (2) a live schema signal: >=3 tables in the haklolarp DB (init.sql effect).
# HOW TO FURNISH: docker compose -f <cells-file> --profile migrate up cell-data-migrate
if migcid=$(find_cid_any cell-data-migrate); then
  migexit=$(docker inspect --format '{{.State.ExitCode}}' "$migcid" 2>/dev/null || echo "?")
  migstatus=$(docker inspect --format '{{.State.Status}}' "$migcid" 2>/dev/null || echo "?")
  if [ "$migexit" = "0" ]; then
    pass "migrate receipt: $migcid status=$migstatus exit=0"
  else
    fail "migrate container $migcid status=$migstatus exit=$migexit — re-run: --profile migrate up cell-data-migrate"
  fi
else
  skip "no cell-data-migrate container found (never run, or run with --rm) — falling back to live schema signal"
  schema="unknown"
  if chc=$(find_cid cell-data-clickhouse); then
    schema=$(docker exec "$chc" clickhouse-client --query "SELECT count() FROM system.tables WHERE database='haklolarp'" 2>/dev/null || echo "unknown")
  fi
  case "$schema" in
    ''|*[!0-9]*) fail "E migrate: no receipt AND schema signal unreadable ('$schema') — run --profile migrate up cell-data-migrate, then re-run proof";;
    *)
      if [ "$schema" -ge 3 ] 2>/dev/null; then
        pass "E migrate: no container receipt, but live schema present ($schema tables in haklolarp DB)"
      else
        fail "E migrate: no receipt and only $schema tables in haklolarp DB (want >=3) — volumes are EMPTY: seed, run --profile migrate up cell-data-migrate, re-run proof"
      fi
      ;;
  esac
fi

echo "=== SUMMARY: PASS=$PASS FAIL=$FAIL SKIP=$SKIP WARN=$WARN (MIN_PASS=$MIN_PASS) ==="
if [ "$PASS" -lt "$MIN_PASS" ]; then
  fail "hollow run: only $PASS PASS (< MIN_PASS=$MIN_PASS) — cells not running or proof mis-scoped; refusing green"
fi
[ "$FAIL" -eq 0 ]
