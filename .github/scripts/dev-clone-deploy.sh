#!/bin/bash
# dev-clone deploy script (remote side, run over SSH via appleboy/ssh-action).
#
# Extracted from the `Deploy clone over SSH` step of
# .github/workflows/dev-clone.yml to keep the workflow file under GitHub's
# loader limit for inline scripts. Intentional deltas vs the former inline
# block are D1..D3 below; everything else is byte-identical to the inline
# body at extraction time (base a1fcd76: lock-under-HOME + stale-recovery +
# comment fixes preserved).
#
# --- GitHub-expression mapping table (D1) ---
# appleboy/ssh-action does NOT interpolate template expressions inside
# external script content, so every value the script needs arrives via envs
# + step env (namespaced DC_*). Gate: this file must contain zero template
# markers; assert with grep before commit.
#   DC_BRANCH     <- needs.vars.outputs.branchsan  (clone branch)
#   DC_DEV        <- needs.vars.outputs.dev         (dev segment)
#   DC_PROJECT    <- needs.vars.outputs.project    (compose -p name)
#   DC_DIR        <- needs.vars.outputs.dir        (slot dir)
#   DC_APP_PORT   <- needs.vars.outputs.port       (headline web port)
#   DC_RECLONE    <- inputs.reclone                (explicit wipe flag; empty on push)
#   DC_RESEED_REQ <- needs.vars.outputs.reseed     (vars-derived reseed request)
#   DC_REPOSITORY <- github.repository             (owner/repo for git clone)
# The step assigns the short names (BRANCH/DEV/PROJECT/DIR/APP_PORT/RECLONE/
# RESEED_REQ + DC_REPOSITORY in the clone URL) from these at the top; the
# rest of the body is untouched.
#
# D2 (strict mode + set -u compat): header is `set -euo pipefail` (the old
#   inline first line `set -e` is dropped here, not duplicated). Every DC_*
#   deref uses the default-expansion form so an absent var can never abort
#   with an unbound-variable error first: the in-script guardrails
#   downstream (branch/DIR/port asserts) still refuse empties, fail closed.
#   Same for the six CELL_*_SRC refs in the live-path guard loop, which are
#   unset by default (nothing forwards them): bare refs would abort under
#   `set -u`, so they use the default-expansion form. Same fail-closed
#   semantics, no behavior change when the vars are set. Dialect risk (OPEN
#   ITEM): `pipefail` is a bashism - if the remote shell is dash, setting it
#   aborts the script immediately. Remote dialect is unverified from here
#   (no Xeon access); confirm bash before relying on this file live.
# D3 (action input name, verified against the pinned action.yml at
#   0ff4204): the input is `script_path` (mapped to INPUT_SCRIPT_FILE; the
#   pin has NO script_file input - that name would silently run an empty
#   script). Invoked with `script_path:
#   .github/scripts/dev-clone-deploy.sh` plus `envs:` carrying the
#   comma-separated DC_* names.
#
# Required env (forwarded via `envs:` in the workflow step):
#   DC_BRANCH DC_DEV DC_PROJECT DC_DIR DC_APP_PORT DC_RECLONE DC_RESEED_REQ DC_REPOSITORY
# Missing/empty values fail fast at the in-script guardrails below
# (default-expanded above so `set -u` never aborts first), fail closed.
set -euo pipefail
export GIT_SSH_COMMAND="ssh -i ~/.ssh/haklolarp_deploy -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts"
BRANCH="${DC_BRANCH:-}"
DEV="${DC_DEV:-}"
PROJECT="${DC_PROJECT:-}"
DIR="${DC_DIR:-}"
APP_PORT="${DC_APP_PORT:-}"
RECLONE="${DC_RECLONE:-}"
# Gap 2: explicit manual full-reseed flag. Dispatch feeds it from the
# boolean input via vars.reseed; push flows render empty inputs -> off.
# RESEED_REQ only — no FULL_RESEED env passthrough (never automatic);
# only the exact word `true` enables.
RESEED_REQ="${DC_RESEED_REQ:-}"
FULL_RESEED=false
case "$RESEED_REQ" in true) FULL_RESEED=true;; *) FULL_RESEED=false;; esac
# Gap 3: clone target = shared cells file (DECISION (b), see header).
CELLS="ops/sandboxes/cells/docker-compose.cells.yml"
# --- guardrails: branch refusal + sanitize (mirror vars, fail closed on host too) ---
case "$BRANCH" in
  main|master|preview*|devops|web|ml|data|staging*) echo "refusing: branch '$BRANCH' is not a clone target" >&2; exit 1;;
esac
case "$BRANCH" in ''|*[!a-z0-9-]*|'') echo "refusing: branch not lowercase alnum+dash: $BRANCH" >&2; exit 1;; esac
case "$BRANCH" in dev-*-*) ;; *) echo "refusing: want dev-<dev>-<topic>, got: $BRANCH" >&2; exit 1;; esac
REST="${BRANCH#dev-}"; D2="${REST%%-*}"; T2="${REST#*-}"
[ "$D2" = "$DEV" ] || { echo "refusing: dev segment mismatch ($D2 vs $DEV)" >&2; exit 1; }
case "$T2" in ''|*-|-*|*[!a-z0-9-]*) echo "refusing: bad topic segment" >&2; exit 1;; esac
# --- guardrails: DIR prefix + previews refusal + rm safety ---
case "$DIR" in /opt/haklolarp-dev/?*) ;; *) echo "refusing: bad DIR: $DIR" >&2; exit 1;; esac
case "$DIR" in /opt/haklolarp-previews/*) echo "refusing: never touch preview dirs" >&2; exit 1;; esac
[ "$DIR" != "/" ] || { echo "refusing: DIR is /" >&2; exit 1; }
[ "$DIR" != "/opt/haklolarp-dev" ] || { echo "refusing: DIR is parent" >&2; exit 1; }
[ "$DIR" != "/opt/haklolarp-dev/" ] || { echo "refusing: DIR is parent" >&2; exit 1; }
# --- guardrails: port numeric + denylist (live stands, preview range, sandbox range) ---
case "$APP_PORT" in ''|*[!0-9]*) echo "refusing: bad APP_PORT: $APP_PORT" >&2; exit 1;; esac
DENY=" 8080 8081 8082 5434 6379 8123 8000 "
case "$DENY" in *" $APP_PORT "*) echo "refusing: APP_PORT $APP_PORT is a LIVE stand port (denylist:$DENY)" >&2; exit 1;; esac
if [ "$APP_PORT" -ge 9000 ] && [ "$APP_PORT" -le 9100 ]; then echo "refusing: APP_PORT $APP_PORT in preview range 9000-9100" >&2; exit 1; fi
if [ "$APP_PORT" -ge 18081 ]; then echo "refusing: APP_PORT $APP_PORT in sandbox range 18081+" >&2; exit 1; fi
export APP_PORT
# --- per-dev atomic lock (B2): serialize concurrent same-dev pushes over check→up ---
# Lock lives in $HOME (hakloboss-writable), NOT /opt (root-owned:
# mkdir there always fails and used to misreport as contention).
LOCK_BASE="$HOME/.sandbox-locks"
mkdir -p "$LOCK_BASE" || { echo "refusing: cannot create lock base $LOCK_BASE" >&2; exit 1; }
LOCK="$LOCK_BASE/dev-$DEV.lock"
# TTL 1800s (30 min) exceeds any legit check→up hold (ssh
# command_timeout is 25m incl. the 15x60s fairness wait + build), so
# a lock older than this cannot be a live holder — only a SIGKILL/
# reboot/transport-death leak (trap below covers EXIT/INT/TERM only
# and can never run on KILL/reboot).
LOCK_TTL=1800
LOCK_ERR=""
if ! LOCK_ERR="$(mkdir "$LOCK" 2>&1)"; then
  case "$LOCK_ERR" in
    *"File exists"*)
      TRY=0
      while [ "$TRY" -lt 3 ]; do
        TRY=$((TRY+1))
        MTIME="$(stat -c %Y "$LOCK" 2>/dev/null || true)"
        NOW="$(date +%s 2>/dev/null || true)"
        case "$MTIME" in ''|*[!0-9]*) echo "refusing: cannot acquire lock $LOCK (cannot stat contender age; mkdir said: $LOCK_ERR)" >&2; exit 1;; esac
        case "$NOW" in ''|*[!0-9]*) echo "refusing: cannot acquire lock $LOCK (cannot read clock; mkdir said: $LOCK_ERR)" >&2; exit 1;; esac
        AGE=$((NOW-MTIME))
        if [ "$AGE" -gt "$LOCK_TTL" ]; then
          rmdir "$LOCK" 2>/dev/null || true
          if LOCK_ERR="$(mkdir "$LOCK" 2>&1)"; then
            echo "dev-clone: recovered stale lock $LOCK (age ${AGE}s > TTL ${LOCK_TTL}s, try $TRY/3)" >&2
            LOCK_ERR=""
            break
          fi
          case "$LOCK_ERR" in *"File exists"*) continue;; *) echo "refusing: cannot acquire lock $LOCK (infra-error after stale rmdir: $LOCK_ERR)" >&2; exit 1;; esac
        else
          echo "refusing: concurrent push for dev '$DEV' holds $LOCK (check→up in progress, age ${AGE}s); retry shortly — second contender is refused, never corrupting the first ($LOCK_ERR)" >&2; exit 1
        fi
      done
      case "$LOCK_ERR" in '') ;; *) echo "refusing: concurrent push for dev '$DEV' holds $LOCK (stale-steal race lost after 3 tries); retry shortly — second contender is refused, never corrupting the first ($LOCK_ERR)" >&2; exit 1;; esac
      ;;
    *) echo "refusing: cannot acquire lock $LOCK (infra-error: $LOCK_ERR)" >&2; exit 1;;
  esac
fi
# trap covers EXIT/INT/TERM only — SIGKILL/reboot/transport-death
# cannot trap, hence the TTL steal above (rmdir only, never rm -rf).
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT INT TERM
# Cells-aware teardown: prefer the cells file when the slot checkout
# has it, fall back to the legacy root stub, then sweep any residual
# project-labelled containers/volumes (covers lock-less dirs and the
# migrate receipt container). Only ever touches
# /opt/haklolarp-dev/* paths (asserted by callers).
compose_down() {
  slot="$1"; op="$2"
  case "$op" in "haklolarp-$DEV-"*) ;; *) echo "refusing: bad PROJECT for down: $op" >&2; return 1;; esac
  case "$op" in ''|*[!a-z0-9-]*) echo "refusing: bad PROJECT for down: $op" >&2; return 1;; esac
  if [ -f "$slot/$CELLS" ]; then ( cd "$slot" && docker compose -f "$CELLS" -p "$op" down --rmi local --volumes --remove-orphans ) || true
  elif [ -f "$slot/compose.yaml" ]; then ( cd "$slot" && docker compose -p "$op" down --rmi local --volumes --remove-orphans ) || true; fi
  docker ps -aq --filter "label=com.docker.compose.project=$op" 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
  docker volume ls -q --filter "label=com.docker.compose.project=$op" 2>/dev/null | xargs -r docker volume rm 2>/dev/null || true
}
# --- one clone per dev: second branch FAILS unless explicit reclone ---
if [ "$RECLONE" = "true" ]; then
  echo "dev-clone: reclone=true, wiping own dev slot /opt/haklolarp-dev/$DEV-* explicitly"
  for d in /opt/haklolarp-dev/"$DEV"-*; do
    [ -e "$d" ] || continue
    case "$d" in /opt/haklolarp-dev/?*) ;; *) continue;; esac
    if [ -f "$d/CLONE.lock" ]; then
      OP=$(awk -F= '$1=="PROJECT" {print $2}' "$d/CLONE.lock" 2>/dev/null || true)
      case "$OP" in "haklolarp-$DEV-"*) ;; *) OP="";; esac
      case "$OP" in ''|*[!a-z0-9-]*) OP="";; esac
      if [ -n "$OP" ]; then compose_down "$d" "$OP" || true; fi
    else
      b=$(basename "$d"); br="${b#$DEV-}"; OP="haklolarp-$DEV-$br"
      case "$OP" in "haklolarp-$DEV-"*) ;; *) OP="";; esac
      case "$OP" in ''|*[!a-z0-9-]*) OP="";; esac
      if [ -n "$OP" ]; then echo "dev-clone: WARN $d has no CLONE.lock, tearing down derived project $OP" >&2; compose_down "$d" "$OP" || true; fi
    fi
    rm -rf -- "$d"
  done
else
  for d in /opt/haklolarp-dev/"$DEV"-*; do
    [ -e "$d" ] || continue
    [ "$d" = "$DIR" ] || { echo "refusing: one clone per dev — '$DEV' already owns '$d' (stale slot). Recovery: delete that branch (auto-cleanup removes the clone), or explicitly wipe+recreate via: gh workflow run dev-clone.yml --ref $BRANCH -f reclone=true" >&2; exit 1; }
  done
fi
# --- fairness: max 3 concurrent clones, check-then-wait, never kill others ---
MAX_CLONES=3; WAIT_ROUNDS=15; WAIT_SECS=60
count_others() {
  n=0
  for d in /opt/haklolarp-dev/*; do
    [ -e "$d" ] || continue
    [ "$d" = "$DIR" ] && continue
    case "$d" in /opt/haklolarp-dev/?*) ;; *) continue;; esac
    if [ -f "$d/CLONE.lock" ]; then
      OP=$(awk -F= '$1=="PROJECT" {print $2}' "$d/CLONE.lock" 2>/dev/null || true)
      if [ -n "$OP" ] && [ -z "$(docker ps -q --filter "label=com.docker.compose.project=$OP" 2>/dev/null | head -n 1)" ]; then continue; fi
    fi
    n=$((n+1))
  done
  echo "$n"
}
OTHERS=$(count_others)
i=0
while [ "$OTHERS" -ge "$MAX_CLONES" ] && [ "$i" -lt "$WAIT_ROUNDS" ]; do
  echo "dev-clone: $OTHERS/$MAX_CLONES slots busy, waiting ${WAIT_SECS}s (round $((i+1))/$WAIT_ROUNDS, never killing others)"
  sleep "$WAIT_SECS"
  OTHERS=$(count_others); i=$((i+1))
done
if [ "$OTHERS" -ge "$MAX_CLONES" ]; then echo "refusing: clone slots full ($MAX_CLONES/$MAX_CLONES busy), retry later" >&2; exit 1; fi
# --- port block across devs: web + 4 cell service ports from 9101..9198 ---
# APP_PORT stays the headline (web ingress, smoke curl, receipt). The
# other four cell ports derive as headline+1..+4 with wrap inside
# 9101..9198. A clone owns the whole block: bump the headline (+1,
# wrap) until NONE of the 5 collide with any port recorded in other
# devs' CLONE.lock files (legacy APP_PORT + CELL_* lines, so mixed
# old/new clones coexist), inside the per-dev lock. Idempotent
# re-push reuses the own block from CLONE.lock when still in range.
port_add() { pa="$1"; po="$2"; pa=$((pa+po)); while [ "$pa" -gt 9198 ]; do pa=$((pa-98)); done; echo "$pa"; }
BASE_PORT="$APP_PORT"
WEB="$APP_PORT"
if [ -f "$DIR/CLONE.lock" ]; then
  OWN_WEB=$(awk -F= '$1=="CELL_WEB_PORT" {print $2}' "$DIR/CLONE.lock" 2>/dev/null || true)
  case "$OWN_WEB" in ''|*[!0-9]*) OWN_WEB=$(awk -F= '$1=="APP_PORT" {print $2}' "$DIR/CLONE.lock" 2>/dev/null || true);; esac
  case "$OWN_WEB" in ''|*[!0-9]*) ;; *) if [ "$OWN_WEB" -ge 9101 ] && [ "$OWN_WEB" -le 9198 ]; then WEB="$OWN_WEB"; fi;; esac
fi
block_taken() {
  for want in "$@"; do
    for d in /opt/haklolarp-dev/*; do
      [ -e "$d" ] || continue; [ "$d" = "$DIR" ] && continue
      case "$d" in /opt/haklolarp-dev/?*) ;; *) continue;; esac
      [ -f "$d/CLONE.lock" ] || continue
      if awk -F= '$1=="APP_PORT"||$1=="CELL_WEB_PORT"||$1=="CELL_ML_PORT"||$1=="CELL_REDIS_PORT"||$1=="CELL_CH_HTTP_PORT"||$1=="CELL_DATA_API_PORT" {print $2}' "$d/CLONE.lock" 2>/dev/null | grep -qx -- "$want"; then return 0; fi
    done
  done
  return 1
}
if block_taken "$WEB" "$(port_add "$WEB" 1)" "$(port_add "$WEB" 2)" "$(port_add "$WEB" 3)" "$(port_add "$WEB" 4)"; then
  P="$BASE_PORT"; TRIED=0
  while true; do
    if ! block_taken "$P" "$(port_add "$P" 1)" "$(port_add "$P" 2)" "$(port_add "$P" 3)" "$(port_add "$P" 4)"; then break; fi
    P=$((P+1)); if [ "$P" -gt 9198 ]; then P=9101; fi
    TRIED=$((TRIED+1))
    if [ "$TRIED" -ge 98 ]; then echo "refusing: all 98 clone ports 9101..9198 busy (hash block full), retry later" >&2; exit 1; fi
  done
  WEB="$P"
  echo "dev-clone: port-block collision on $BASE_PORT, bumped web to $WEB (block persisted in CLONE.lock)"
fi
APP_PORT="$WEB"; export APP_PORT
CELL_WEB_PORT="$WEB"; CELL_ML_PORT=$(port_add "$WEB" 1); CELL_REDIS_PORT=$(port_add "$WEB" 2); CELL_CH_HTTP_PORT=$(port_add "$WEB" 3); CELL_DATA_API_PORT=$(port_add "$WEB" 4)
export CELL_WEB_PORT CELL_ML_PORT CELL_REDIS_PORT CELL_CH_HTTP_PORT CELL_DATA_API_PORT
# --- guardrails: derived cell ports pass the same denylist (fail closed by construction, asserted anyway) ---
for cp in "$CELL_WEB_PORT" "$CELL_ML_PORT" "$CELL_REDIS_PORT" "$CELL_CH_HTTP_PORT" "$CELL_DATA_API_PORT"; do
  case "$cp" in ''|*[!0-9]*) echo "refusing: bad cell port: $cp" >&2; exit 1;; esac
  case "$DENY" in *" $cp "*) echo "refusing: cell port $cp is a LIVE stand port (denylist:$DENY)" >&2; exit 1;; esac
  if [ "$cp" -ge 9000 ] && [ "$cp" -le 9100 ]; then echo "refusing: cell port $cp in preview range 9000-9100" >&2; exit 1; fi
  if [ "$cp" -ge 18081 ]; then echo "refusing: cell port $cp in sandbox range 18081+" >&2; exit 1; fi
done
# --- sync: shared branch checkout, base SHA of origin/main ---
if [ ! -d "$DIR/.git" ]; then
  git clone "git@github.com:${DC_REPOSITORY:-}.git" "$DIR"
fi
cd "$DIR"
git fetch origin "$BRANCH"
git checkout -f FETCH_HEAD
SHA=$(git rev-parse HEAD)
git fetch origin main
BASE_SHA=$(git rev-parse origin/main)
BASE_DATE=$(git log -1 --format=%ci origin/main)
BR_DATE=$(git log -1 --format=%ci HEAD)
# --- compose guards: shared cells file only, per-dev port vars, no live refs ---
[ -f "$CELLS" ] || { echo "refusing: no $CELLS in branch checkout (rebase the dev branch onto main; no compose forks)" >&2; exit 1; }
for pv in CELL_WEB_PORT CELL_ML_PORT CELL_REDIS_PORT CELL_CH_HTTP_PORT CELL_DATA_API_PORT; do
  grep -q -F -- "\${$pv" "$CELLS" || { echo "refusing: cells file must publish via $pv (shared-file pattern)" >&2; exit 1; }
done
# Refuse live-checkout bind overrides: CELL_*_SRC default to the repo
# root (safe); an explicit override pointing at a live stand is refused.
for s in "${CELL_WEB_SRC:-}" "${CELL_ML_SRC:-}" "${CELL_DATA_SRC:-}" "${CELL_WEB_RECORDS_SRC:-}" "${CELL_WEB_QUEUE_SRC:-}" "${CELL_WEB_RESEARCH_SRC:-}"; do
  [ -z "${s:-}" ] && continue
  case "$s" in /opt/haklolarp-staging*|/opt/haklolarp-web*|/opt/haklolarp-ml*|/opt/haklolarp-stack*|/opt/haklolarp-previews*|/opt/haklolarp-main*) echo "refusing: CELL_*_SRC points at live path: $s" >&2; exit 1;; esac
done
CFG=$(docker compose -f "$CELLS" -p "$PROJECT" config)
if printf '%s\n' "$CFG" | grep -q -E '0\.0\.0\.0'; then echo "refusing: wildcard 0.0.0.0 bind in rendered compose (loopback-only)" >&2; exit 1; fi
if printf '%s\n' "$CFG" | grep -E '^[[:space:]]*host_ip:' | grep -v -E '127\.0\.0\.1' | grep -q .; then echo "refusing: non-loopback host_ip in rendered compose" >&2; exit 1; fi
if printf '%s\n' "$CFG" | grep -q -E "network_mode:[[:space:]]*[\"']?host[\"']?"; then echo "refusing: network_mode: host in rendered compose (host network forbidden)" >&2; exit 1; fi
for cp in "$CELL_WEB_PORT" "$CELL_ML_PORT" "$CELL_REDIS_PORT" "$CELL_CH_HTTP_PORT" "$CELL_DATA_API_PORT"; do
  printf '%s\n' "$CFG" | grep -E -q -- "(^|[^0-9A-Za-z_])$cp([^0-9A-Za-z_]|$)" || { echo "refusing: rendered compose missing cell port $cp" >&2; exit 1; }
done
for LIVE in /opt/haklolarp-staging /opt/haklolarp-web /opt/haklolarp-ml /opt/haklolarp-previews /opt/haklolarp-stack; do
  if printf '%s\n' "$CFG" | grep -q -F -- "$LIVE"; then echo "refusing: rendered compose references live path $LIVE (no live-volume mounts)" >&2; exit 1; fi
done
if printf '%s\n' "$CFG" | grep -q -E 'docker\.sock'; then echo "refusing: docker.sock mount in rendered compose" >&2; exit 1; fi
# --- loopback-only assert on every published port (short + long syntax, stanza-aware) ---
# Stanza-aware: ONLY list items under a `ports:` key are parsed. Rendered
# `docker compose config` expands every other digit-leading list (redis
# `command: ["redis-server", ..., "4gb", ...]` renders as `- 4gb`, healthcheck
# `test: ["CMD", ..., "3", ...]` renders as `- "3"`, plus expose/dns/tmpfs
# entries) to the same `- <digit>` shape the old global grep matched — that is
# what choked the old parser with `cannot parse published port '4gb'`. Those
# lines never reach the parser below: awk enters on a `ports:` key at indent N
# and leaves on the next key at indent <= N, emitting only the `- ...` item
# starters and their (long-syntax) continuation lines.
PORTS_STANZA=$(mktemp)
printf '%s\n' "$CFG" | awk '
  function ind(s,   i, n, c) {
    n = length(s)
    for (i = 1; i <= n; i++) { c = substr(s, i, 1); if (c != " " && c != "\t") return i - 1 }
    return n
  }
  !in_ports {
    if ($0 ~ /^[[:space:]]*ports:[[:space:]]*(\[\][[:space:]]*)?(#.*)?$/) { in_ports = 1; ports_ind = ind($0); next }
    if ($0 ~ /^[[:space:]]*ports:[[:space:]]*[^[:space:]#]/) { in_ports = 1; ports_ind = ind($0); print $0; next }
    next
  }
  in_ports {
    if ($0 ~ /^[[:space:]]*$/) next
    if ($0 ~ /^[[:space:]]*#/) next
    n = ind($0)
    if (n <= ports_ind && $0 ~ /^[[:space:]]*[A-Za-z_][A-Za-z0-9_.-]*:/) {
      in_ports = 0
      if ($0 ~ /^[[:space:]]*ports:[[:space:]]*(\[\][[:space:]]*)?(#.*)?$/) { in_ports = 1; ports_ind = n }
      else if ($0 ~ /^[[:space:]]*ports:[[:space:]]*[^[:space:]#]/) { in_ports = 1; ports_ind = n; print $0 }
      next
    }
    if ($0 ~ /^[[:space:]]*-([[:space:]]|$)/) { print $0; next }
    if (n > ports_ind) { print $0; next }
    in_ports = 0
  }
' > "$PORTS_STANZA" || { echo "refusing: cannot extract ports stanzas from rendered compose" >&2; exit 1; }
# Per-item long-syntax check: a `- target: ...` (or any long-key) starter opens
# a block; the block MUST carry `host_ip: 127.0.0.1` (a missing host_ip
# defaults to 0.0.0.0 at runtime, so it is REFUSED, never defaulted) and its
# `published:` value must clear the same denylist/range guards as short syntax.
LONG_ACTIVE=0; BLOCK_PUB=""; BLOCK_HIP=""
check_long_block() {
  if [ "$LONG_ACTIVE" != "1" ]; then return 0; fi
  case "$BLOCK_HIP" in
    '') echo "refusing: long-syntax publish without host_ip (defaults to 0.0.0.0; pin host_ip: 127.0.0.1)" >&2; exit 1;;
    127.0.0.1) ;;
    *) echo "refusing: long-syntax publish with non-loopback host_ip '$BLOCK_HIP' (pin 127.0.0.1)" >&2; exit 1;;
  esac
  case "$BLOCK_PUB" in
    ''|*[!0-9]*) echo "refusing: bad long-syntax published port '$BLOCK_PUB'" >&2; exit 1;;
  esac
  case "$DENY" in *" $BLOCK_PUB "*) echo "refusing: published denylisted port $BLOCK_PUB (long-syntax)" >&2; exit 1;; esac
  if [ "$BLOCK_PUB" -ge 9000 ] && [ "$BLOCK_PUB" -le 9100 ]; then echo "refusing: published port $BLOCK_PUB in preview range (long-syntax)" >&2; exit 1; fi
  if [ "$BLOCK_PUB" -ge 18081 ]; then echo "refusing: published port $BLOCK_PUB in sandbox range (long-syntax)" >&2; exit 1; fi
  return 0
}
while IFS= read -r line; do
  case "$line" in '') continue;; esac
  if printf '%s' "$line" | grep -q -E '^[[:space:]]*-([[:space:]]|$)'; then
    check_long_block
    LONG_ACTIVE=0; BLOCK_PUB=""; BLOCK_HIP=""
    rest=$(printf '%s' "$line" | sed -e 's/^[[:space:]]*-[[:space:]]*//')
    case "$rest" in
      *target:*|*published:*|*host_ip:*|*protocol:*|*mode:*|*app_protocol:*)
        LONG_ACTIVE=1
        if printf '%s' "$rest" | grep -q -E '(^|[[:space:]])published:[[:space:]]*[^[:space:]]'; then
          BLOCK_PUB=$(printf '%s' "$rest" | sed -e 's/.*published:[[:space:]]*//' -e 's/#.*$//' | awk '{print $1}' | tr -d '"' | tr -d "'" | tr -d ',{}[]')
        fi
        if printf '%s' "$rest" | grep -q -E '(^|[[:space:]])host_ip:[[:space:]]*[^[:space:]]'; then
          BLOCK_HIP=$(printf '%s' "$rest" | sed -e 's/.*host_ip:[[:space:]]*//' -e 's/#.*$//' | awk '{print $1}' | tr -d '"' | tr -d "'" | tr -d ',{}[]')
        fi
        ;;
      *)
        m=$(printf '%s' "$line" | sed -e 's/^[[:space:]]*-[[:space:]]*//' | cut -d' ' -f1 | tr -d '"' | tr -d "'")
        case "$m" in
          127.0.0.1:*) hp=$(printf '%s' "$m" | sed -e 's/^127\.0\.0\.1://' -e 's/:.*//'); UNBOUND=0;;
          *:*:*) echo "refusing: published '$m' has explicit non-loopback bind" >&2; exit 1;;
          *:* ) hp=$(printf '%s' "$m" | cut -d: -f1); UNBOUND=1;;
          *) echo "refusing: cannot parse published port '$m' (bare ports auto-publish on 0.0.0.0; pin 127.0.0.1:<port>:<container>)" >&2; exit 1;;
        esac
        case "$hp" in ''|*[!0-9]*) echo "refusing: bad host port '$hp' in '$m'" >&2; exit 1;; esac
        case "$DENY" in *" $hp "*) echo "refusing: published denylisted port $hp ($m)" >&2; exit 1;; esac
        if [ "$hp" -ge 9000 ] && [ "$hp" -le 9100 ]; then echo "refusing: published port $hp in preview range ($m)" >&2; exit 1; fi
        if [ "$hp" -ge 18081 ]; then echo "refusing: published port $hp in sandbox range ($m)" >&2; exit 1; fi
        if [ "$UNBOUND" = "1" ] && [ "$hp" != "$APP_PORT" ]; then echo "refusing: published '$m' without loopback bind (only APP_PORT $APP_PORT allowed unbound; pin 127.0.0.1:<port> in compose)" >&2; exit 1; fi
        ;;
    esac
  else
    if [ "$LONG_ACTIVE" != "1" ]; then echo "refusing: unexpected ports-stanza line '$line' (fail closed)" >&2; exit 1; fi
    if printf '%s' "$line" | grep -q -E '(^|[[:space:]])published:[[:space:]]*[^[:space:]]'; then
      BLOCK_PUB=$(printf '%s' "$line" | sed -e 's/.*published:[[:space:]]*//' -e 's/#.*$//' | awk '{print $1}' | tr -d '"' | tr -d "'" | tr -d ',{}[]')
    fi
    if printf '%s' "$line" | grep -q -E '(^|[[:space:]])host_ip:[[:space:]]*[^[:space:]]'; then
      BLOCK_HIP=$(printf '%s' "$line" | sed -e 's/.*host_ip:[[:space:]]*//' -e 's/#.*$//' | awk '{print $1}' | tr -d '"' | tr -d "'" | tr -d ',{}[]')
    fi
  fi
done < "$PORTS_STANZA"
check_long_block
rm -f "$PORTS_STANZA"
# --- up: full cells program set, fresh prefixed named volumes (no live data ever) ---
# pull is warn-only: build services have no pullable image; `up --build`
# below is the real delivery (rebuilt every push: the dev branch moved).
docker compose -f "$CELLS" -p "$PROJECT" pull || echo "dev-clone: WARN compose pull failed, continuing to build" >&2
docker compose -f "$CELLS" -p "$PROJECT" up -d --build --remove-orphans
# --- full reseed (Gap 2): explicit manual flag ONLY, default off ---
# Empty volumes + migrate schema stay the default. When
# FULL_RESEED=true: stop the CLONE's data services (clone ONLY —
# live is NEVER stopped/restarted/removed, only inspected), copy
# live volumes read-only (`--volumes-from <live>:ro` tar pipe per
# docs), restart the clone services. Fail closed: a missing live
# source aborts with NO green (never silently empty).
SEED=cells-migrate
if [ "$FULL_RESEED" = "true" ]; then
  echo "dev-clone: full_reseed=true, snapshotting live data volumes read-only"
  LIVE_STACK_PROJECT="${LIVE_STACK_PROJECT:-haklolarp-stack}"
  case "$LIVE_STACK_PROJECT" in ''|*[!a-z0-9-]*) echo "refusing: bad LIVE_STACK_PROJECT: $LIVE_STACK_PROJECT" >&2; exit 1;; esac
  [ "$LIVE_STACK_PROJECT" != "$PROJECT" ] || { echo "refusing: live project resolves to own clone project" >&2; exit 1; }
  # Helper image: the clone's own redis image (busybox sh+tar,
  # already pulled by up). Fail closed if the clone is not up.
  HELPER_C=$(docker ps -q --filter "label=com.docker.compose.project=$PROJECT" --filter "label=com.docker.compose.service=cell-data-redis" | head -n 1)
  [ -n "$HELPER_C" ] || { echo "reseed FAIL: clone cell-data-redis missing (up failed?)" >&2; exit 1; }
  HELPER_IMG=$(docker inspect --format '{{.Config.Image}}' "$HELPER_C" 2>/dev/null || true)
  [ -n "$HELPER_IMG" ] || { echo "reseed FAIL: cannot resolve helper image" >&2; exit 1; }
  RESEED_LIVE_IDS=""
  # reseed_pair $live_svc $mount_path $clone_svc
  reseed_pair() {
    live_svc="$1"; mnt="$2"; clone_svc="$3"
    live_c=$(docker ps -q --filter "label=com.docker.compose.project=$LIVE_STACK_PROJECT" --filter "label=com.docker.compose.service=$live_svc" | head -n 1)
    [ -n "$live_c" ] || { echo "reseed FAIL: live source missing (project=$LIVE_STACK_PROJECT service=$live_svc not running) — refusing to mark green on empty volumes" >&2; exit 1; }
    live_proj=$(docker inspect --format '{{index .Config.Labels "com.docker.compose.project"}}' "$live_c" 2>/dev/null || true)
    [ "$live_proj" != "$PROJECT" ] || { echo "refusing: live container resolves to own clone project" >&2; exit 1; }
    clone_c=$(docker ps -aq --filter "label=com.docker.compose.project=$PROJECT" --filter "label=com.docker.compose.service=$clone_svc" | head -n 1)
    [ -n "$clone_c" ] || { echo "reseed FAIL: clone container missing ($clone_svc)" >&2; exit 1; }
    src_vol=$(docker inspect --format "{{range .Mounts}}{{if eq .Destination \"$mnt\"}}{{.Name}}{{end}}{{end}}" "$live_c" 2>/dev/null || true)
    dst_vol=$(docker inspect --format "{{range .Mounts}}{{if eq .Destination \"$mnt\"}}{{.Name}}{{end}}{{end}}" "$clone_c" 2>/dev/null || true)
    case "$src_vol" in ''|*/*) echo "reseed FAIL: live $live_svc has no NAMED volume at $mnt (bind mounts refused)" >&2; exit 1;; esac
    case "$dst_vol" in ''|*/*) echo "reseed FAIL: clone $clone_svc has no NAMED volume at $mnt" >&2; exit 1;; esac
    [ "$src_vol" != "$dst_vol" ] || { echo "refusing: live and clone volumes are the same ($src_vol)" >&2; exit 1; }
    docker compose -f "$CELLS" -p "$PROJECT" stop "$clone_svc"
    docker run --rm --volumes-from "$live_c:ro" -v "$dst_vol:/restore" "$HELPER_IMG" sh -c "set -e; find /restore -mindepth 1 -delete; tar cf - -C '$mnt' . | tar xf - -C /restore"
    docker compose -f "$CELLS" -p "$PROJECT" start "$clone_svc"
    RESEED_LIVE_IDS="$RESEED_LIVE_IDS $live_svc=$live_c"
    echo "reseed OK: $LIVE_STACK_PROJECT/$live_svc:$mnt -> $PROJECT/$clone_svc ($dst_vol)"
  }
  reseed_pair redis /data cell-data-redis
  reseed_pair clickhouse /var/lib/clickhouse cell-data-clickhouse
  RESEED_IMGS=""
  for pair in $RESEED_LIVE_IDS; do
    svc="${pair%%=*}"; cid="${pair#*=}"
    img=$(docker inspect --format '{{.Image}}' "$cid" 2>/dev/null || echo missing)
    RESEED_IMGS="$RESEED_IMGS $svc=$img"
  done
  SEED=full-reseed
  RESEED_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "reseed provenance:$RESEED_IMGS date=$RESEED_DATE"
fi
# --- data: migrate one-shot receipt gate (schema proof) ---
# Backend /healthz is 200 even on EMPTY volumes, so health-only is
# NOT proof: the migrate exit-0 receipt (kept, never --rm) is the gate.
docker compose -f "$CELLS" -p "$PROJECT" --profile migrate up --force-recreate cell-data-migrate
MIGRATE_C=$(docker ps -aq --latest --filter "label=com.docker.compose.project=$PROJECT" --filter "label=com.docker.compose.service=cell-data-migrate")
[ -n "$MIGRATE_C" ] || { echo "migrate FAIL: no receipt container for cell-data-migrate" >&2; exit 1; }
MIGRATE_EXIT=$(docker inspect --format '{{.State.ExitCode}}' "$MIGRATE_C" 2>/dev/null || echo missing)
[ "$MIGRATE_EXIT" = "0" ] || { echo "migrate FAIL: cell-data-migrate exit=$MIGRATE_EXIT (schema not applied)" >&2; exit 1; }
echo "migrate OK: cell-data-migrate exited 0 (receipt $MIGRATE_C)"
# --- health gates: every long-running cell service must be healthy before green ---
i=0
while true; do
  UNHEALTHY=""
  for s in cell-web cell-web-api cell-ml cell-data-redis cell-data-clickhouse cell-data-backend cell-data-ingester; do
    cid=$(docker ps -q --filter "label=com.docker.compose.project=$PROJECT" --filter "label=com.docker.compose.service=$s" | head -n 1)
    if [ -z "$cid" ]; then UNHEALTHY="$UNHEALTHY $s(missing)"; continue; fi
    h=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$cid" 2>/dev/null || echo missing)
    running=$(docker inspect --format '{{.State.Running}}' "$cid" 2>/dev/null || echo false)
    if [ "$h" = "none" ]; then
      [ "$running" = "true" ] || UNHEALTHY="$UNHEALTHY $s(not-running)"
    else
      [ "$h" = "healthy" ] || UNHEALTHY="$UNHEALTHY $s($h)"
    fi
  done
  [ -z "$UNHEALTHY" ] && break
  i=$((i+1))
  if [ "$i" -ge 24 ]; then echo "health FAIL:$UNHEALTHY (after ~120s)" >&2; exit 1; fi
  sleep 5
done
echo "health OK: all cell services healthy"
docker ps --filter "label=com.docker.compose.project=$PROJECT"
# --- light smoke: ps + curl + manifest ---
if [ -z "$(docker ps -q --filter "label=com.docker.compose.project=$PROJECT" | head -n 1)" ]; then echo "smoke FAIL: no running containers for $PROJECT" >&2; exit 1; fi
i=0
until curl -fsS "http://127.0.0.1:$APP_PORT/" >/dev/null 2>&1; do
  i=$((i+1))
  if [ "$i" -ge 24 ]; then echo "smoke FAIL: curl http://127.0.0.1:$APP_PORT/ failed after ~120s" >&2; exit 1; fi
  sleep 5
done
echo "smoke OK: curl http://127.0.0.1:$APP_PORT/ reachable"
curl -fsS --max-time 10 --retry 2 --retry-delay 2 "http://127.0.0.1:$CELL_ML_PORT/health" >/dev/null || { echo "smoke FAIL: curl http://127.0.0.1:$CELL_ML_PORT/health" >&2; exit 1; }
curl -fsS --max-time 10 --retry 2 --retry-delay 2 "http://127.0.0.1:$CELL_DATA_API_PORT/healthz" >/dev/null || { echo "smoke FAIL: curl http://127.0.0.1:$CELL_DATA_API_PORT/healthz" >&2; exit 1; }
echo "smoke OK: ml /health + data /healthz reachable"
IMGS=$(docker compose -f "$CELLS" -p "$PROJECT" images 2>/dev/null || true)
{
  echo "PROJECT=$PROJECT"; echo "BRANCH=$BRANCH"; echo "DEV=$DEV"; echo "APP_PORT=$APP_PORT"
  echo "CELL_WEB_PORT=$CELL_WEB_PORT"; echo "CELL_ML_PORT=$CELL_ML_PORT"; echo "CELL_REDIS_PORT=$CELL_REDIS_PORT"; echo "CELL_CH_HTTP_PORT=$CELL_CH_HTTP_PORT"; echo "CELL_DATA_API_PORT=$CELL_DATA_API_PORT"
  echo "CELLS_FILE=$CELLS"
  echo "SHA=$SHA"; echo "BASE_SHA=$BASE_SHA"; echo "BASE_DATE=$BASE_DATE"; echo "BR_DATE=$BR_DATE"
  echo "DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "SEED=$SEED"
  echo "MIGRATE_EXIT=${MIGRATE_EXIT:-unknown}"
  echo "RESEED=$FULL_RESEED"
  if [ "$FULL_RESEED" = "true" ]; then
    echo "RESEED_LIVE_PROJECT=${LIVE_STACK_PROJECT:-unknown}"; echo "RESEED_SOURCES=${RESEED_IMGS:-unknown}"; echo "RESEED_DATE=${RESEED_DATE:-unknown}"
  fi
  printf '%s\n' "$IMGS"
} > "$DIR/CLONE.lock"
test -s "$DIR/CLONE.lock" || { echo "smoke FAIL: CLONE.lock manifest missing" >&2; exit 1; }
mkdir -p "$DIR/clone-bundle"
printf '%s\n' "$CFG" > "$DIR/clone-bundle/compose.config"
docker ps --filter "label=com.docker.compose.project=$PROJECT" > "$DIR/clone-bundle/ps.txt"
docker compose -f "$CELLS" -p "$PROJECT" logs --no-color --tail=200 > "$DIR/clone-bundle/logs.txt" 2>&1 || true
cp "$DIR/CLONE.lock" "$DIR/clone-bundle/CLONE.lock"
echo "dev-clone ready: $PROJECT @ $DIR :$APP_PORT base=$BASE_SHA"
