#!/usr/bin/env bash
# CLOSED team deployment. Default action applies the deployment; --check is
# the explicitly read-only preflight.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

PROJECT=haklolarp-web
TEAM_ENV_FILE="${DTS_TEAM_ENV_FILE:-/etc/haklolarp-web/team.env}"
ACTION=apply
SESSION=""
for arg in "$@"; do
  case "$arg" in
    --check) ACTION=check ;;
    --apply) ACTION=apply ;;
    --help|-h)
      cat <<'USAGE'
Usage: bash ops/deploy-web.sh [--check|--apply] <tmux-session>

With --check, validates the private team configuration and Compose model only.
Otherwise this script backs up existing PostgreSQL/runtime state, then deploys
the CLOSED team stack. The session label is required for compatibility with
the existing remote runner; this script does not create the tmux session.
USAGE
      exit 0
      ;;
    *)
      if [[ -z "$SESSION" ]]; then SESSION="$arg"; else
        echo "deploy-web: unexpected argument" >&2
        exit 2
      fi
      ;;
  esac
done

COMPOSE_ARGS=(
  --env-file "$TEAM_ENV_FILE"
  -p "$PROJECT"
  -f compose.web.yaml
  -f compose.platform.yaml
  -f compose.team.yaml
)
export DTS_BUILD_SHA="$(git rev-parse HEAD)"

die() {
  echo "deploy-web: $1" >&2
  exit 1
}

bash "$REPO_DIR/scripts/verify-team-deployment.sh" --preflight
if [[ "$ACTION" == "check" ]]; then
  echo "Team deployment configuration is valid; no services or data were changed."
  exit 0
fi

# Serialize updates without putting credentials or runtime data in the lock.
mkdir -p .runtime
exec 9>.runtime/team-deploy.lock
flock -n 9 || die "another team deployment is already in progress"

# Keep compatibility with the reusable workflow's tmux caller, while allowing
# an operator to run the deployment script directly.
if [[ -z "$SESSION" ]]; then
  die "usage: deploy-web.sh [--apply] <tmux-session>"
fi
if [[ -n "$SESSION" && ! "$SESSION" =~ ^[A-Za-z0-9_.-]{1,80}$ ]]; then
  die "invalid tmux session label"
fi

docker compose "${COMPOSE_ARGS[@]}" config -q
# Build the archive helper before stopping any running writer.
docker compose "${COMPOSE_ARGS[@]}" build archive

VOLUME_RUNTIME="${PROJECT}_workbench-runtime"
VOLUME_POSTGRES="${PROJECT}_platform-db"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
runtime_archive="preupdate-${timestamp}-runtime.tar.gz.age"
postgres_archive="preupdate-${timestamp}-postgres.dump.age"

volume_exists() {
  docker volume inspect "$1" >/dev/null 2>&1
}

check_mount() {
  local service="$1" target="$2" expected="$3" id
  while IFS= read -r id; do
    [[ -z "$id" ]] && continue
    docker inspect "$id" | python3 -c 'import json,sys; c=json.load(sys.stdin)[0]; target,expected=sys.argv[1:]; mounts=[m for m in c["Mounts"] if m["Destination"]==target]; assert len(mounts)==1 and mounts[0].get("Type")=="volume" and mounts[0].get("Name")==expected, "existing data mount differs; inventory required before update"' "$target" "$expected" || die "existing $service data mount is not the expected preserved volume"
  done < <(docker ps -aq --filter "label=com.docker.compose.project=${PROJECT}" --filter "label=com.docker.compose.service=$service")
}
check_mount api /runtime "$VOLUME_RUNTIME"
check_mount postgres /var/lib/postgresql/data "$VOLUME_POSTGRES"

has_project_service_container() {
  local ids
  ids="$(docker ps -aq \
    --filter "label=com.docker.compose.project=${PROJECT}" \
    --filter "label=com.docker.compose.service=$1")"
  [[ -n "$ids" ]]
}

if ! volume_exists "$VOLUME_RUNTIME" && {
  has_project_service_container api || has_project_service_container worker ||
    has_project_service_container stream-worker
}; then
  die "existing app containers have no expected runtime volume; refusing update until the data mount is inventoried"
fi
if ! volume_exists "$VOLUME_POSTGRES" && has_project_service_container postgres; then
  die "an existing PostgreSQL container has no expected named database volume; refusing bootstrap until its data mount is inventoried"
fi

run_archive() {
  local name="$1"
  shift
  # The archive helper receives only encrypted output. No plaintext file is
  # written to the host or to the backup volume.
  "$@" | docker compose "${COMPOSE_ARGS[@]}" run --rm --no-deps -T archive \
    sh -eu -c 'exec age -r "$DTS_BACKUP_RECIPIENT" -o "/backups/$1.pending" -' sh "$name"
  docker compose "${COMPOSE_ARGS[@]}" run --rm --no-deps -T archive \
    sh -eu -c 'mv "/backups/$1.pending" "/backups/$1"' sh "$name"
}

pending_runtime=0
pending_postgres=0
stopped_writers=()
backup_complete=0
cleanup_before_update() {
  local status=$?
  if (( backup_complete == 0 )); then
    if (( pending_runtime )); then
      docker compose "${COMPOSE_ARGS[@]}" run --rm --no-deps -T archive \
        sh -eu -c 'rm -f "/backups/$1.pending"' sh "$runtime_archive" >/dev/null 2>&1 || true
    fi
    if (( pending_postgres )); then
      docker compose "${COMPOSE_ARGS[@]}" run --rm --no-deps -T archive \
        sh -eu -c 'rm -f "/backups/$1.pending"' sh "$postgres_archive" >/dev/null 2>&1 || true
    fi
    if ((${#stopped_writers[@]})); then
      docker start "${stopped_writers[@]}" >/dev/null || true
    fi
  fi
  return "$status"
}

# Stop app writers while capturing one consistent runtime archive. PostgreSQL
# remains online for pg_dump, which is transactionally consistent.
trap cleanup_before_update EXIT
for service in api worker stream-worker backup; do
  while IFS= read -r container_id; do
    [[ -n "$container_id" ]] && stopped_writers+=("$container_id")
  done < <(docker ps -q \
    --filter "label=com.docker.compose.project=${PROJECT}" \
    --filter "label=com.docker.compose.service=${service}")
done
if ((${#stopped_writers[@]})); then
  docker stop --time 30 "${stopped_writers[@]}" >/dev/null
fi

if volume_exists "$VOLUME_RUNTIME"; then
  pending_runtime=1
  echo "Creating encrypted pre-update archive of the existing runtime volume."
  run_archive "$runtime_archive" \
    docker compose "${COMPOSE_ARGS[@]}" run --rm --no-deps -T archive \
      sh -eu -c 'tar -czf - -C /runtime .'
  pending_runtime=0
else
  echo "No existing runtime volume found; no runtime archive was needed."
fi

if volume_exists "$VOLUME_POSTGRES"; then
  if ! docker compose "${COMPOSE_ARGS[@]}" ps -q postgres | grep -q .; then
    die "PostgreSQL volume exists but its service container is unavailable; refusing update without a database dump"
  fi
  pending_postgres=1
  echo "Creating encrypted pre-update PostgreSQL dump."
  run_archive "$postgres_archive" \
    docker compose "${COMPOSE_ARGS[@]}" exec -T postgres \
      pg_dump -U workbench -d workbench -Fc
  pending_postgres=0
else
  echo "No existing PostgreSQL volume found; first database bootstrap is allowed and runtime data remains preserved."
fi

backup_complete=1
trap - EXIT

# No down, volume removal, or optional platform flag: PostgreSQL, Redis, worker
# and encrypted backup are part of every CLOSED team deployment.
services=(web api postgres redis ml-compat worker backup)
# Preserve an already configured replay stack rather than leaving its stopped
# worker behind after the consistent backup.
if has_project_service_container stream-worker; then
  COMPOSE_ARGS+=(--profile stream)
  services+=(stream-worker stream-backend ingester clickhouse)
fi
docker compose "${COMPOSE_ARGS[@]}" up -d --build --wait "${services[@]}"
bash "$REPO_DIR/scripts/verify-team-deployment.sh" --verify
