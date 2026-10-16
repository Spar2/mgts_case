#!/usr/bin/env bash
# Read-only CLOSED team deployment checks and legacy-volume inventory.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

PROJECT=haklolarp-web
TEAM_ENV_FILE="${DTS_TEAM_ENV_FILE:-/etc/haklolarp-web/team.env}"
MODE="${1:---verify}"
case "$MODE" in
  --preflight|--inventory|--verify) ;;
  --help|-h)
    cat <<'USAGE'
Usage: bash scripts/verify-team-deployment.sh [--preflight|--inventory|--verify]

--preflight checks the private env file, required Compose settings, and backup
directory without changing services. --inventory lists persistent volumes and
legacy SQLite file paths/sizes. --verify checks service health, the public live
endpoint, and PostgreSQL/worker readiness. All modes are read-only.
USAGE
    exit 0
    ;;
  *) echo "verify-team-deployment: unknown mode" >&2; exit 2 ;;
esac

die() {
  echo "verify-team-deployment: $1" >&2
  exit 1
}

if [[ ! -f "$TEAM_ENV_FILE" || -L "$TEAM_ENV_FILE" ]]; then
  die "private team configuration file is absent or is not a regular file"
fi
if [[ "$(stat -c '%a' -- "$TEAM_ENV_FILE")" != 600 ]]; then
  die "private team configuration must have mode 0600"
fi
if [[ "$(stat -c '%u' -- "$TEAM_ENV_FILE")" != "$(id -u)" || ! -r "$TEAM_ENV_FILE" ]]; then
  die "private team configuration must be owned and readable by the deployment account"
fi
export DTS_BUILD_SHA="$(git rev-parse HEAD)"

COMPOSE_ARGS=(
  --env-file "$TEAM_ENV_FILE"
  -p "$PROJECT"
  -f compose.web.yaml
  -f compose.platform.yaml
  -f compose.team.yaml
)
docker compose "${COMPOSE_ARGS[@]}" config -q
docker compose "${COMPOSE_ARGS[@]}" config --format json |
  python3 -c 'import json,sys; c=json.load(sys.stdin); e=c["services"]["api"]["environment"]; host=c["services"]["web"]["environment"]["DTS_PUBLIC_HOST"]; origin=c["services"]["web"]["environment"]["DTS_PUBLIC_ORIGIN"]; ports=c["services"]["web"]["ports"]; ok=(e.get("DTS_ENV")=="production" and str(e.get("DTS_SECURE_COOKIES")).lower()=="true" and e.get("DTS_ALLOWED_HOSTS", "").split(",")[0]==host and e.get("DTS_ALLOWED_ORIGINS")==origin and origin=="https://"+host and e.get("FORWARDED_ALLOW_IPS")=="172.30.89.10" and len(ports)==1 and ports[0].get("host_ip")=="127.0.0.1" and str(ports[0].get("published"))=="8081" and "postgres" in c["services"] and "worker" in c["services"]); sys.exit(0 if ok else 1)' ||
  die "Compose must enforce Secure cookies, exact Host/HTTPS Origin, trusted nginx proxy, PostgreSQL, and worker"

BACKUP_SOURCE="$(
  docker compose "${COMPOSE_ARGS[@]}" config --format json |
    python3 -c 'import json,sys; c=json.load(sys.stdin); print(next(v["source"] for v in c["services"]["backup"]["volumes"] if v["target"]=="/backups"))'
)"
if [[ ! -d "$BACKUP_SOURCE" || -L "$BACKUP_SOURCE" ]]; then
  die "private backup directory must already exist and must not be a symlink"
fi
if [[ "$(stat -c '%a' -- "$BACKUP_SOURCE")" != 700 ]]; then
  die "private backup directory must have mode 0700"
fi
if [[ "$(stat -c '%u' -- "$BACKUP_SOURCE")" != 10001 ]]; then
  die "private backup directory must be owned by UID 10001 for the backup container"
fi

if [[ "$MODE" == "--preflight" ]]; then
  echo "Private team config and mandatory PostgreSQL Compose settings are valid."
  exit 0
fi

volume_names=(
  "${PROJECT}_workbench-runtime"
  "${PROJECT}_platform-db"
  "${PROJECT}_platform-redis"
)
if [[ "$MODE" == "--inventory" ]]; then
  echo "Persistent volume inventory:"
  for volume in "${volume_names[@]}"; do
    if docker volume inspect "$volume" >/dev/null 2>&1; then
      echo "  present: $volume"
    else
      echo "  absent: $volume"
    fi
  done
  runtime_volume="${PROJECT}_workbench-runtime"
  if docker volume inspect "$runtime_volume" >/dev/null 2>&1 &&
    [[ -n "$(docker compose "${COMPOSE_ARGS[@]}" ps -q api 2>/dev/null || true)" ]]; then
    echo "Legacy SQLite files under /runtime (relative path and byte size only):"
    docker compose "${COMPOSE_ARGS[@]}" exec -T api python -c \
      'from pathlib import Path; root=Path("/runtime"); files=sorted(p for p in root.rglob("*.sqlite3*") if p.is_file()); [print(f"{p.relative_to(root)} {p.stat().st_size}") for p in files]; print(f"count={len(files)}")'
  else
    echo "Runtime file inventory unavailable until an API container is running."
  fi
  exit 0
fi

for service in web api postgres redis ml-compat worker backup; do
  id="$(docker compose "${COMPOSE_ARGS[@]}" ps -q "$service")"
  [[ -n "$id" ]] || die "required service is not running: $service"
  state="$(docker inspect --format '{{.State.Status}}' "$id")"
  [[ "$state" == running ]] || die "required service is not running: $service"
done

for service in api postgres redis ml-compat; do
  id="$(docker compose "${COMPOSE_ARGS[@]}" ps -q "$service")"
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$id")"
  [[ "$health" == healthy ]] || die "required service health check is not healthy: $service"
done

PUBLIC_HOST="$(
  docker compose "${COMPOSE_ARGS[@]}" config --format json |
    python3 -c 'import json,sys; c=json.load(sys.stdin); print(c["services"]["api"]["environment"]["DTS_ALLOWED_HOSTS"].split(",")[0])'
)"
APP_PORT="$(
  docker compose "${COMPOSE_ARGS[@]}" config --format json |
    python3 -c 'import json,sys; c=json.load(sys.stdin); p=c["services"]["web"]["ports"][0]; print(p["published"])'
)"
curl --fail --silent --show-error --max-time 10 \
  -H "Host: $PUBLIC_HOST" -o /dev/null \
  "http://127.0.0.1:${APP_PORT}/api/v2/health/live"

curl --fail --silent --show-error --max-time 10 -H "Host: $PUBLIC_HOST" \
  "http://127.0.0.1:${APP_PORT}/api/v1/health" |
  python3 -c 'import json,os,sys; h=json.load(sys.stdin); assert h.get("auth_enabled") is True and h.get("access_mode")=="closed-team" and h.get("build_sha")==os.environ["DTS_BUILD_SHA"], "closed access mode or deployed SHA mismatch"'
for path in /api/v1/records /api/v1/queue /api/v2/investigations /api/v2/cases /api/public/v1/package; do
  status="$(curl --silent --show-error --max-time 10 -H "Host: $PUBLIC_HOST" -o /dev/null -w '%{http_code}' "http://127.0.0.1:${APP_PORT}${path}")"
  [[ "$status" == 401 ]] || die "anonymous data access is not closed: $path"
done
status="$(curl --silent --show-error --max-time 10 -H "Host: $PUBLIC_HOST" -o /dev/null -w '%{http_code}' "http://127.0.0.1:${APP_PORT}/works")"
[[ "$status" == 303 ]] || die "anonymous work page must redirect to login"

# Verify DB connectivity without migrations or writes. The public readiness
# endpoint requires a login, so check the compute heartbeat inside the API.
docker compose "${COMPOSE_ARGS[@]}" exec -T api python -c '
import os,time
from pathlib import Path
import psycopg
with psycopg.connect(os.environ["DTS_DATABASE_URL"], connect_timeout=5) as db:
    with db.transaction():
        db.execute("SET TRANSACTION READ ONLY")
        assert db.execute("SELECT 1").fetchone()[0] == 1
p=Path("/runtime/worker-compute.heartbeat")
age=time.time()-float(p.read_text())
assert 0 <= age <= 30, "compute worker heartbeat is stale"
'
echo "CLOSED team deployment is running; PostgreSQL and the compute worker are ready."
