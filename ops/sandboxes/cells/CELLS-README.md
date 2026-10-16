# Branch workload cells (FIGHTER-B)

Sandbox-only, additive confinement wrappers around the REAL per-branch workloads.
Read with `docker-compose.cells.yml` + `cell-proof.sh` (same directory).

## 0. Per-cell table

| Cell service(s) | Branch | Source dir / build | Published ports (host:container) | Mode | Health endpoint |
|---|---|---|---|---|---|
| `cell-web` | `web` | `apps/web/Dockerfile` (node build → `nginx:1.28-alpine` ⚠️ TODO-pin, unpinned upstream) | `127.0.0.1:${CELL_WEB_PORT:-8081}:80` ONLY | nginx UI, `depends_on` api healthy | `GET http://127.0.0.1:8081/api/v1/health` (via nginx) |
| `cell-web-api` | `web` | `services/web-api/Dockerfile` (`python@sha256:78387…` pinned) | none (upstream parity) | uvicorn `:8091`, `:ro` data mounts | container-local `GET http://127.0.0.1:8091/api/v1/health` (via `docker exec`) |
| `cell-ml` | `ml` | `Dockerfile.ml` (`python:3.12-slim` ⚠️ TODO-pin, unpinned upstream) | `127.0.0.1:${CELL_ML_PORT:-8082}:8082` (rebound from upstream open bind — §B1) | uvicorn `:8082` | `GET http://127.0.0.1:8082/health` (+ `/methods`, `POST /denoise/profile`, `POST /detect/field`) |
| `cell-data-migrate` | `data` | pinned `clickhouse` image, client only (`profiles:["migrate"]`, `restart:"no"`) | none | one-shot: applies `stack/clickhouse/init.sql` via `--queries-file` | exited `0` receipt (`docker ps -a`) + `≥3` tables in `haklolarp` DB |
| `cell-data-redis` | `data` | `redis@sha256:858f…` (pinned, `redis:7-alpine`) | `127.0.0.1:${CELL_REDIS_PORT:-6379}:6379` | `--appendonly yes --maxmemory 4gb --maxmemory-policy noeviction` | `redis-cli ping` |
| `cell-data-clickhouse` | `data` | `clickhouse/clickhouse-server@sha256:82499…` (pinned) | `127.0.0.1:${CELL_CH_HTTP_PORT:-8123}:8123` | server + `:ro` init.sql | `clickhouse-client --query "SELECT 1"` |
| `cell-data-backend` | `data` | `stack/backend/Dockerfile` (`python@sha256:02108…` pinned) | `127.0.0.1:${CELL_DATA_API_PORT:-8000}:8000` | `init:true`, `pull_policy: build` (upstream parity) | `GET http://127.0.0.1:8000/healthz` |
| `cell-data-ingester` | `data` | `stack/ingester/Dockerfile` (same pinned python) | none (upstream parity) | `init:true`, heartbeat `/tmp/ingester.heartbeat` (tmpfs) | heartbeat mtime < 90 s |

Upstream DNS names are kept VERBATIM (`api:8091` baked in `nginx.conf`,
`REDIS_HOST=redis`, `CH_URL=http://clickhouse:8123`) and served via
`cells-back` network aliases (`api`, `redis`, `clickhouse`) — zero image drift.
The one intentional DNS change: `cell-web-api` reaches ML at
`http://cell-ml:8082` (compose-DNS on the shared `cells-front` net,
env-overridable via `CELL_ML_URL`) instead of `host.docker.internal` —
this breaks the circular host-gateway dependency and lets ML bind loopback
(§B1 in the compose file; re-adding `extra_hosts: host-gateway` FAILs proof §S).

`cell-data-migrate` is INCLUDED (profile-gated one-shot, `restart:"no"`):
run it once WITHOUT `--rm` BEFORE/ON first data-cell start so the exited-0
container persists as the migrate receipt —
`docker compose -f ops/sandboxes/cells/docker-compose.cells.yml --profile migrate up cell-data-migrate`.
Excluded BY DESIGN (not a long-running workload): data `streamer`
(`profiles:[manual]` batch loader).

## 1b. Source-root allowlist (`CELL_*_SRC` env-mount smuggling guard)

`CELL_WEB_SRC` / `CELL_ML_SRC` / `CELL_DATA_SRC` / `CELL_WEB_*_SRC` keep the
`${VAR:-default}` env-overridable shape, but proof §C FAILs any running cell
whose bind/mount Source is not allowlisted. Allowed roots:

- (a) the file default — repo root (`../../..`) or a bare named volume
  (`cell-web-records`, …);
- (b) absolute paths under `/opt/haklolarp-*`, `/opt/cells-*`,
  `/srv/haklolarp-*`, `/srv/cells-*`, `/home/*` (per-branch checkouts on Xeon);
- (c) docker's own volume dir rendering (`/var/lib/docker/volumes/*`).

NEVER valid: `/`, `/etc`, `/root`, `docker.sock`, `/proc`, `/sys`,
`*staging*`, `*preview*`. A legit checkout outside (b) FAILs loudly — do NOT
work around it with env tricks; extend the allowlist in `cell-proof.sh` by
commit instead.

## 1. Users (non-root) + documented exclusions

| Service | `user:` | Basis |
|---|---|---|
| `cell-web` | `101` (nginx) + `cap_add: NET_BIND_SERVICE` (minimal add, ONLY service with one — uid 101 must bind :80) | image supports; tmpfs covers `/var/cache/nginx /var/run /tmp /var/log/nginx` under `read_only:true` |
| `cell-web-api` | `10001` | matches image `USER workbench` |
| `cell-ml` | `65532` (nobody) — DEVIATION: upstream `Dockerfile.ml` runs ROOT | read-only inference; verifier must confirm; fallback: delete `user:` |
| `cell-data-redis` | EXCLUSION (root) — entrypoint/volume init expects root | mitigated: `cap_drop ALL` + `no-new-privileges` + `read_only:true` |
| `cell-data-clickhouse` | EXCLUSION (root) — entrypoint init expects root | same mitigations; `/var/log/clickhouse` on tmpfs |
| `cell-data-backend` | `65532` (nobody) — DEVIATION: upstream runs ROOT | app only reads FS (`open(path,"r")` only); `PYTHONDONTWRITEBYTECODE=1` replaces the CONTRACTS `__pycache__` reason for writable rootfs |
| `cell-data-ingester` | `65532` (nobody) — DEVIATION: upstream runs ROOT | only FS write is the heartbeat → tmpfs `/tmp` |
| `cell-data-migrate` | EXCLUSION (root) — same image/entrypoint basis as clickhouse | same mitigations; one-shot, no ports |

(`postgres:999` from the boilerplate user list is N/A — no postgres in any stack.)

Fallback knobs if a service crash-loops on Xeon (in order): 1) drop `read_only:true`
for that service, 2) drop `user:`, 3) report back — NEVER relax `cap_drop`,
`no-new-privileges`, loopback binds, or add mounts.

## 2. DevOps control plane — READ-ONLY rules for cells

Cells are OBSERVED, never operated, by anyone except an explicit promotion change.
Allowed (read-only): `docker compose … config` (render check), `docker ps`,
`docker inspect`, `docker logs`, `docker stats --no-stream`, `ss -ltn`,
`bash ops/sandboxes/cells/cell-proof.sh` (assert-only script).
Forbidden against cells AND host: `up/down/restart/rm/kill`,
`docker exec` with writes (reads for health/negative tests only),
`--privileged`, `-v /:/…`, ANY mount of `/etc`, `/root`, `/`, `docker.sock`,
`git push`, `git merge`, `workflow_dispatch` runs, Xeon SSH sessions beyond
running the proof script, and starting/stopping ANY container (including cells).

## 3. MAIN EXCLUSION NOTICE — REAL DEPLOY TARGET, DO NOT TOUCH

`main` has NO cell. The live `staging-web-1` container on `0.0.0.0:8080`
(Up 19 h at scope freeze) is the REAL staging deploy. Cells must NEVER bind
`:8080`, NEVER use `container_name` (avoids collision), and are always started
with an explicit `-p <cell-project>` name. `cell-proof.sh` asserts
`staging-web-1` is still `Up` as a guard (§A).

## 4. PREVIEW EXCLUSION NOTICE

Preview (`APP_PORT=9000+N`, `preview.yml`) is EXCLUDED from cells scope: no
preview service, no `9000+` ports, no preview env in this file. Preview stays on
its existing workflow untouched.

## 5. Promotion / integration notes (NO changes made — cells are additive)

- This directory adds 3 files and modifies NOTHING: `deploy-web.yml`,
  `ops/deploy-*.sh`, existing compose files, and CI are byte-identical.
- To promote later (devops decision, not this change): copy the confinement
  stanza pattern (`cap_drop`, `no-new-privileges`, `read_only`, limits,
  loopback binds) into branch compose files; adopt `cell-proof.sh` asserts as a
  post-deploy gate; pin `nginx:1.28-alpine` + `python:3.12-slim` digests and
  propagate here (clearing the two TODO-pins).
- Proof gates (do NOT weaken): `cell-proof.sh` has static YAML asserts (§S:
  users/ports/`internal:true`/no-main/no-`container_name`/migrate-gate present),
  a MIN_PASS floor (hollow all-SKIP runs exit nonzero), a strengthened breakout
  probe (`/proc/1/root/etc/shadow`), and the migrate receipt gate (§E: backend
  `/healthz` is unconditionally 200, so health-only on empty volumes FAILs).
- Source-root contract: `CELL_WEB_SRC` / `CELL_ML_SRC` / `CELL_DATA_SRC`
  (default: repo root of the checkout containing this file; allowlist §1b).
  On Xeon either merge
  this directory into each branch or point the vars at per-branch checkouts,
  e.g. `CELL_DATA_SRC=/opt/haklolarp-stack`. Run the migrate one-shot first
  (`--profile migrate up cell-data-migrate`, no `--rm`); seed web `:ro` mounts via `CELL_WEB_*_SRC` binds or pre-populated named
  volumes (`cell-web-records/-queue/-research`, `cell-web-runtime`).

## 6. Open risks

1. **ml loopback rebind (RESOLVED in-file, needs Xeon confirmation).**
   `cell-ml` now binds `127.0.0.1:${CELL_ML_PORT:-8082}:8082` and
   `cell-web-api` dials it over compose DNS (`http://cell-ml:8082` on shared
   `cells-front`), breaking the old circular host-gateway dependency. The
   verifier must confirm api→ML inference end-to-end on Xeon; if it fails,
   report back — NEVER fix by re-adding `host-gateway`/`0.0.0.0` (proof §S
   FAILs both).
2. **`internal:true` vs published-port ingress (MITIGATED in-file).**
   `redis`/`clickhouse` are now dual-attached (`cells-front` + `cells-back`)
   so loopback ingress no longer depends on the userland-proxy/internal-net
   path; binds stay loopback-only (no exposure change). If `ss` still shows no
   `127.0.0.1:6379/8123` listener on Xeon, report the Docker version — do not
   move them off `cells-back` (east-west aliases live there).
3. **Non-root + `read_only` without local docker.** No container was started
   locally (forbidden); `docker compose config` + live proofs are deferred to
   the Xeon verifier. §1 fallbacks cover the likely failure modes
   (nginx cache/log dirs, python bytecode, CH log dir).
4. **Named-volume defaults start EMPTY.** Fresh volumes mean empty web data dirs
   and empty redis/CH until seeded — healthchecks pass, but functional smoke
   needs seeding (§5). This is intentional (never auto-create host dirs).
