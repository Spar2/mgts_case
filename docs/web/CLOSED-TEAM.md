# CLOSED team deployment

This deployment serves the authenticated team application through the host's
existing Tailscale HTTPS reverse proxy. Tailscale terminates TLS; the web
container listens only on the host loopback address and forwards to the API on
the private Compose network. The team Compose stack always includes
PostgreSQL, Redis, the compute worker, and encrypted backups. There is no
deployment switch that omits the platform.

## Operator configuration

Use Linux with Docker Compose v2, `age`, `curl`, `python3`, and `git`. Install
the private environment file as `/etc/haklolarp-web/team.env`, owned by the deployment account
with mode `0600`. The scripts accept `DTS_TEAM_ENV_FILE` when the site uses a
different fixed path. Never commit or print this file.

It must contain actual values for:

| Variable | Operator value |
| --- | --- |
| `DTS_PUBLIC_HOST` | Exact Tailscale DNS hostname, without scheme or path |
| `DTS_PUBLIC_ORIGIN` | Exact `https://` origin matching that hostname |
| `DTS_POSTGRES_PASSWORD` | Privately generated PostgreSQL password |
| `DTS_RECORDS` | Existing prepared records directory |
| `DTS_QUEUE` | Existing queue directory |
| `DTS_RESEARCH_PACK` | Optional: existing real research package directory; omission disables this package |
| `DTS_IMPORT_ROOT` | Optional: existing import directory; defaults to `.runtime/imports` |
| `DTS_BACKUP_DIRECTORY` | Existing persistent directory for encrypted backup files |
| `DTS_BACKUP_RECIPIENT` | Public `age` recipient; keep the private identity offline |

Create the backup directory before deployment on persistent storage. Set its
owner to UID `10001` and mode `0700` so the non-root backup container can write
to it. The operator must confirm this storage is included in the host's
off-machine backup plan. The server needs only the public age recipient.

The default data volumes are `haklolarp-web_workbench-runtime`,
`haklolarp-web_platform-db`, and `haklolarp-web_platform-redis`. Keep the
Compose project name `haklolarp-web`; changing it creates another set of
volumes and can make existing data appear absent. Never use `down -v`, remove
these volumes, or replace the runtime mount as part of an ordinary update.
Input datasets are mounted read-only from their configured host directories.

## Check and deploy

Run read-only checks and inspect the existing volume/runtime inventory before
an operator-authorized deployment:

```sh
bash scripts/verify-team-deployment.sh --preflight
bash scripts/verify-team-deployment.sh --inventory
```

The inventory lists SQLite file paths and sizes only. It does not inspect
authors or assign identities. The existing Tailscale Serve target must send
traffic to `127.0.0.1:8081`; only this loopback port is published by the team
stack.

The deployment script defaults to applying the update; `--check` is the
explicit non-mutating mode:

```sh
bash ops/deploy-web.sh --check
bash ops/deploy-web.sh deploy-team-operator
```

Apply mode requires the private environment file and valid Compose
configuration before touching services. If a PostgreSQL volume exists, the
script stops application writers and the daily backup worker, writes a
custom-format `pg_dump`, encrypts that stream directly with age, then archives
the complete runtime volume read-only and encrypts that stream. PostgreSQL
stays online for the consistent `pg_dump`. No plaintext backup is written to
disk. Both backups must complete before update; on failure the old writers are
restarted and deployment aborts. On first database bootstrap, an absent
PostgreSQL volume is reported and allowed, while any existing runtime volume
is still archived and retained.

After backup, the script builds and starts web, API, PostgreSQL, Redis, the ML
compatibility service, compute worker, and encrypted daily backup service.
PostgreSQL migrations run transactionally on the existing database volume
when the API starts. The deployment never removes volumes.

The read-only verifier checks required services and health checks, requests
the public live-health route, runs `SELECT 1` in a read-only transaction
inside the API container, and verifies the compute-worker heartbeat. The
readiness endpoint requires a logged-in session and is not probed anonymously.
It additionally checks the exact deployed Git SHA, closed-team access mode,
HTTP 401 on anonymous data APIs and HTTP 303 on the work page. These checks do
not replace a clean-browser test through the external HTTPS address.

## Existing SQLite state and identity

The runtime volume may contain older SQLite workspaces and review journals.
Inventory them before the first database-backed rollout, retain the encrypted
pre-update archive, and agree on any migration with the data owner. SQLite
migration is an explicit operator action; it is not run automatically by the
deployment script:

```sh
docker compose --env-file /etc/haklolarp-web/team.env -p haklolarp-web \
  -f compose.web.yaml -f compose.platform.yaml -f compose.team.yaml \
  exec api python -m workbench.maintenance migrate-sqlite /runtime
```

The migration archives the original SQLite bytes and marks identities as
unassigned. It does not convert self-declared authors into authenticated
experts or administrators. An authorized human must decide any future mapping;
the operator must not infer it from a name, reviewer field, or historical
entry. Preserve the original volume and encrypted pre-update archive until the
owner has accepted the migrated state.

## Accounts and backups

After PostgreSQL and the worker are healthy, create or recover the first
administrator through the interactive operator command in
[PRODUCTION-OPERATOR.md](PRODUCTION-OPERATOR.md). Enter the password
interactively; do not put it in an environment variable or command line.
Invite other team members through the administrator interface and deliver
one-time links through a verified private channel.

The root `/` is a public product landing with a login link and invitation
instructions. It does not request or display workspace data and is available
even while PostgreSQL is unavailable. No public data API is reopened.
The closed mode has no anonymous demo or self-registration: `/try`, `/demo`,
the archive, reports, bundles and all work APIs require a valid server session.
The login, invitation-acceptance and recovery pages remain accessible so a
participant can obtain access. Only required scripts, styles, fonts and images
are served publicly; do not place data, credentials or configuration in the
frontend build. Sessions are stored as hashes in PostgreSQL. A login rotates
the previous session in the same browser; logout and account revocation stop
open data streams. Independent sessions on other devices remain revocable
through the account page.

Do not set `DTS_TEST_ALLOW_ANONYMOUS` on a shared server. It is a legacy
fixture option accepted only alongside `DTS_ENV=test`; production cannot use
it to bypass missing PostgreSQL. A database or authentication-service outage
must result in a controlled refusal, never in an anonymous workspace.

The daily age-encrypted backup service preserves platform tables and
referenced artifacts, retaining seven daily archives in the configured
directory. It does not replace the pre-update `pg_dump` or full runtime-volume
archive and does not replace off-server retention. Keep the age identity
offline and test restoration in an isolated environment. The deploy script
does not claim an existing database was backed up unless the encrypted
`pg_dump` completed.

The pre-update native PostgreSQL dump contains the database as it stood,
including session and one-time-link tables. Restore it only into an isolated
empty database. Before exposing a restored database, invalidate all sessions,
invitations and password-reset links using the existing maintenance procedure;
do not treat a native `pg_restore` as the application's sanitized restore.

## Network and application settings

The team nginx template is HTTP-only because TLS ends at the host's existing
Tailscale reverse proxy. It always overwrites `X-Forwarded-Proto` with
`https`; requests cannot choose their own scheme header. It passes the exact
Host through to FastAPI and appends the forwarded client chain. Uvicorn trusts
forwarded headers only from the fixed Compose nginx address `172.30.89.10`.
The API enforces production mode, Secure cookies, the configured Host allowlist
and exact HTTPS Origin. PostgreSQL, Redis, and ML compatibility ports are not
published on the host. The fixed `172.30.89.0/24` Compose subnet and nginx
address must not overlap with another Docker network on the server.

The operator must confirm the Tailscale Serve target and access ACLs before
the first deployment. The deployment helper does not make network, DNS,
Tailscale Serve, ACL, or cloud changes.
