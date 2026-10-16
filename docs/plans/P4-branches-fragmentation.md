# P4 — Branches rollout + canon fragmentation

Scope: after the web first-green (P3), roll out the remaining stands
(`preview-infra → ml → data → main`) and fragment the reusable single deploy
job into an observable staged chain — with zero renames, zero big-bang
secret rotation, and an additive-only canon until the last cutover. Builds on
P3 and all prior verdicts; serializer rulings are normative (§7).

Reference SHAs (stage baseline): `origin/devops` = `origin/main` =
`2eb7855`; `origin/web` = `f637d86`; `origin/ml` = `237168f`;
`origin/data` = `e34157c`.

Canon files: `.github/workflows/deploy-reusable.yml` (canon),
`deploy-web.yml` / `deploy-ml.yml` / `deploy-staging.yml` (+ data's
`deploy-data.yml` on `origin/data`), `preview.yml`, `notify-*.yml`,
`web-checks.yml` (+ `test-ml-pr.yml`, `lint-data.yml` on their branches),
`ops/deploy-{staging,web,ml}.sh` (+ data bootstrap via
`ansible/playbooks/data-stand.yml`), `compose.yaml` / `compose.web.yaml` /
`compose.ml.yaml` / `stack/compose.yaml`, `ansible/roles/{teams,app,docker}`,
`ansible/KEYS.md`.

## 1. Branch inventory (mapped)

| Branch / stand | Caller → canon | Dir / project | Mode | Principals / keys | Status at P3 |
|---|---|---|---|---|---|
| `devops` (flows underneath) | none (infra-only) | — (owns canon `@main`, ansible, preview root) | n/a | Admin keys (`xeon_staging`, `haklolarp_deploy`, `runner_admin`, `kvm_wake` per `KEYS.md`) | Base `2eb7855`; **zero-diff, no cutover of its own** |
| `preview-infra` (PR envs) | `preview.yml` (standalone, no canon) | `/opt/haklolarp-previews/pr-<N>` / `haklolarp-pr-<N>` | compose pull/up per-PR | Shared `STAGING_*`/`KVM_*` CI key + `DISCORD_WEBHOOK_DEVOPS` | Ships with canon base; rolls out FIRST (§2) |
| `ml` | `deploy-ml.yml` → `deploy-reusable.yml@main` | `/opt/haklolarp-ml` / `haklolarp-ml` | tmux (`ops/deploy-ml.sh`), `run_tests: true`, `poll_rounds: 80` | `haklo-ml` user + future per-team CI pubkey (`teams_ci_pubkeys`), shared secrets today | `ops/deploy-ml.sh` + `compose.ml.yaml` + `test-ml-pr.yml` exist on `origin/ml` |
| `data` | `deploy-data.yml` (wake + prepare + `deploy` → canon `mode: plain`) | `/opt/haklolarp-stack` / `haklolarp-stack` (adopts EXISTING stack, no new dir) | prepare (ansible `data-stand.yml`) then plain (`stack/compose.yaml`) | `ansible_user` + `STAGING_BECOME_PASSWORD` (sudo on Xeon); `app_data_enabled` gate | Full split + `lint-data.yml`, `notify-data.yml`, `app/tasks/data.yml` on `origin/data` |
| `main` | `deploy-staging.yml` → canon | `/opt/haklolarp-staging` / `haklolarp-staging` | plain (`compose.yaml`) | `ansible_user` + shared CI key; `inline_notify: false` (via `notify-runs.yml`) | Canon base; rolls out LAST among deploys (§2) |

`devops` never "cutsover": it is the pipe through which canon/ansible flow
(`uses: …/deploy-reusable.yml@main`, `site.yml` roles). Its changes land as
normal additive canon edits + ansible runs, verified by `pr-gate.yml` /
`ansible-lint.yml`.

## 2. Cutover order + DEP edges

Rollout order (strict): **preview-infra → ml → data → main**.

- **S-1 preview-infra first.** Lowest blast radius (ephemeral `pr-<N>` dirs,
  auto-cleanup on close), proves wake+tailnet+compose+notify plumbing and the
  `APP_PORT=9000+N` range before any team stand moves. DEP: none (needs only
  canon base + ansible `app_preview_root`).
- **S-2 ml second.** First tmux consumer after web; exercises `run_tests`
  (pytest) + long poll (80 rounds) + `compose.ml.yaml` on 8082. DEP: S-1
  (preview ports proven free) + P3 (tmux fire/poll/collect pattern green).
- **S-3 data third.** Highest state risk (adopts live `haklolarp-stack` with
  existing volumes; project name MUST stay `haklolarp-stack` or compose forks
  a second ClickHouse onto 8123/6379/8000). DEP: S-2 (tmux + pytest lanes
  stable) + ansible `data-stand.yml` dry-proven (`app_data_enabled=false`
  default keeps `site.yml` hands-off until the playbook flips it).
- **S-4 main last.** The `staging-deploy` (non-cancelling, queued) plain
  deploy of `/opt/haklolarp-staging`. Rolls only after all team lanes are
  green, so a main regression can't be blamed on unfinished branch infra.
  DEP: S-3.

Sunset tickets S-1..S-8 (one per scope; close only with its proof):

- S-1 preview-infra live (PR open → env on 9000+N → close → removed + both
  Discord notes). S-2 ml green (pytest + tmux + 8082 health). S-3 data
  adopted (prepare + plain, volumes intact, loopback only). S-4 main queued
  deploy green (no cancel-in-progress violation). S-5 per-branch secrets
  complete (every branch on its own prefix, old shared names revoked — §4).
  S-6 staged chain cut over on all callers (§5–§6). S-7 serializer order
  honored (skeleton @main → branch infra → cutover, §7). S-8 old single-job
  canon path removed (only after S-6 proven on every caller; never big-bang).

## 3. Per-branch principals / keys / dirs

- **Isolation model** (`ansible/roles/teams`): one system user + one dir per
  team — `haklo-web → /opt/haklolarp-web`, `haklo-ml → /opt/haklolarp-ml`
  (data reuses `/opt/haklolarp-stack`, no `haklo-data` user/dir — deliberate,
  recorded in `teams/defaults`). Dirs `0770` while the shared CI user remains
  in team groups (transition state); `0750` + drop the CI-user group task only
  after per-team pubkeys land in `teams_ci_pubkeys` (empty today = users/dirs
  created, keys skipped with warning — stub mode, honest and logged).
- **Deploy key** (`roles/app`): read-only `haklolarp_deploy` (Xeon→GitHub),
  laid by the role from the admin machine only; CI never writes it. Rotation =
  new key → GitHub Deploy Keys → role re-run → age bundle (`backup-keys.sh`).
- **KVM wake key**: forced command `/usr/local/bin/xeon-wake-only` — wake-only
  by construction; reset/ISO unavailable even on leak; no power-off/reset from
  CI ever.
- **Data extra**: `STAGING_BECOME_PASSWORD` (sudo for `data-stand.yml`) lives
  only in the data workflow env → ansible extra-vars file, shredded after the
  run; never logged (`no_log` on key tasks).

## 4. Port / secret / concurrency conflicts (and resolutions)

- **Ports (frozen; NOT all loopback — honest).** main `8080` (`compose.yaml`
  `${APP_PORT:-8080}:80` — implicit 0.0.0.0 wildcard open-bind, reachable via tailnet/LAN), web `8081` (`compose.web.yaml`
  `127.0.0.1:${APP_PORT:-8081}` — loopback-only, tailnet curl NOT WORKING till rebind), ml `8082` (`compose.ml.yaml`
  `127.0.0.1:${APP_PORT:-8082}` — loopback-only, tailnet curl NOT WORKING till rebind), preview `9000+N` (via wildcard `compose.yaml`, range 9000-9100 staging/preview), data loopback stack ports
  (8123/6379/8000 inside `haklolarp-stack` — MUST stay loopback; data never
  publishes host-wide). Pre-flight every dispatch: `ss -ltnp` proves no
  cross-stand collision; preview formula can't collide with 808x for N≥1
  (9001+).
- **Secrets: per-branch split, NO big-bang (S-8 explicitly not a flag day).**
  Today all callers share `STAGING_*`/`KVM_*`/`TS_OAUTH_*` + one Discord
  webhook each (`DISCORD_WEBHOOK_{WEB,ML,DEVOPS,DATA}`). Migration path
  follows rollout order §2: `PREVIEW_*` → `ML_*` → `DATA_*` → `MAIN_*` (`WEB_*` done in P3), one prefix per stage,
  each with dual-read grace (caller accepts new prefix, falls back to shared
  name) then per-branch revocation of the old mapping. Revoking everything at
  once (big-bang S-8) is forbidden — a single typo would brick all stands
  simultaneously with no working caller to repair them.
- **Concurrency (frozen names).** `staging-deploy` (main,
  `cancel-in-progress: false` — queued, never cancels),
  `staging-deploy-web` / `-ml` / `-data` (`true` — latest wins),
  `preview-<PR#>` (`true`). `notify-runs.yml` keys on workflow names
  (`Deploy staging (main -> Xeon)`, `Deploy web/ml (staging)`, …) — renaming
  any workflow or group silently drops run notifications.

## 5. Per-mode specifics

- **ml (pytest + 80 polls).** `run_tests: true` runs `requirements-dev.txt` +
  `pytest tests -q` on the github-hosted runner BEFORE any SSH (fail fast,
  no wake cost). `poll_rounds: 80` (`80 × 15 s = 20 min` cliff) covers ML
  image build + model warmup; `ops/deploy-ml.sh` health-polls
  `127.0.0.1:8082/health` 30×5 s post-up and fails the tmux session (nonzero
  → `.done` code → collect fails the run) rather than declaring victory on
  `up -d` alone.
- **data (prepare/deploy split).** `wake` (probe/skip-wake, copy of canon
  logic — needed because the canon wake is fused inside `deploy-reusable`
  and data must bootstrap BEFORE calling it) → `prepare` (ansible
  `data-stand.yml`: adopts `/opt/haklolarp-stack`, writes `stack/.env` with
  the frozen project name, `app_data_enabled=true` scoped to that playbook
  only) → `deploy` (canon `mode: plain`, `compose_file: stack/compose.yaml`,
  `inline_notify: true`, color `0x16a085`). `prepare` requires
  `STAGING_BECOME_PASSWORD` or fails closed. Plain mode = no tmux, no poll
  loop — health is the stack's own readiness, not a `.done` marker.
- **preview guards.** Per-PR dir/project/port derived SOLELY from
  `github.event.pull_request.number` (no caller input, no collisions);
  `concurrency: preview-<N>` serializes re-runs; `closed` action tears down
  (`down --rmi local --volumes` scoped to `-p haklolarp-pr-<N>` — the ONLY
  sanctioned `--volumes` in the fleet, safe because the project name is
  PR-unique) then `rm -rf` the dir. Root `/opt/haklolarp-previews` owned by
  ansible; subdirs by CI. Compose files MUST publish via `${APP_PORT}` or the
  preview silently binds the default port and collides.

## 6. Staged-jobs design (fragmentation of the canon)

Goal: split the reusable single `deploy` job (fire+poll+collect+notify fused
in one runner) into a linear observable chain while keeping every caller thin.

**DAG (linear, `needs`-chained):**
`wake → test → sync → launch → await → verify → report`.
Each stage is a separate job in `deploy-reusable.yml`; callers gain ZERO new
jobs (they still declare one `deploy:` using the canon).

**Contracts (jobs communicate ONLY via declared outputs + on-Xeon markers):**

- `wake` → out `xeon_up: bool`. `test` (runs iff `run_tests`, else skipped-green)
  → out `tests_ok`. `sync` (checkout ref + record SHA) → out `git_sha`.
  `launch` (tmux new-session, or compose up for plain — see serializer note)
  → out `session` (+ `run_attempt`, §7) and wipes stale
  `.deploy-<branch>-*` markers first. `await` (bounded poll loop) → out
  `done_code`, `last_stage` and NEVER exits nonzero itself: timeout is recorded
  as `done_code=TIMED_OUT`, not as a runner failure (so `verify` always runs
  and can dump diagnostics). `verify` (health curls via `health_path/port`
  inputs + `docker ps`) → out `health_json`. `report` (always-run) → single
  Discord digest `FAILED-at-<stage>` (or success) + `$GITHUB_STEP_SUMMARY`
  table + artifacts.
- On-Xeon markers stay the inter-step transport across SSH invocations:
  `$DIR/.deploy-$SESS.done` (exit code), pane dumps, health JSON sidecar.
  `launch` wipes stale markers; `await` only reads; `verify`/`report` collect.

**Visibility (per-stage observability, failure-only noise):** `report` sends
the ONE Discord message per run (title carries `FAILED-at-<stage>` for instant
triage); every job appends its own section to `$GITHUB_STEP_SUMMARY` (SHA,
session, timings, tail of logs); artifacts uploaded with
`if-no-files-found: warn` — tmux pane dumps (14 d retention) + health JSON
and compose config (30 d). No per-stage Discord spam (P5 single-owner rule).

**Failure semantics:** any stage failure short-circuits downstream `needs`
except `report` (`if: always()`); `await` timeout → `TIMED_OUT` code flows to
`verify` (which attempts health anyway, records `health_json.unreached`) and
`report` marks `FAILED-at-await`. Exit codes from `.done` propagate unchanged
through `await.done_code` so the run conclusion equals the deploy script's
verdict, not the poller's.

**Retry:** stage-level `retry` is NOT automatic — re-dispatch the caller
(manual `workflow_dispatch` for web; push-retry for others). `launch` is
idempotent (kills same-prefix stale sessions, recreates `$SESS`); `sync` is
`--ff-only` (divergence fails loudly instead of merging). `run_attempt`
(output, starts at 1, bumps on caller re-run linkage) disambiguates sessions
(`$SESS` includes it) so a retry never collides with a zombie session.

**Timeouts:** `wake` 15 min (KVM boot wait ≤ 12 min), `test` 15 min,
`sync` 5 min, `launch` 5 min, `await` = `poll_rounds × 15 s + 5 min slack`
(web 40→15 min cap, ml 80→25 min cap), `verify` 10 min, `report` 5 min.
Overall ≤ current 45 min canon budget per branch mode.

**New optional inputs (all defaulted → backward compatible):**
`enable_staged` (`boolean`, default `false` — old fused path until cutover),
`stage_notify` (digest webhook override, default empty = caller's
`discord_webhook`), `health_path` (default `/api/v1/health`),
`health_port` (default per-branch: main 8080 / web 8081 / ml 8082),
`compose_project` (default empty = derived from `staging_dir`). Old callers
pass nothing new and behave exactly as today.

## 7. Thin-caller compat + migration (shadow → cutover)

- **Callers stay thin and frozen:** names (`Deploy * (staging)`), groups
  (`staging-deploy*`), gates (`needs: checks` + pilot flag on web; pytest gate
  on ml; become-password gate on data) never change. Migration adds at most one
  passthrough input (`enable_staged: true`) per caller at cutover — no job
  surgery in callers, ever.
- **Shadow-then-cutover, order web → ml → main → data → preview:**
  (1) ship staged chain behind `enable_staged: false` (default OFF — zero
  behavior change); (2) shadow: dispatch with `enable_staged: true` alongside
  the fused path on `web`, diff digests/summaries; (3) cut `web`, then `ml`
  (long-poll proves `await` caps), then `main` (queued plain proves
  serializer plain handling), then `data` (prepare→staged-plain proves split
  interop), then `preview` (highest fan-out, last). Each cutover keeps the old
  path one release as fallback (flag flip back, no code revert).
- **Serializer order (normative rulings):**
  1. `enable_staged` skeleton (default OFF) lands **@main FIRST**, before any
     branch infra — so all `uses: …@main` callers see a no-op update.
  2. Then per-branch infra (health endpoints, marker hygiene, compose
     `${APP_PORT}` compliance).
  3. Then cutover per the order above. Reordering (branch infra before the
     OFF-skeleton, or cutover before infra) is forbidden — it strands callers
     on unknown inputs.
  4. `SESS` MUST include `run_attempt`, and `launch` wipes stale markers
     (prevents zombie-session false greens across retries).
  5. Staged chain is **tmux-only** for fire/poll/collect semantics; plain-mode
     callers (`main`, `data`) traverse the same DAG but `launch`/`await`
     collapse to compose-up + health-wait (no `.done` markers), keeping
     `report`'s contract identical.
  6. Notify stays single-owner (P5): one digest per run from `report`,
     branch webhook preserved; `stage_notify` only reroutes, never duplicates.

## 8. Good practices (carried forward)

Least-change promotion (ff-only, detached-prev-SHA rollback) · manual dispatch
with pilot flag (no auto-deploy surprises) · evidence package per run (7 proofs,
§P3.4) · additive-only canon until the last cutover (flag-gated, old path
intact) · per-stage observability (digest + summary + artifacts, failure-only
noise) · instant rollback (one-line flag flip / prev-SHA redeploy, never
force-push) · frozen public names (workflows, groups, dirs, projects, ports).
