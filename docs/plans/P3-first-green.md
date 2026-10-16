# P3 — First Green: web pilot landing strategy

Scope: land branch `web` on the Xeon staging stand (`/opt/haklolarp-web`) as the
first end-to-end green, without touching `main`, without ungating anything, and
without changing any public workflow/concurrency names. Builds on prior
verdicts; this doc is the executable stage plan.

Reference SHAs (frozen for this stage):

- `origin/web` = `f637d86` ("fix(web): keep analysis slices…", first-green
  candidate tip).
- Local `web` checkout at stage start = `a5362f9` (20 commits behind tip).
- `origin/devops` = `origin/main` = `2eb7855` (canon base; deploy-reusable,
  callers, preview, notify-runs, ansible roles).

## 1. Landing strategy (least-change promotion)

1. **Fast-forward only.** On the Xeon stand (and in any local mirror):
   `git fetch origin && git checkout web && git pull --ff-only origin web`,
   i.e. local `a5362f9 → f637d86` fast-forward. No merge commits, no rebase of
   published history, no squash. If `--ff-only` fails, STOP: the branch has
   diverged and needs a human decision — never force-push, never reset-shared.
2. **KEEP the gate.** The gated `deploy-web.yml` (as of `f637d86`) stays as-is:
   `checks:` job (`uses: ./.github/workflows/web-checks.yml`) is mandatory,
   `deploy:` has `needs: checks` plus
   `if: github.event_name == 'workflow_dispatch' && inputs.deploy_local_pilot`.
   Consequences: pushes to `web` run checks only; promotion to the stand
   happens **exclusively** via manual `workflow_dispatch` with
   `deploy_local_pilot: true`. No auto-deploy on push.
3. **Dispatch promotion (manual, pilot-flagged).** No branch merge, no caller
   edit, no canon edit is needed to promote. Promotion = one manual dispatch
   (see §3). The `deploy_local_pilot` boolean is the only promotion switch;
   platform/stream flags (`DTS_PLATFORM_ENABLED`, `DTS_STREAM_ENABLED`) stay
   operator-managed env on the stand, default OFF.
4. **Canon untouched.** `deploy-reusable.yml@main` (wake → tmux
   fire/poll/collect → discord) is consumed by ref, not edited. `web-checks.yml`
   is consumed as a local reusable (`./.github/workflows/web-checks.yml`) from
   the `web` ref at dispatch time.

## 2. Pre-dispatch checklist (all must be true; any NO = stop)

Run in this order; record evidence (command + output) for the run package.

- [ ] **Tree state.** `git status --porcelain` clean on the dispatching
      checkout; `git rev-parse HEAD` == `f637d86`; `git rev-parse
      origin/web` == `f637d86` after `git fetch origin web`.
- [ ] **Checks green on candidate.** `web-checks.yml`
      (`contracts-and-build`) green on `f637d86` (PR/branch run). Do not
      dispatch on a red or stale (pre-`f637d86`) checks run.
- [ ] **Secrets present (8).** `gh secret list` shows all eight names the
      caller/canon need: `TS_OAUTH_CLIENT_ID`, `TS_OAUTH_SECRET`,
      `STAGING_HOST`, `KVM_HOST`, `STAGING_USER`, `STAGING_SSH_KEY`,
      `KVM_WAKE_KEY`, `DISCORD_WEBHOOK_WEB`. Values are never printed —
      presence only. Missing → STOP, no dispatch.
- [ ] **Stand provision decision — `.runtime`.** Decide explicitly and log it:
      (a) **Queue-crash expected** (no `.runtime/records|queue|runtime` seeded
      → health may flap / journeys fail), or (b) **green-but-empty** accepted
      (pilot proves plumbing, not data). Either is a valid pilot outcome, but
      it must be declared BEFORE dispatch so the verdict is not misread.
- [ ] **`APP_PORT` unset (default path).** Pilot runs on the default
      `compose.web.yaml` mapping `127.0.0.1:${APP_PORT:-8081}:80`. Confirm no
      stray `APP_PORT` export on the stand that would move the pilot off 8081
      (preview ports `9000+N` are a separate range and must not collide).
- [ ] **Disk/RAM headroom.** On Xeon: `df -h /opt /var/lib/docker` and
      `free -h` show room for `--build` (web + api images) plus volume
      `workbench-runtime`. Near-full disk is the classic `--build` killer —
      abort and prune only per standing policy (never `down -v` on data
      volumes; see prohibitions).
- [ ] **`ss` pre-check.** `ss -ltnp | grep -E '8080|8081|8082|9000'` on Xeon:
      8081 must be free (or owned only by a previous pilot instance this run
      will replace); 8080 (main), 8082 (ml), 9000+N (previews) must be
      unaffected. Any conflict → resolve by stopping the stale pilot project
      only (`-p haklolarp-web`), never by touching other projects.

## 3. Dispatch procedure + 10-minute poll-cliff math

Exact dispatch (from any host with `gh` auth to `werd0y45/haklolarp`):

```bash
git fetch origin web
# sanity: expect f637d86
git rev-parse origin/web
gh workflow run "Deploy web (staging)" --ref web -f deploy_local_pilot=true
```

Then watch the run:

```bash
gh run list --workflow "Deploy web (staging)" --branch web --limit 5
gh run watch <RUN_ID> --exit-status
```

**Poll-cliff math (tmux mode, web).** Caller sets `poll_rounds: 40`, canon
poll step sleeps 15 s → `40 × 15 s = 600 s = 10 min` maximum fire→collect
window. The `Poll tmux deploy` step has `command_timeout: 30m` but the loop
itself exits after 40 rounds; if `.deploy-$SESS.done` never appears, `Collect`
reads `CODE=$(cat DONE || echo 99)` → run fails with 99 (stale/missing marker,
not a hang). Full budget chain: `wake` job 15 min (KVM forced-command waits up
to 12 min for boot) → `deploy` job 45 min (covers 10 min poll + build + health
curls). If the run exceeds the 10-minute cliff, the failure is in
`ops/deploy-web.sh` (build/health), not in the poller — read the captured
tmux pane (`-S -1000` dump in Collect) rather than re-running blindly.

## 4. Seven acceptance proofs (evidence package per run)

Collect all seven; the run is green only with all seven attached (log excerpts
or command outputs + run URL).

1. **Checks job green** — `checks` (web-checks `contracts-and-build`) success
   on the dispatched SHA; link the run.
2. **Dispatch provenance** — `gh run view <RUN_ID>` shows
   `event=workflow_dispatch`, `ref=web`, input `deploy_local_pilot=true`,
   head SHA `f637d86`.
3. **Canon path executed** — job log shows wake probe (skip-wake if up, else
   KVM wake-triggered) then tmux fire/poll/collect for session
   `deploy-web-<run_id>` in `/opt/haklolarp-web`, `git pull --ff-only`
   clean, no divergence error.
4. **Compose config valid** — `docker compose -p haklolarp-web -f
   compose.web.yaml config -q` exit 0 (plus platform/stream variants only if
   operator enabled them — default pilot: base file only).
5. **Containers up** — `docker ps` shows `haklolarp-web-ui` and
   `haklolarp-web-api` (project `haklolarp-web`), `--wait` satisfied.
6. **Health gate** — `curl --fail http://127.0.0.1:8081/api/v1/health` exit 0
   from the stand (and `/api/v2/health/ready` only if platform enabled).
7. **Isolation intact** — `main` stand (`/opt/haklolarp-staging`, project
   `haklolarp-staging`, port 8080) untouched: its containers still running,
   `git -C /opt/haklolarp-staging rev-parse HEAD` unchanged pre/post run;
   no preview/ml/data dirs touched.

## 5. Rollback (instant, no force-push)

- **Redeploy previous SHA (detached, canonical).** On the stand:
  ```bash
  cd /opt/haklolarp-web
  git fetch origin
  PREV=<last-good-sha>   # e.g. pre-pilot f637d86^ or recorded good SHA
  git checkout --detach "$PREV"
  docker compose -p haklolarp-web -f compose.web.yaml up -d --build --wait
  curl --fail --max-time 10 http://127.0.0.1:8081/api/v1/health
  ```
  then return to branch tracking when ready:
  `git checkout web && git pull --ff-only origin web` (+ redeploy).
  History on `origin/web` is never rewritten; rollback is a stand-local
  redeploy, one command sequence, < 5 min.
- **One-line flag retreat.** If the pilot misbehaves but the previous
  containers are still present, `docker compose -p haklolarp-web stop` (or
  `down` WITHOUT `-v`) halts the pilot while preserving volumes; main/ml
  projects are never in the command line.

## 6. Prohibitions (hard NO for this stage)

- NO ungating: do not remove `needs: checks` or the `deploy_local_pilot`
  condition; do not add `push`-triggered deploy.
- NO renames: workflow name `Deploy web (staging)`, concurrency group
  `staging-deploy-web`, project `haklolarp-web`, dir `/opt/haklolarp-web` —
  frozen (`notify-runs.yml` and operators key on them).
- NO platform/stream enablement from CI: `DTS_PLATFORM_ENABLED` /
  `DTS_STREAM_ENABLED` stay OFF unless the stand operator sets them with a
  private `DTS_POSTGRES_PASSWORD` out of band.
- NO `docker … down -v` / `down --volumes` on ANY stand (pilot or otherwise);
  volume deletion is data loss, never a "cleanup".
- NO force-push to `web` (or any shared branch); NO `merge to main`; NO
  dispatching workflows for other branches as a side effect.
- NO touching `/opt/haklolarp-staging`, `/opt/haklolarp-ml`,
  `/opt/haklolarp-stack`, `/opt/haklolarp-previews`.

## 7. Risks / cost / time

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `.runtime` unseeded → Queue-crash / empty journeys | High (pilot default) | Pilot looks red though plumbing OK | Declare provision decision pre-dispatch (§2); judge plumbing proofs 3–6, not content |
| `APP_PORT` stray export moves pilot off 8081 | Low | Health curl hits wrong port | Pre-check unset; health URL fixed to 8081 |
| Disk pressure from `--build` | Medium | Build fail mid-pilot | `df` gate; prune only dangling images, never volumes |
| Xeon asleep → wake path | Medium | + up to 12 min | Expected; KVM forced-command is wake-only (no reset/ISO by construction) |
| Poll cliff (10 min) exceeded by slow build | Medium | CODE=99, misleading "timeout" | Read tmux pane dump; fix script/build, not the poller |
| Discord webhook missing | Low | Notify step warns (`ack_no_webhook`), run still valid | Proofs are in run logs, not Discord |

**Cost:** github-hosted minutes only (wake ~1–2 min + deploy ≤ ~20 min on
green path); no infra spend, no new secrets, no new hosts.
**Time:** checks ~5–10 min (Node/pytest/browser journeys) + dispatch run
~5–15 min (wake + build + health). Total pilot wall-clock typically
15–30 min; budget 45 min (canon `deploy` timeout) before declaring stuck.
