# P1 — identity split (web team principal)

Staged implementation plan. Builds on the prior consilium verdict; this file is
the buildable final word for P1. Scope: give the web team its own deploy-only
principal (`haklo-web` via a new `WEB_SSH_KEY`) while `hakloboss` keeps the
legacy path for ml / main / preview. No new `ci-web` user. No big-bang
rotation (S-8 explicitly rejected). Additive-before-strip throughout; every
step independently rollbackable.

Phased order for web (fixed): **C → A-additive → B-shadow → A-strip →
B-enforce**.

- **C (content):** land web deploy content (`ops/deploy-web.sh`,
  `compose.web.yaml`, caller wiring) — already staged on `origin/web`
  (`ops/deploy-web.sh` + `ops/web/nginx-production.conf.template` + gated
  `deploy-web.yml` caller). P1 assumes C is merged first; P1 touches identity
  only, never web app content.
- **A-additive:** additive identity changes (teams flags, applier script +
  sudoers, `WEB_*` secrets). Zero behavior change; legacy path live.
- **B-shadow:** shadow web deploy as `haklo-web` through the applier while
  legacy stays live; compare, verify, do NOT cut over.
- **A-strip:** cut web caller to `WEB_*` + strip legacy web access
  (dirs → `0750`, shim removal behind flag, per-branch revocation,
  `docker_users` → `[]` + ejection).
- **B-enforce:** negative asserts in CI so future runs can't silently reunite
  identities.

## 0. Naming rulings (fixed, do not relitigate)

Canonical secret names: **`WEB_SSH_KEY` / `WEB_HOST` / `WEB_USER`**, symmetric
with `STAGING_SSH_KEY` / `STAGING_HOST` / `STAGING_USER`. Minimal caller diff:
the reusable caller already parameterises `staging_host` / `staging_user` /
`staging_ssh_key`, so the web caller swaps secret bindings only.

- `WEBCT_*` **rejected**: asymmetric with `STAGING_*`, larger caller diff,
  invents a third naming family for no benefit.
- `WEB_HOST` value will initially equal `STAGING_HOST` (same Xeon tailnet IP)
  but lives as a **separate secret** so host and key rotate independently per
  branch later. Same for `WEB_USER` (value `haklo-web` today): name, not
  value, is the contract.
- Secrets are referenced **only by name** in code/docs. Fingerprints live in
  `KEYS.md`; private halves never appear in repo, logs (`no_log`), or Discord.

## 1. Target identity map (steady state after P1)

| Branch / path              | SSH user    | Secret set (`*_HOST/*_USER/*_SSH_KEY`) | Sudo        | docker group | Dir / mode              |
|----------------------------|-------------|----------------------------------------|-------------|--------------|-------------------------|
| web → `/opt/haklolarp-web` | `haklo-web` | `WEB_*`                                | applier only (§4) | **no**   | `haklo-web:haklo-web 0750` |
| ml → `/opt/haklolarp-ml`   | `hakloboss` | `STAGING_*` (legacy, kept)             | full via CI | yes (legacy) | `haklo-ml:haklo-ml 0770` (unchanged) |
| main → `/opt/haklolarp-staging` | `hakloboss` | `STAGING_*` (legacy, kept)        | full via CI | yes (legacy) | `0755`, owned by `ansible_user` (role `app`, unchanged) |
| preview (`/opt/haklolarp-previews`) | `hakloboss` | `STAGING_*` (legacy, kept)     | —           | yes (legacy) | unchanged               |
| KVM wake                   | `root@KVM`  | `KVM_WAKE_KEY` (shared, unchanged)     | —           | —            | forced command (precedent only, §4) |

What explicitly does **NOT** change in P1: ml caller, staging caller,
`preview.yml`, `KVM_WAKE_KEY`, `haklolarp_deploy` GitHub deploy-key, `base`
role's `STAGING_*` authorized_keys for `ansible_user`
(`ansible/roles/base/tasks/main.yml:38-44`), tailnet/OAuth wiring.

## 2. teams-role surgery (exact hunks)

Current state (`origin/devops`):

- `ansible/roles/teams/defaults/main.yml:8-10` — shim comment (CI user in team
  groups until teams deliver pubkeys).
- `ansible/roles/teams/defaults/main.yml:14-21` — `teams_list` (web/ml,
  user + dir only).
- `ansible/roles/teams/tasks/main.yml:11-23` — team users created with
  **hardcoded `groups: docker`** (`:15`).
- `ansible/roles/teams/tasks/main.yml:25-30` — **CI-from-all-groups shim**:
  `ansible_user` (hakloboss) added to every team group.
- `ansible/roles/teams/tasks/main.yml:32-42` — team dirs at **`0770`** (`:38`).
- `ansible/roles/teams/tasks/main.yml:78-94` — `safe.directory` trust entries
  for the shared CI user (legacy-deploy support).

### Hunk 2a — defaults: per-team `mode` / `allow_docker` + legacy flag

File `ansible/roles/teams/defaults/main.yml`, replace `:14-21`:

```yaml
teams_list:
  - name: web
    user: haklo-web
    dir: /opt/haklolarp-web
    mode: "0750"          # NEW: per-team dir mode (ml stays 0770 until P2)
    allow_docker: false   # NEW: web deploys via applier (§4), never docker group
  - name: ml
    user: haklo-ml
    dir: /opt/haklolarp-ml
    mode: "0770"          # legacy until P2
    allow_docker: false   # ml team user itself never had a key; CI used hakloboss
# NEW: legacy compat. true = keep shim tasks (A-additive/B-shadow).
# Flipped to false for web at A-strip; removed entirely at sunset (§9).
teams_legacy_compat: true
```

`teams_ci_pubkeys: {}` (`:21`) unchanged — web's pubkey half arrives via
`WEB_SSH_KEY` rotation rite (§5), then its pubkey is pinned here for audit
(the authorized_keys writer stays the existing `:44-53` task).

### Hunk 2b — tasks: docker membership via flag, not hardcoded

File `ansible/roles/teams/tasks/main.yml`, replace `:11-23` user task's
`groups: docker` (`:15`) with conditional membership:

```yaml
- name: Teams | пользователи команд (без sudo, вход только по ключу)
  ansible.builtin.user:
    name: "{{ team.user }}"
    group: "{{ team.user }}"
    groups: "{{ ['docker'] if (team.allow_docker | default(false)) else [] }}"
    append: true
    ...
```

Rationale: `append: true` never *removes*; flipping the flag alone does not
eject. Ejection is an explicit `gpasswd -d` step in §3 (step 6), asserted in
§6. With `allow_docker: false` for both teams, steady state has **no team
user in `docker`**; hakloboss keeps its existing (legacy) membership for
ml/main until P2.

### Hunk 2c — shim tasks behind `teams_legacy_compat` (A-additive), deleted at strip

File `ansible/roles/teams/tasks/main.yml`:

- `:25-30` (CI user into all team groups): add
  `when: teams_legacy_compat | bool and (ansible_user not in ...)`.
  At A-strip the web group is additionally excluded: change `groups:` to
  `"{{ teams_list | selectattr('name', 'ne', 'web') | map(attribute='user') | list }}"`
  when the flag flips — i.e. shim survives for ml only. Sunset (§9) deletes
  the task.
- `:32-42` dirs: `mode: "{{ team.mode | default('0750') }}"` (web → `0750`
  effective at A-strip; ml stays `0770`).
- `:78-94` `safe.directory` for shared CI user: gate with
  `when: teams_legacy_compat | bool`; at A-strip, remove the web dir entry
  (loop over ml only). haklo-web needs **no** `safe.directory` entry because
  the applier (§4) runs git as root, not as the shared CI user.

Diff-shaped summary: defaults +7 lines; tasks `groups:` 1-line conditional,
2 `when:` gates, 1 `mode:` parameterisation, shim loop narrowed at strip.
No other role's task lines change in P1.

## 3. `docker_users` surgery + 7-step migration order

Current state:

- Default `ansible/roles/docker/defaults/main.yml:2`: `docker_users: []`
  (already the desired steady state — good).
- Override `ansible/inventory/group_vars/staging.yml:19-21`:
  `docker_users: ["{{ ansible_user }}"]` (injects hakloboss on every run).
- Writer `ansible/roles/docker/tasks/main.yml:42-47`: append-only loop, never
  ejects.

Steady state after P1: override **removed** (`docker_users` falls back to
`[]`); group membership enforced by explicit ejection + assert, not by list
absence.

Migration order (each step lands separately, each rollbackable per §7):

1. **Pin (additive).** staging.yml override → literal list with comment:
   `docker_users: [hakloboss]  # P1: literal, no templating; gone at step 6`.
   No behavior change. Assert task (check-mode safe) records
   `getent group docker` members to a fact for later diff.
2. **Flag (additive).** Land hunks 2a/2b with `teams_legacy_compat: true` and
   `allow_docker: false`. No behavior change (users already exist; `groups:
   []` + `append: true` is a no-op). Verify: `id haklo-web` still shows
   `docker` (leftover, expected — ejection comes at step 6).
3. **Applier (additive).** Land §4 script + sudoers. No caller uses it yet.
   Verify `sudo -U haklo-web -l` shows exactly the allowlist line.
4. **Shadow (B-shadow).** Manual `workflow_dispatch` (or local ssh) run of
   the web deploy **as `haklo-web` via the applier** into `/opt/haklolarp-web`
   while legacy stays live. Compare `docker ps`, health endpoint, file
   ownership. Legacy remains the serving path; shadow artifacts must not be
   root-owned (§4 DONE rule).
5. **Cutover web caller (A-strip, part 1).** `deploy-web.yml` secret bindings
   `STAGING_*` → `WEB_*` and fire-step command → applier invocation. ml and
   main callers untouched. First cutover run is the live test (§6).
6. **Strip + eject (A-strip, part 2).** In one ansible run, in this order:
   a. web dir → `0750` (`team.mode` now effective);
   b. shim narrowed to ml-only (hunk 2c);
   c. `gpasswd -d hakloboss haklo-web` (hakloboss out of web group);
   d. `gpasswd -d haklo-web docker` (web principal out of docker);
   e. remove staging.yml `docker_users` override (→ default `[]`);
   f. per-branch revocation: delete the STAGING pubkey line from
      `haklo-web`'s `authorized_keys` (ml/main `hakloboss` keys untouched).
   Verify with the full matrix (§6) before declaring done.
7. **Enforce (B-enforce).** Land negative asserts (§8) in CI (ansible
   `--check` or a dedicated assert play + workflow `assert` job). Any future
   run that would reunite identities fails loudly.

Why this order: steps 1–4 change nothing live (rollback = revert commit);
step 5 is the only behavior flip and it rides on a verified shadow; step 6
removes exactly what step 5 made redundant; step 7 locks the door.

## 4. Applier spec (`haklo-web-apply`)

**Forced-command REJECTED for P1** (recorded decision, do not relitigate):
the tmux deploy protocol (`deploy-reusable.yml:158-224`) opens *several*
distinct SSH sessions per run (fire / poll / collect, plus debugging), and
operators need interactive SSH as `haklo-web` during incidents. A forced
command would collapse all of that to one fixed action and break
poll/collect. Precedent `ansible/files/kvm/xeon-wake-only` (+ installer
`ansible/files/kvm/kvm_install.py`, which writes
`restrict,command="..."` into `/root/.ssh/authorized_keys`) is intentionally
*not* followed here: KVM wake is a single fixed action with no follow-up
sessions; web deploy is not. Least-privilege is achieved with a sudo
allowlist instead. tmux is **preserved** unchanged.

- **Script:** `/usr/local/bin/haklo-web-apply`, root-owned `0755`, deployed by
  ansible (`copy`, checksum-verified). Single argument: run-id (the tmux
  session name, `deploy-web-<run_id>`).
- **Arg allowlist (in-script, strict):** `"$#" -eq 1` else exit 2; `$1` must
  match `^deploy-web-[A-Za-z0-9_-]+$` else exit 2. No subcommands, no flags,
  no paths, no env passthrough except an explicit `ALLOWLIST` of
  `DTS_*` vars (default deny; unset anything else). Script body = the current
  `ops/deploy-web.sh` steps (fetch/checkout/pull, compose up, health curls)
  with `set -eu`, fixed `DIR=/opt/haklolarp-web`, fixed branch `web`.
- **Sudoers line** (via ansible `copy` to `/etc/sudoers.d/haklo-web-apply`,
  mode `0440`, `visudo -c` validated):
  `haklo-web ALL=(root) NOPASSWD: /usr/local/bin/haklo-web-apply *`
  The trailing `*` is the **only** wildcard and covers just the run-id
  position; the script's regex is the real allowlist (sudoers globs cannot
  express regexes). No `NOPASSWD` for anything else; `haklo-web` has no other
  sudo.
- **tmux preservation:** CI fire-step keeps its shape, only the inner command
  changes:
  `tmux new-session -d -s "$SESS" ... "sudo -n /usr/local/bin/haklo-web-apply $SESS; echo \$? > $DIR/.deploy-$SESS.done"`.
  Poll/collect steps unchanged.
- **DONE ownership rule:** the `.deploy-$SESS.done` file is written by the
  tmux wrapper running **as `haklo-web`** (redirection outside sudo), so it
  stays `haklo-web`-owned. The applier itself must not leave root-owned files
  under `/opt/haklolarp-web`: set `umask 022`, and `chown haklo-web:haklo-web`
  anything it creates (compose build cache lives in docker, not the dir;
  audit with `find /opt/haklolarp-web -user root` in §6).

## 5. Key inventory + rotation rite + per-branch revocation

| Secret / key       | Principal(s) after P1       | Change in P1 |
|--------------------|-----------------------------|--------------|
| `WEB_SSH_KEY` (new) | `haklo-web` only           | **created**: `ssh-keygen -t ed25519`; pub → `teams_ci_pubkeys['haklo-web']` + `KEYS.md` row; priv → GitHub secret `WEB_SSH_KEY` (+ owner password-manager copy, per KEYS.md rite) |
| `WEB_HOST` (new)   | value = Xeon tailnet IP     | new secret, same value as `STAGING_HOST` today |
| `WEB_USER` (new)   | value `haklo-web`           | new secret |
| `STAGING_SSH_KEY`  | `hakloboss` (ml/main/preview) | **kept, NOT rotated** in P1; its pubkey line removed from `haklo-web`'s `authorized_keys` only (step 6f) |
| `KVM_WAKE_KEY`     | shared wake, all branches   | unchanged |
| `haklolarp_deploy` | Xeon → GitHub read-only     | unchanged |
| `TS_OAUTH_*`, `STAGING_HOST`, `DISCORD_*` | — | unchanged |

Rotation rite (applies to the one new key now, and all future ones):
generate → pub into repo/authorized_keys via ansible → priv into named secret
(+ offline owner copy) → verify live (§6) → revoke predecessor **for that
branch only**. Revocation is per-branch at cutover (step 6f), never a big-bang
rotation of every secret at once (S-8 rejected: multiplies blast radius and
couples unrelated branches' rollbacks).

`KEYS.md` rows to add (pubkey fingerprints only, no private material):

- `WEB_SSH_KEY` | web-team → Xeon as `haklo-web` (deploy-only, applier sudo)
  | priv: GitHub secret `WEB_SSH_KEY` + owner password manager; pub:
  `authorized_keys` of `haklo-web` via role `teams` | rotation: rite above;
  revocation scoped to `haklo-web`.
- Amend `STAGING_SSH_KEY` row: add "P1: valid for `hakloboss` (ml/main/
  preview) only; no longer accepted as `haklo-web`".

## 6. Verification matrix (exact commands; run from dev-VM / admin box)

Run after step 5 (cutover) and again after step 6 (strip). All must pass.

| # | Check | Command | Expect |
|---|-------|---------|--------|
| V1 | web principal exists, locked password | `id haklo-web` | `uid=…(haklo-web) gid=…(haklo-web) groups=…(haklo-web)` **without** `docker`, **without** `hakloboss` groups |
| V2 | docker group membership | `getent group docker` | no `haklo-web`; `hakloboss` still present (legacy for ml/main) |
| V3 | sudo allowlist exact | `sudo -U haklo-web -l` | exactly `NOPASSWD: /usr/local/bin/haklo-web-apply *`, nothing else |
| V4 | docker denied for web | `ssh -i <WEB_SSH_KEY> haklo-web@<WEB_HOST> 'docker ps'` | non-zero exit, permission denied |
| V5 | applier works | `ssh … haklo-web@… 'sudo -n /usr/local/bin/haklo-web-apply deploy-web-verify1; echo $?'` then `tmux kill-session -t deploy-web-verify1` cleanup + health curl | exit 0, `http://127.0.0.1:8081/api/v1/health` OK |
| V6 | applier rejects junk | `… 'sudo -n /usr/local/bin/haklo-web-apply "a;id"'`, `… 'sudo -n /usr/local/bin/haklo-web-apply x y'`, `… 'sudo -n id'` | exit 2 / denied respectively; nothing executed |
| V7 | dir modes/owners | `stat -c '%a %U %G %n' /opt/haklolarp-web /opt/haklolarp-ml` | `750 haklo-web haklo-web /opt/haklolarp-web`; ml `770 …` unchanged |
| V8 | no root-owned debris | `find /opt/haklolarp-web -user root` | empty |
| V9 | SSH audit: legacy key rejected as web | `ssh -i <STAGING_SSH_KEY> haklo-web@<WEB_HOST> true` | `Permission denied (publickey)` |
| V10 | SSH audit: legacy path intact | `ssh -i <STAGING_SSH_KEY> hakloboss@<STAGING_HOST> 'echo ok; id -nG'` | `ok`, groups include `docker`, `haklo-ml` (shim ml-only) |
| V11 | live deploy | `workflow_dispatch` / push to `web` | green run, Discord `notify-web` fires, health OK |
| V12 | keys audit | `cut -d' ' -f2 ~/.ssh/WEB_SSH_KEY.pub \| …` vs `KEYS.md` fingerprint; `gh secret list` | fingerprint match; `WEB_SSH_KEY/WEB_HOST/WEB_USER` present; no other new secrets |

## 7. Rollback per step (§3 numbering)

1. Pin: revert commit, re-run ansible. No live effect.
2. Flags: revert, re-run. `append: true` no-op both ways.
3. Applier: `rm /etc/sudoers.d/haklo-web-apply /usr/local/bin/haklo-web-apply`;
   nothing references them yet.
4. Shadow: kill shadow tmux session, `rm -f` shadow DONE file; serving path
   untouched. `chown -R haklo-web:haklo-web /opt/haklolarp-web` if shadow left
   debris.
5. Cutover: revert `deploy-web.yml` to `STAGING_*` bindings (one-commit
   revert, re-run workflow). `WEB_*` secrets stay configured but unused —
   safe to leave for the retry.
6. Strip: re-add `hakloboss` to `haklo-web` group + STAGING pubkey to
   `haklo-web` authorized_keys (one ansible run with `teams_legacy_compat:
   true`); dir back to `0770`; restore override. Note: `gpasswd -d` ejection
   of `haklo-web` from `docker` needs **no** rollback (web never needed it).
7. Enforce: asserts are additive checks; revert if they false-positive, fix
   forward preferred.

No step requires restoring from backup; worst case (step 5/6) is a single
revert + one workflow re-run on the legacy path, which stays functional until
step 6f and recoverable after it via the step-6 rollback.

## 8. Negative guards (B-enforce; fail the run, don't warn)

Ansible assert tasks (dedicated `assert` play or post-tasks in `teams`, run in
CI after every `site.yml`/`deploy.yml` that touches staging):

- `haklo-web` not in `docker`: `getent group docker` stdout must not match
  `(^|,)haklo-web(,|$)`.
- `hakloboss` not in `haklo-web` group: `id -nG hakloboss` must not contain
  `haklo-web`.
- STAGING pubkey absent from `haklo-web` authorized_keys: slurp
  `/home/haklo-web/.ssh/authorized_keys`, compare fingerprint set against the
  pinned STAGING pub fingerprint var — must be disjoint.
- Dir mode exact: `stat -c %a /opt/haklolarp-web` == `750`.
- Sudo allowlist exact: `sudo -U haklo-web -l` output == single
  `haklo-web-apply` line (regex `^haklo-web ALL=\(root\) NOPASSWD:
  /usr/local/bin/haklo-web-apply \*$`).
- Workflow guard (`deploy-web.yml`): a `lint` job greps the caller file —
  `STAGING_SSH_KEY` / `STAGING_USER` must not appear in it (catches
  copy-paste reunification at PR time).

## 9. Risks, cost, time; sunset tickets owned by this phase

Risks: (a) step-6 ansible run partially applies (dir mode flipped but ejection
missed) → mitigated by the §6 matrix run *inside* the same maintenance window
before declaring done; (b) applier regex bypass → mitigated by V6 plus
shellcheck on the script; (c) `WEB_HOST` drift from `STAGING_HOST` later →
documented as intentional (§0), health check catches it; (d) tmux sessions
orphaned across cutover → fire-step already kills same-branch stale sessions
(`deploy-reusable.yml:178-184`), applier keeps that logic outside sudo.

Cost: ~€0 (no new infra; one extra GitHub secret set; same single Xeon host).
Time: 2–4 h wall-clock across two windows (steps 1–4 offline any time; steps
5–6 in one window with the matrix; step 7 follow-up). No downtime for
ml/main; web deploy paused only between step 5 revert-point and V11 green
(minutes if shadow was clean).

Sunset tickets owned by P1 (filed, not done here):

- S-P1a: delete `teams_legacy_compat` + CI-shim task entirely after ml
  migrates (P2); dirs ml → `0750`.
- S-P1b: rotate `STAGING_SSH_KEY` after P2 (postponed deliberately — rotating
  now would couple ml/main rollbacks to P1).
- S-P1c: P2 identity split for ml (`ML_SSH_KEY`/`ML_HOST`/`ML_USER`,
  `haklo-ml` applier) mirroring this plan; preview ownership decision rides
  with P2.
- S-P1d: quarterly `gh secret list` ⇄ `KEYS.md` reconciliation already stands;
  add the three `WEB_*` names to that checklist.
