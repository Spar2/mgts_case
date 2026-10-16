# P2 — Tailnet ACL staged plan (FINAL)

> Scope: Tailscale tailnet lockdown for the Xeon staging stand. Console is
> authoritative; repo holds a reviewed mirror + drift check. LAN path is
> never gated by tailnet ACL and is the break-glass.
> Branch note: this plan lands on `devops`. Push to `devops` fires only a
> Discord push-note; devops deploys nothing.

## 0. Verified ground truth (2026-10-13, `origin/devops` @ `2eb7855`, ref `origin/web` @ `f637d86`)

Checked in scratch worktree `plan-t2` (HEAD == `origin/devops` == `2eb7855`, clean):

- **No ACL/policy file in repo.** `git grep "tagOwners|grants|nodeAttrs|autoApprovers"`
  returns empty; no `*.hujson` / `policy*` / `acl*` files. Tailnet is
  **all-to-all today** — every node can reach every node.
- **Single real tailnet node:** `xeon-staging` = `100.66.240.22`
  (`ansible/files/ssh_known_hosts`, `ansible/files/kvm/xeon-wake-only` probe target).
  KVM `kvm-2354` tailnet address `100.68.100.66` (`ansible/README.md`, secret `KVM_HOST`).
- **CI is ephemeral OAuth:** `deploy-reusable.yml` + `preview.yml` join via
  `tailscale/github-action` with `oauth-client-id/secret` (`TS_OAUTH_CLIENT_ID/SECRET`),
  `tags: tag:ci`, per-run hostnames (`wake-<run>`, `deploy-<branch>-<run>`, `pr-<N>`).
  Legacy `TAILSCALE_AUTH_KEY_RUNNERS` secret deleted, key revoked (README table).
- **Host key is reusable, tagged-by-intent:** `TAILSCALE_AUTH_KEY` (secret, set)
  used by role `tailscale` (`tailscale up --auth-key=... --hostname=xeon-staging`);
  README instructs generating it reusable, preferably with `tag:staging`.
  Condition is no-logout-only: role runs `tailscale up` only when status shows
  `Logged out` / rc != 0 (`ansible/roles/tailscale/tasks/main.yml:64-66`).
- **Funnel / Serve unmanaged:** no `tailscale serve` / `funnel` config anywhere in
  repo or roles; no `nodeAttrs`/`autoApprovers` to audit — nothing to migrate,
  everything to lock down (default-off).
- **LAN ground truth (break-glass, ACL-independent):**
  `xeon-staging` `192.168.1.123` (br0 static; local override via untracked
  `host_vars/xeon-staging/local.yml`), `kvm-2354` `192.168.1.4`,
  runners `.100–.102` (manual, future DHCP reservation).
  Inventory: `ansible/inventory/hosts.yml`; staging vars:
  `ansible/inventory/group_vars/staging.yml`.
- **Team access today (pre-ACL):** `TEAM_GUIDE.md` (exists only on `web`, ref
  `f637d86`): collaborator + **one-shot key**
  `tailscale up --auth-key=<one-shot> --hostname=<machine>`, verify via
  `curl http://100.66.240.22:8081`. Ports: main `8080`, web `8081`, ml `8082`,
  previews `9000+N` (`/opt/haklolarp-previews/pr-<N>`, project `haklolarp-pr-<N>`).
- **OS-level team split exists, tailnet-level does not:** role `teams`
  (`ansible/roles/teams/`) creates `haklo-web`/`haklo-ml`, dirs
  `/opt/haklolarp-web|ml` (`0770`), per-team CI pubkeys via `teams_ci_pubkeys`
  (currently `{}` — stub). Shared `ansible_user` still in team groups (to be
  removed at key split). Docker-group caveat documented in role defaults:
  container visibility is host-wide; real boundary is hypervisor (runner VMs).

## 1. Tag taxonomy + tagOwners (exact names, no `tag:grok` until its box exists)

| Tag | Who / what | Owner |
|---|---|---|
| `tag:staging` | `xeon-staging` (Xeon host in tailnet) | `autogroup:admin` (owner) |
| `tag:kvm` | `kvm-2354` (NanoKVM-PCIe) | `autogroup:admin` |
| `tag:ci` | Ephemeral GitHub-runner nodes (`wake-*`, `deploy-*`, `pr-*`) via OAuth client | `autogroup:admin` (OAuth client pinned to this tag) |
| `tag:devops` | Owner/admin laptops + debug VM (`gritsko@192.168.1.20`-class machines) | `autogroup:admin` |
| `tag:web` | Web-team member devices (one-shot keys) | `autogroup:admin` issues keys; team lead approves membership list |
| `tag:ml` | ML-team member devices (one-shot keys) | same as web |

Rules:

- Tag names are exact; no `tag:grok` (or any new team tag) until that team's box
  exists and an owner is named. Adding a tag = ACL PR + key runbook update.
- `tagOwners` maps each tag to `autogroup:admin` only. No team self-approves tags
  (prevents privilege self-escalation via key reuse).
- Hostnames stay stable for servers (`xeon-staging`, `kvm-2354`); CI hostnames stay
  per-run ephemeral; team devices use `<person>-<device>` (auditable in admin panel).

```jsonc
// mirror sketch — console authoritative, this copy is reviewed (see §8)
{
  "tagOwners": {
    "tag:staging": ["autogroup:admin"],
    "tag:kvm":     ["autogroup:admin"],
    "tag:ci":      ["autogroup:admin"],
    "tag:devops":  ["autogroup:admin"],
    "tag:web":     ["autogroup:admin"],
    "tag:ml":      ["autogroup:admin"]
  }
}
```

## 2. Node-by-node tagging runbook (no-logout attach, LAN fallback, keep MagicDNS `xeon-staging`)

Principle: **never `tailscale logout`** on a tagged node. Re-tag = `tailscale up`
(or `tailscale set --advertise-tags=...` where supported) with the same identity;
logout would drop the node from the tailnet and force a full re-auth over a
possibly already-tightened ACL. Every step has a LAN fallback that bypasses the
tailnet entirely.

Pre-conditions per node: note current `tailscale status`, `tailscale ip -4`,
`tailscale debug prefs` (advertised tags/routes/exit-node), and keep an open LAN
SSH session (`ssh hakloboss@192.168.1.123`, `ssh root@192.168.1.4`) for the whole
operation.

| # | Node | Command (from LAN session) | Verify | Fallback |
|---|---|---|---|---|
| 1 | `xeon-staging` | `sudo tailscale up --advertise-tags=tag:staging --hostname=xeon-staging` (reuses existing auth; no logout; keep `TS_FORCE_NOISE_443=1` drop-in) | `tailscale status` shows `tag:staging`; MagicDNS `xeon-staging` still resolves; `tailscale ip -4` still `100.66.240.22` | LAN `192.168.1.123` unaffected; re-run via Ansible role `tailscale` with `TAILSCALE_AUTH_KEY` (tagged reusable key) |
| 2 | `kvm-2354` | KVM console / `ssh root@192.168.1.4`: `tailscale up --advertise-tags=tag:kvm --hostname=kvm-2354` (or `tailscale set --advertise-tags=tag:kvm` if already up) | `tailscale status` on admin shows `kvm-2354 [tag:kvm]`; `100.68.100.66` reachable from `tag:ci` only (post-grants) | LAN `192.168.1.4` + physical console; wake script `xeon-wake-only` is LAN-local, never tailnet-gated |
| 3 | CI runners | No change on hosts: pin OAuth client to `tag:ci` in admin panel (Devices → OAuth clients → allowed tags = `tag:ci` only) | Next `deploy-reusable` run joins as `tag:ci` (run log: Join Tailscale step); node auto-expires after run | Nothing to fall back (ephemeral); if OAuth mis-tags, runs fail closed at probe — LAN deploy from debug VM still works |
| 4 | devops machines | Each: `tailscale up --advertise-tags=tag:devops --hostname=<person>-<device>` with one-shot tagged key (see §6) | `tailscale status` shows tag; `ssh hakloboss@xeon-staging` (tailnet) works post-G3 | LAN `ssh gritsko@192.168.1.20` / `hakloboss@192.168.1.123` |
| 5 | web/ml devices | Same pattern with `tag:web` / `tag:ml` one-shot keys, hostnamed per person | Appear tagged; reach only their ports (§3 G4/G5) | No LAN (remote members) — onboarding stays in CHECK/allow-http stage until verified (§7 stage 6) |

MagicDNS: keep `xeon-staging` (and `kvm-2354`) names pinned; do not rename during
tagging. Secrets `STAGING_HOST`/`KVM_HOST` keep IP literals until stage 5 proves
MagicDNS under ACL, then optionally switch to names (separate PR, not this plan).

## 3. Grants matrix G1–G9 + SSH-check rules + DENY list

Least-privilege ports per tag. Staging listens: `22` (SSH), `8080` (main),
`8081` (web), `8082` (ml), `9000+N` (previews). KVM listens: `22` (forced-command
SSH only).

Allow-grants (stateful TCP):

- **G1 — `tag:ci → tag:staging:22`.** CI probe + deploy SSH. The only writer path
  for `deploy-reusable.yml` / `preview.yml`.
- **G2 — `tag:ci → tag:kvm:22`.** CI wake path (`KVM_WAKE_KEY`, forced command
  `xeon-wake-only`; power-off/reset unreachable by construction).
- **G3 — `tag:devops → tag:staging:22,80,443,8080,9000-9100`** (narrowed: 8081/8082 excluded — NOT ENFORCEABLE, web/ml loopback-only 127.0.0.1 till rebind; previews range 9000-9100 = 9000+N).
  Owner/admin full staging access (SSH + reachable HTTP surfaces incl. previews).
- **G4 — `tag:web → tag:staging:8081,9000-9100` ONLY (8081 NOT ENFORCEABLE till rebind — loopback-only; effectively 9000-9100 only).** No `:22`, no `:8080`,
  no `:8082`. Previews range covers `9000+N` PR envs (9001+, grant covers 9000-9100).
- **G5 — `tag:ml → tag:staging:8082,9000-9100` ONLY (8082 NOT ENFORCEABLE till rebind — loopback-only; effectively 9000-9100 only).** Mirror of G4 for ML.
- **G6 — `tag:ci → tag:staging:8080,9000-9100`** (narrowed: 8081/8082 excluded — NOT ENFORCEABLE, loopback-only; CI tailnet smoke works only for wildcard 8080 + previews 9000-9100).
  No other CI→staging ports.

Deny-groups (explicit, auditable; tailnet default-deny still applies last):

- **G7 — DENY `tag:web,tag:ml → tag:staging:22`.** Teams never SSH over tailnet
  (they deploy via CI; shell access = owner-assisted LAN session only).
- **G8 — DENY `tag:staging → tag:web,tag:ml,tag:ci,tag:devops` (all ports).**
  Staging initiates nothing toward clients (no lateral movement off a
  compromised stand). Covers also DENY cross-team (`web↔ml` have no grants
  between them at all — no rule, no path).
- **G9 — DENY subnet routes / exit-node transit.** `nodeAttrs` empty (no
  `nextdns`, no `mullvad`, no `funnel` attrs), `autoApprovers` empty; no
  `--advertise-routes` / `--advertise-exit-node` approved on any node.
  Any route advertisement stays unapproved = inactive.

Consolidated DENY list (negative tests in §9 target each):

- **D1:** `web/ml → staging:22` refused.
- **D2:** `staging → any client tag` refused (all ports).
- **D3:** `web ↔ ml` refused (all ports, both directions).
- **D4:** subnet/exit-node paths refused (no approved routes; exit-node offers ignored).

Tailscale SSH (CHECK mode only):

- `ssh` section stays `"action": "check"` for all rules during rollout — connections
  are logged/evaluated, never auto-accepted by the tailnet. Actual SSH auth remains
  host keys (`authorized_keys`), unchanged.
- Flip any rule to `accept` **only after** the teams-key split lands:
  per-team users `haklo-web`/`haklo-ml` own their dirs with their own CI keys
  (`teams_ci_pubkeys` filled), shared `ansible_user` removed from team groups,
  dir modes back to `0750` (role `teams` TODO). Until then, `accept` would grant
  the shared user across team boundaries — the exact hole G7/D3 closes at L3/L4.

```jsonc
// grants sketch (mirror)
{
  "grants": [
    {"src": ["tag:ci"],     "dst": ["tag:staging"], "ip": ["tcp:22"], "action": "accept"}, // G1
    {"src": ["tag:ci"],     "dst": ["tag:kvm"],     "ip": ["tcp:22"], "action": "accept"}, // G2
    {"src": ["tag:devops"], "dst": ["tag:staging"], "ip": ["tcp:22","tcp:80","tcp:443","tcp:8080","tcp:9000-9100"], "action": "accept"}, // G3 narrowed: 8081/8082 NOT ENFORCEABLE (loopback-only till rebind)
    {"src": ["tag:web"],    "dst": ["tag:staging"], "ip": ["tcp:8081","tcp:9000-9100"], "action": "accept"}, // G4: 8081 NOT ENFORCEABLE till rebind
    {"src": ["tag:ml"],     "dst": ["tag:staging"], "ip": ["tcp:8082","tcp:9000-9100"], "action": "accept"}, // G5: 8082 NOT ENFORCEABLE till rebind
    {"src": ["tag:ci"],     "dst": ["tag:staging"], "ip": ["tcp:8080","tcp:9000-9100"], "action": "accept"}, // G6 narrowed: 8081/8082 NOT ENFORCEABLE (loopback-only)
    {"src": ["tag:web","tag:ml"], "dst": ["tag:staging"], "ip": ["tcp:22"], "action": "deny"}, // G7/D1
    {"src": ["tag:staging"], "dst": ["tag:web","tag:ml","tag:ci","tag:devops"], "ip": ["*"], "action": "deny"}, // G8/D2+D3(server side)
    {"src": ["tag:web"], "dst": ["tag:ml"], "ip": ["*"], "action": "deny"}, // G8 ctd D3
    {"src": ["tag:ml"],  "dst": ["tag:web"], "ip": ["*"], "action": "deny"}  // G8 ctd D3
  ],
  "nodeAttrs": [],
  "autoApprovers": {"routes": {}, "exitNode": []},
  "ssh": [
    {"action": "check", "src": ["autogroup:admin"], "dst": ["autogroup:self"], "users": ["root","hakloboss","haklo-web","haklo-ml"]}
  ]
}
```

Order note (LAN-first): default-deny goes **last**, after every allow + deny +
verification stage is green. Any earlier flip risks locking CI out of wake/SSH
with no self-serve recovery.

## 4. Funnel / exit-node / subnet lockdown + per-node proof

Target state: everything off, explicitly.

- **Funnel:** no `tailscale funnel` on any node. Proof per node:
  `tailscale serve status` → `No status available` / empty; admin panel
  Devices → node → no Funnel badge.
- **Serve:** no `tailscale serve` exposures (staging serves via compose
  ports + tailnet L4 grants, not via Serve — honestly: main `compose.yaml` is implicit 0.0.0.0 wildcard open-bind, web/ml `compose.*.yaml` are 127.0.0.1 loopback-only (tailnet grants for 8081/8082 NOT ENFORCEABLE till rebind), stack `compose.yaml` loopback-only). Proof: same `serve status` empty.
- **Exit nodes:** no node advertises `--advertise-exit-node`; none offers exit
  service. Proof: `tailscale exit-node list` → `No exit nodes available`;
  `tailscale debug prefs | grep -i exit` shows false/empty on both servers.
- **Subnet routes:** no `--advertise-routes`; `autoApprovers.routes` = `{}` so
  even a mistaken advertisement stays unapproved. Proof:
  `tailscale status --json | jq .SelfNode.AllowedIPs` = host `/32` only;
  admin panel Routes page empty.
- **Lockdown representation:** `"nodeAttrs": []`, `"autoApprovers": {"routes": {}, "exitNode": []}`
  in the mirror; console must show zero approved routes / zero exit nodes after stage 5.

## 5. Key hygiene table + rotation + snapshot-revert runbook

| Key | Type | Tag | Expiry | Stored in | Rotation |
|---|---|---|---|---|---|
| Host `TAILSCALE_AUTH_KEY` (xeon-staging) | reusable, non-ephemeral | `tag:staging` | 90 d | GH secret `TAILSCALE_AUTH_KEY` + owner age-backup (`ansible/KEYS.md` flow) | Re-generate ≤90 d; attach via LAN `tailscale up` (no logout); update secret; revoke old |
| KVM key (kvm-2354) | reusable, non-ephemeral | `tag:kvm` | 90 d | owner backup only (never in repo) | Same as host; KVM console/LAN fallback |
| CI OAuth client (`TS_OAUTH_CLIENT_ID/SECRET`) | OAuth, ephemeral nodes | `tag:ci` only | short-lived creds; nodes die post-run | GH secrets `TS_OAUTH_CLIENT_ID/SECRET` | Rotate secret pair on leak/schedule; allowed-tags pinned to `tag:ci`; no repo change needed |
| Team one-shot keys (web/ml/devops onboarding) | single-use auth keys | per-team tag | shortest offered (1 h–24 h where available) | generated in console, handed 1:1, never committed | Each onboarding = fresh key; used key auto-invalidates; compromised device = remove node in console + fresh key |
| `STAGING_SSH_KEY` / `KVM_WAKE_KEY` | SSH (not tailnet) | n/a | no expiry, forced-command limits blast radius | GH secrets; pubs in `ansible/files/` | Unaffected by ACL; rotate per `ansible/KEYS.md` if leaked |

Snapshot-revert runbook (before every stage):

1. Admin panel → Access controls → copy current JSON to
   `~/haklolarp-keys/acl-snapshot-<date>.json` (+ age-backup alongside keys).
2. Record: `tailscale status`, both servers' `tailscale ip -4`, OAuth allowed-tags,
   approved routes/exit nodes (screenshots or JSON dump).
3. Revert = paste snapshot JSON back in console, Save. No repo/deploy involved;
   LAN SSH never depends on this. Rollback detail in §10.

## 6. 7-stage rollout (with abort criteria; LAN break-glass every stage)

- **S0 Preflight.** Snapshot (§5). Confirm LAN SSH to `.123`/`.4` from admin net.
  Confirm CI green on all three callers. Confirm `serve status`/routes/exit empty.
  Abort if: LAN unreachable, CI red, or snapshot not saved — fix first, no ACL edits.
- **S1 Shadow grants (monitor-only).** Add G1–G6 as new rules *alongside* the
  current allow-all (no deny, no tag enforcement). Watch CI + team curls for 24 h.
  Abort if: any allow misses (CI probe fails, team curl fails) — rules are wrong,
  fix and re-shadow. LAN unaffected throughout.
- **S2 Tag attach.** Runbook §2 node-by-node (servers → OAuth pin → devops).
  Keep allow-all still on. Verify tags visible in console + MagicDNS stable.
  Abort if: any node drops (logout/mis-tag) — re-attach over LAN, no ACL change yet.
- **S3 Dry-run deny (CHECK/log).** Add G7/G8/D1–D4 as `check`/log-only (or lowest
  priority) — collect what *would* break. Run V-matrix (§9). Abort if: V-ci red
  or V-dev red — allows incomplete, return to S1.
- **S4 Tighten.** Enforce denies (G7/G8 active), keep SSH `check`. Re-run full
  V-matrix incl. NEG tests. Abort if: any POS test fails — revert to S3 snapshot
  (one-click console restore), LAN deploy path covers staging meanwhile.
- **S5 Default-deny last.** Remove allow-all / enable default deny. Full V-matrix
  + 24 h soak (CI cron + team spot checks). Abort if: soak shows drops —
  one-click restore S4 snapshot. **LAN break-glass: every abort path includes
  `ssh hakloboss@192.168.1.123` / `ssh root@192.168.1.4` which survives any ACL
  mistake by design (not tailnet).**
- **S6 Team onboarding.** Issue one-shot `tag:web`/`tag:ml` keys, verify G4/G5-only
  reachability per device, then (and only then) gate SSH `accept` on the teams-key
  split (§3). Abort per-device: remove node in console, re-issue key; global abort
  = S5 snapshot restore.

## 7. In-repo mirror (policy.hujson sketch + drift-check job sketch + check.sh lint)

Console authoritative; repo holds a reviewed copy + drift check. New files
(proposed, not created by this plan):

`tailscale/policy.hujson` — full sketch = §1 `tagOwners` + §3 `grants`/`ssh` +
`"nodeAttrs": []`, `"autoApprovers": {"routes": {}, "exitNode": []}` (assembled
from the two JSONC blocks above; comments stripped for the console paste).

`.github/workflows/tailnet-drift.yml` (sketch):

```yaml
name: Tailnet drift check
on: { schedule: [{ cron: "17 3 * * *" }], workflow_dispatch: {} }
permissions: { contents: read }
jobs:
  drift:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - name: Join Tailscale (audit OAuth, tag:ci)
        uses: tailscale/github-action@v4
        with:
          oauth-client-id: ${{ secrets.TS_OAUTH_CLIENT_ID }}
          oauth-secret: ${{ secrets.TS_OAUTH_SECRET }}
          tags: tag:ci
          hostname: drift-${{ github.run_id }}
      - name: Probe matrix (expect allow/deny per §9)
        run: |
          set -e
          nc -z -w5 "$STAGING_HOST" 22        # G1 POS
          curl -sf "http://$STAGING_HOST:8081" >/dev/null  # G6 POS (web port up)
        env: { STAGING_HOST: ${{ secrets.STAGING_HOST }} }
      # NOTE: full NEG probes (D1–D4 from team-tagged nodes) need per-tag audit
      # identities; until then the job asserts POS + validates policy.hujson parses
      # (huJSONC strip + jq) and matches the console revision noted in the file header.
```

`ansible/check.sh` addition (lint): validate `tailscale/policy.hujson`
(comment-strip + `jq empty` + tag-name allowlist
`^(staging|ci|web|ml|devops|kvm)$` + forbid `tag:grok`) — offline, no tailnet
needed, runs in existing lint lane.

## 8. Verification matrix (V-ci / V-dev / V-team + NEG)

From the indicated source tags (POS = must succeed, NEG = must fail):

| ID | From → To | Expect | Command |
|---|---|---|---|
| V-ci-1 | `tag:ci → staging:22` (G1) | POS | CI probe `echo xeon-up` via `appleboy/ssh-action` |
| V-ci-2 | `tag:ci → kvm:22` (G2) | POS | CI wake step (forced command) |
| V-ci-3 | `tag:ci → staging:8080/8081/8082` (G6) | POS | `curl -sf http://<staging>:8081` in CI post-deploy |
| V-dev-1 | `tag:devops → staging:22` (G3) | POS | `ssh hakloboss@xeon-staging echo ok` |
| V-dev-2 | `tag:devops → staging:8080+8081+8082+previews` (G3) | POS | `curl` each port |
| V-team-web | `tag:web → staging:8081` (G4) | POS | `curl http://100.66.240.22:8081` (TEAM_GUIDE flow) |
| V-team-ml | `tag:ml → staging:8082` (G5) | POS | `curl http://<staging>:8082` |
| NEG-D1a | `tag:web → staging:22` | REFUSED | `nc -z -w5 <staging> 22` → fail; `ssh -v` → filtered |
| NEG-D1b | `tag:ml → staging:22` | REFUSED | same |
| NEG-D2 | `tag:staging → tag:web/ml/ci/devops` any port | REFUSED | from Xeon: `nc -z -w5 <client-ip> <port>` → fail |
| NEG-D3a | `tag:web → tag:ml` | REFUSED | no path (no grant) |
| NEG-D3b | `tag:ml → tag:web` | REFUSED | no path (no grant) |
| NEG-D4 | subnet/exit transit | REFUSED | `tailscale status --json` shows no approved routes; exit-node list empty |

Stage gates: S3 requires V-ci + V-dev POS; S4 requires full matrix POS+NEG;
S5 requires full matrix + 24 h soak; S6 requires V-team POS + NEG-D1/D3 per device.

## 9. Rollback (one-click console restore, LAN unaffected)

- **One-click restore:** admin panel → Access controls → replace JSON with the
  stage snapshot (`~/haklolarp-keys/acl-snapshot-<stage>.json`) → Save.
  Propagation is seconds; no deploy, no reboot, no key rotation needed.
- **LAN is unaffected by definition:** `192.168.1.123` / `192.168.1.4` SSH,
  WoL + `xeon-wake-only`, Ansible `site.yml`/`deploy.yml`/`kvm.yml` all run over
  LAN and never consult tailnet ACL. A total ACL lockout still leaves full
  staging control + CI-equivalent manual deploy from the debug VM.
- **Key mistakes:** wrong tag on a key = revoke key + remove node in console +
  re-issue one-shot; no snapshot restore needed. Wrong OAuth pin = re-pin to
  `tag:ci`, re-run one CI caller as proof.

## 10. Risks / mitigations / cost

| Risk | Mitigation | Cost |
|---|---|---|
| Locking CI out of wake/SSH (S4/S5) | LAN-first order (deny last), per-stage snapshots, LAN break-glass, soak before S6 | ~0 (process) |
| `tailscale logout` during tagging drops a server | No-logout rule (§2) + open LAN session + reusable tagged key ready | ~0 |
| Team device over-reach (pre-split shared SSH user) | SSH stays CHECK until key split; G7/D1 enforced at L4 regardless | key-split work already tracked in role `teams` |
| MagicDNS rename churn breaks `STAGING_HOST` | Keep IP literals until S5 proves names; rename is a separate PR | ~0 |
| Funnel/Serve accidentally enabled later | `nodeAttrs: []` + drift job + review rule: any Serve/Funnel = ACL PR | ~0 |
| Stale mirror (console drifts from repo) | Drift-check job (§7) + header revision note in `policy.hujson` | one cron runner, minutes/day |
| New team (grok) needs access | Explicit non-goal: no `tag:grok` until box exists; then tag + G-rule + key runbook PR | scoped later |

Cost total: no new infra, no paid features (ACLs, tags, OAuth, MagicDNS are
native). Time: ~2–3 h hands-on across stages + 24 h soak.

## 11. Good practices applied

- **Default-deny last (LAN-first order):** S0→S5 never removes the working path
  before the replacement is proven; LAN is the path that needs no proof.
- **Break-glass survives any ACL mistake:** LAN SSH + WoL + KVM console are
  physically independent of tailnet policy; every stage lists its fallback.
- **Policy-as-reviewed-mirror:** console authoritative (only place ACL enforces),
  repo holds the reviewed copy + drift check — reviews catch intent errors,
  console stays the single writer.
- **Least-privilege ports per tag:** CI gets `:22` + HTTP smoke only; teams get
  exactly their port + previews; staging initiates nothing.
- **Key expiry as rotation:** 90 d reusable host/KVM keys, short-lived OAuth,
  one-shot team keys — expiry does the rotation work instead of calendar reminders.
