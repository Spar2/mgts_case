# Promotion pilot → prod: gates & contracts

One active promotion at a time (lock). No parallel runs.

## Lock
- `promote.lock`: holder, epoch (monotonic), TTL 30 min + heartbeat.
- Expiry = auto-release. Break-glass only with audit record
  (who/when/reason) + post-review. Fencing: stale epoch ignored.

## Gates (stop on first FAIL, fail-closed with journal entry)
- `G0 pre-check`: V3 GO confirmed, pilot healthy, lock free, target prod
  empty/ready. Exit 0 = proceed; else abort.
- `G1 snapshot`: snapshot pilot (app + both DB contours). Exit 0 iff
  checksum present, size > 0, manifest complete, stubs excluded.
- `G2 restore-to-empty`: restore only into empty prod env (emptiness
  verified before start). Exit 0 iff post-restore integrity == G1 manifest.
- `G3 migrate`: migrations for both DBs in order, no skips. Exit 0 iff
  `migrate --check` clean and both DB versions match expected.
- `G4 verify`: preprod config equals prod sample (except env);
  bind loopback unchanged; subnet `172.30.89.0/24` conflict-free;
  health/readiness OK; dual-DB read/write probe OK. Exit 0 iff all green.
- `G5 promote/switch`: only after G4 green. Exit 0 iff smoke 200 and
  version matches.
- `G6 post`: lock released, snapshot + manifest archived. Exit 0 iff done.

## Dual-DB
- Snapshot/restore/migrate both contours atomically per step.
- Single outbox table + relay; order key + idempotent consumer
  (dedupe by event-id). Version skew = FAIL the step, rollback to G1.
- Required test: order under retries, no dupes/loss, one DB down.

## Env secrets
- Only from prod environment, never files/snapshots; substitution on
  G2/G3 without echo to log. Missing key = FAIL G2 (no defaults).

## Bind
- Loopback bindings immutable; G4 diffs pilot vs prod bind config —
  any difference = FAIL.

## Rollback & quarantine
- Any FAIL: stop chain, prod to quarantine. Quarantine has TTL
  (auto-exit: drop/return, no sticky). Re-promote only manually with
  audit, never auto-retry over dirt, never auto-release by timer.
