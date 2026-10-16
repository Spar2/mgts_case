# WAKE-FREEZE — заморозка KVM/WoL

Wake-блок заморожен. Не переносить, не рефакторить, не удалять.

## Rationale

- Xeon — always-on. Wake (probe → WoL → forced command `xeon-wake-only`) —
  страховочный контур на случай просадки питания, а не штатный путь включения.
- Wake не переносить: канон probe/wake живёт в `deploy-reusable.yml`,
  копии в `deploy-data.yml` / `preview.yml` держатся в синхроне с каноном
  и никуда не переезжают (self-hosted раннер wake не делает).
- `xeon-wake-only` — wake-only: только WoL и ATX short-press при доказанном
  down. Reset / long-press / ISO / консоль недоступны by design.

## Замороженные файлы (guard: `# FROZEN: KVM/WoL — не трогать (always-on)`)

1. `.github/workflows/deploy-reusable.yml` — job `wake` (канон probe/wake).
2. `.github/workflows/deploy-data.yml` — job `wake` (копия канона).
3. `.github/workflows/preview.yml` — инлайн wake в `preview` и `cleanup`.
4. `ansible/playbooks/kvm.yml` — установка wake-скрипта и forced-команды.
5. `ansible/files/kvm/xeon-wake-only` — wake-only скрипт (WoL → ATX short press).
6. `ansible/files/kvm/kvm_install.py` — установщик forced command.
7. `ansible/roles/base/tasks/main.yml` — WoL persistent (udev) + WoL сейчас.
8. `ansible/roles/base/defaults/main.yml` — `base_wol_macs`.
9. `ansible/inventory/group_vars/staging.yml` — `base_wol_macs` стенда.
