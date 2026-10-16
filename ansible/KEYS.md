# Ключи и секреты: инвентарь, бэкапы, ротация

ПРИВАТНЫХ КЛЮЧЕЙ И ТОКЕНОВ В ЭТОМ ФАЙЛЕ НЕТ. Только карта: что где лежит
и как восстанавливать/ротировать. Сам бэкап: `bash ansible/backup-keys.sh`
→ `~/haklolarp-keys/keys-backup-*.tar.gz.age` (age, recipient в
`ansible/.age-recipient`). Приватный ключ age (`~/.config/age/backup.key`
на дев-ВМ) хранится ОФЛАЙН у админа — без него бандл не открыть.

## Приватные ключи (входят в age-бандл)

| Ключ | Назначение | Где живёт | Ротация при утечке |
|---|---|---|---|
| `xeon_staging` | админ → Xeon (ansible/ssh) | дев-ВМ `~/.ssh`, authorized_keys `hakloboss` на Xeon | новый ключ → `authorized_keys` на Xeon → обновить бандл |
| `haklolarp_deploy` | Xeon → GitHub read-only deploy-key | дев-ВМ `~/.ssh`, `/home/hakloboss/.ssh/` на Xeon, паблик в Deploy Keys репо | новый ключ → Deploy Keys в GitHub → разложить ролью `app` → бандл |
| `runner_admin` | админ → ВМ раннеров | дев-ВМ `~/.ssh`, cloud-init ВМ | новый ключ → пересоздать ВМ ролью `runner_vm` → бандл |
| `kvm_wake` | CI/dev → KVM forced command | дев-ВМ `~/.ssh`, секрет `KVM_WAKE_KEY`, `/root/.ssh/authorized_keys` на KVM | новый ключ → KVM (`kvm_install.py`) + секрет → бандл |

## Только в GitHub Secrets (api не отдаёт — восстановить НЕЛЬЗЯ, только пересоздать)

- `STAGING_SSH_KEY` (приват; паблик — в `authorized_keys` hakloboss): утечка = новый ключ → секрет + authorized_keys.
- `STAGING_BECOME_PASSWORD` (sudo `hakloboss` на Xeon): job `prepare` в `deploy-data.yml`. В репо нет. Утечка = сменить пароль на хосте и обновить секрет.
- `TS_OAUTH_CLIENT_ID` / `TS_OAUTH_SECRET`: пересоздать OAuth-клиент в tailnet-админке → секреты.
- `TAILSCALE_AUTH_KEY`: новый ключ в админке → секрет + `tailscale up` на Xeon при нужде.
- `DISCORD_WEBHOOK_DEVOPS/WEB/ML/DATA`: новые вебхуки в Discord → секреты.
- `KVM_WAKE_KEY`: копия есть в бандле (см. выше).

Владелец обязан иметь копии всего раздела «только GitHub» вне GitHub
(менеджер паролей). Проверка раз в квартал: `gh secret list` сверяется с этой таблицей.

## Восстановление из бандла

```bash
age -d -i /путь/к/backup.key -o keys.tar.gz keys-backup-<дата>.tar.gz.age
tar -xzf keys.tar.gz   # MANIFEST.txt внутри: сверь отпечатки (ssh-keygen -lf)
```

## Пароли (нигде не хранятся)

- `hakloboss`/sudo на Xeon, `root` на KVM, `gritsko` на дев-ВМ — в голове
  админа. Смена пароля sudo/root не требует правок в репо (become — через
  `/tmp/become_pw`, shred после прогона).
