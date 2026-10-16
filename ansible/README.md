# Ansible — staging-стенд haklolarp (изолированный, НЕ прод)

Разворачивает на целевой ВМ (Xeon, **Ubuntu**, будет позже): `tailscale`,
`cloudflared`, `docker`, клон `git@github.com:werd0y45/haklolarp.git`
(репо **приватное** — клон только по SSH через read-only deploy-key)
в `/opt/haklolarp-staging` + `docker compose up`.
Пуш в `main` = автодеплой через `.github/workflows/deploy-staging.yml`,
результаты ранов уходят в Discord через `.github/workflows/notify-runs.yml`.

Коннект только к Ubuntu-таргету. Отладочная ВМ (Arch) используется только
для `syntax-check`/`lint` через venv — плейбук на неё ничего не ставит.

## Быстрый старт (venv `~/.venv/ansible` уже создан на отладочной ВМ)

```bash
ssh gritsko@192.168.1.20
source ~/.venv/ansible/bin/activate
cd ~/haklolarp  # git clone https://github.com/werd0y45/haklolarp.git && git checkout devops
ansible-galaxy collection install -r ansible/requirements.yml
cd ansible

# 1. Проверка синтаксиса и инвентаря (без целевой ВМ):
ansible-playbook --syntax-check playbooks/site.yml
ansible-inventory -i inventory/hosts.yml --list

# 2. Когда Xeon-ВМ включена — поправь inventory/hosts.yml (xeon-staging),
#    затем прогон:
ansible-playbook -i inventory/hosts.yml playbooks/site.yml \
  -e "TAILSCALE_AUTH_KEY=$TAILSCALE_AUTH_KEY CLOUDFLARED_TUNNEL_TOKEN=$CLOUDFLARED_TUNNEL_TOKEN"
# Токены можно передать и через env: TAILSCALE_AUTH_KEY=... CLOUDFLARED_TUNNEL_TOKEN=...

# 3. Только обновить приложение (без переустановки софта):
ansible-playbook -i inventory/hosts.yml playbooks/deploy.yml
```

## Секреты GitHub Actions

| Секрет | Статус | Назначение |
|---|---|---|
| `DISCORD_WEBHOOK_DEVOPS/WEB/ML` | заданы | уведомления о пушах |
| `STAGING_SSH_KEY` | задан | приватный ключ GitHub -> staging |
| `STAGING_HOST` | задан | tailnet-IP Xeon |
| `STAGING_USER` | задан | пользователь на Xeon-ВМ |
| `TAILSCALE_AUTH_KEY` | задан | ключ коробки: `tailscale up` + переразвёртка новой ноды через Ansible, если текущая откиснет |
| `TAILSCALE_AUTH_KEY_RUNNERS` | удалён везде (секрет снесён, ключ отозван) | был reusable+ephemeral, заменён OAuth |
| `TS_OAUTH_CLIENT_ID` / `TS_OAUTH_SECRET` | заданы | OAuth-клиент CI (тег `tag:ci`, эфемерные ноды, короткоживущие креды) |

## Tailscale: ключ и ввод машины в tailnet

1. В админке tailnet: Settings -> Keys -> Generate auth key:
   reusable, желательно с тегом (например `tag:staging`), expiry по вкусу.
2. Кинь ключ мне (или сам): `gh secret set TAILSCALE_AUTH_KEY --body "..."`.
3. Ввожу Xeon в tailnet прогоном с дев-ВМ:
   `TAILSCALE_AUTH_KEY=... ansible-playbook -i inventory/hosts.yml playbooks/site.yml`
   (роль `tailscale` делает `tailscale up`, ключ в логах скрыт через `no_log`).
4. Узнаю tailnet-IP (`tailscale ip -4` / имя `xeon-staging`) и обновляю секрет:
   `gh secret set STAGING_HOST --body "<tailnet-ip>"`.
5. После этого пуш в `main` деплоится: раннер заходит в tailnet
   (`tailscale/github-action`) и идёт по SSH на tailnet-адрес.

На этом аплинке сессии падают без `TS_FORCE_NOISE_443=1` — роль `tailscale`
кладёт systemd drop-in (`force-noise-443.conf`, замена ручному
`S97tailscale-force443`), переживает обновления пакета.

## Tailscale: отдельный ключ для CI-раннеров

В админке tailnet: Settings -> Keys -> Generate auth key с галками:
reusable + ephemeral (+ expiry по вкусу, описание `github-runners`).
Ephemeral = каждый ран регистрируется новым узлом, который сам выпиливается
из tailnet после офлайна. Дальше секрет:
`gh secret set TAILSCALE_AUTH_KEY_RUNNERS` (спросит значение интерактивно).
Деплой (`deploy-staging.yml`) ходит в tailnet только этим ключом.

## KVM wake (NanoKVM-PCIe `kvm-2354`, `100.68.100.66`)

CI будит Xeon сам с github-hosted раннеров: job `wake` в каноне
`deploy-reusable.yml` (зовётся из `deploy-staging/web/ml`), инлайн в `preview`.
Раннер джойнится в tailnet
(`tailscale/github-action`, `tag:ci`) и идёт по tailnet-адресам из секретов:
probe `STAGING_HOST`, wake `KVM_HOST` (`100.68.100.66`). Зависимости
от self-hosted `run-devops` нет (он жил на самом Xeon и не мог разбудить
свой же хост). Ключ `KVM_WAKE_KEY` — forced command
(`restrict,command="/usr/local/bin/xeon-wake-only"`): выполняет только скрипт,
чужие команды игнорируются, reset/ISO/консоль недоступны даже при утечке.
Скрипт: up? → WoL (MAC `00:e0:26:5c:23:20`) → wait → короткий Power-press
(gpio503, никогда reset) → wait. Power OFF/reset из CI нет.
Скрипт и установщик лежат в `ansible/files/kvm/` (`xeon-wake-only`, `kvm_install.py`).
Прошивка KVM может снести `/usr/local/bin` — тогда переустановить плейбуком
`playbooks/kvm.yml` с машины с LAN-доступом к `192.168.1.4` (см. историю ранов wake).
Секреты: `TS_OAUTH_CLIENT_ID/SECRET`, `STAGING_HOST`, `KVM_HOST`, `KVM_WAKE_KEY`.
Имена `*.lan` (`xeon-staging.lan`, `kvm.lan` в `/etc/hosts`) — legacy для локалки,
CI их больше не использует.
Пароль root на KVM — длинный уникальный, дефолты `admin/admin root/root` недопустимы.
Этот же общий ключ используют CI-раннеры команд (web/ml).
Свои машины команды вводят одноразовыми ключами вручную:
`tailscale up --auth-key=<одноразовый> --hostname=<имя-машины>`.

Публичный ключ лежит в `ansible/files/staging_autodeploy.pub` — плейбук (`base`)
кладёт его в `authorized_keys` пользователя staging, приватный — только в секрете.

## Data-стенд

- Каталог `/opt/haklolarp-stack`, ветка `data`, проект `haklolarp-stack`
  (`stack/.env`). Это уже работающий стек, не второй ClickHouse.
- Job `prepare` в `.github/workflows/deploy-data.yml` гоняет
  `playbooks/data-stand.yml`: один раз отодвигает scp-дерево в
  `/opt/haklolarp-stack.prev`, кладёт клон, CSV возвращает в
  `/opt/haklolarp-stack/data`. Тома `haklolarp-stack` не удаляет.
  Нужен секрет `STAGING_BECOME_PASSWORD`.
- `docker compose up` делает следующий job — caller канона, `mode: plain`.
- Полный `site.yml` с GitHub-раннера не запускается (нет LAN; hardening
  ждёт `192.168.1.123` с control node).

## Командные стенды (web / ml)

- `/opt/haklolarp-web` — ветка `web`, проект `haklolarp-web`, порт `8081`,
  деплой через `.github/workflows/deploy-web.yml` (пуш в `web`).
- `/opt/haklolarp-ml` — ветка `ml`, проект `haklolarp-ml`, порт `8082`,
  деплой через `.github/workflows/deploy-ml.yml` (пуш в `ml`).
- Ansible эти каталоги не трогает — команды докручивают свои workflow
  агентами (в файлах есть TODO: compose-файл, тесты, self-hosted раннеры).
- Раннерам команд выдай доступ в tailnet (ключи), дальше они сами.

## Self-hosted runner / линтер (позже)

Роль под раннер пока не ставим: место зарезервировано. Когда понадобится —
добавим роль `runner` (отдельный пользователь `gha-runner`, свой workdir
`/opt/gha-runner`, labels `staging,lint`), чтобы CI-изоляция не трогала стенд.
