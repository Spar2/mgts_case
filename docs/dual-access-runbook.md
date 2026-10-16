# Dual-access runbook

Схема утверждена: сайт на Xeon staging → туннель → VPS → интернет.
Домен `ларпкорп.рф` (`xn--80aubodcdd.xn--p1ai`), DNS у Selectel.
Binds: всё loopback, кроме allowlist `compose.yaml`. UFW: 22, 80, 443, 8080
+ `tailscale0` (8080 KEEP — staging stub жив; fail2ban только sshd).

## 1. Тракт (победитель арены: tailscale)

VPS вступает в tailnet (`tailscale up`, роль уже есть), Caddy на VPS
терминирует LE-TLS (HTTP-01, порт 80) и проксирует на Xeon по tailnet.
WireGuard-точка-точка отклонён (нет роли, дороже), cloudflared —
только запасной БЕЗ VPS (выпадает из тракта, режимы взаимоисключающие).

Зазор: `:8081`/`:8443` на Xeon — loopback, по tailnet недоступны;
перебинд ломает loopback-lint гейта, `tailscale serve` запрещён
(`pr-gate.yml`). Закрывается хост-прокси на Xeon (вне compose):

```sh
# Xeon: слушать tailnet-IP, лить на loopback production-nginx (systemd-unit).
# Строго bind=<xeon-tailnet-ip>, НЕ 0.0.0.0. Чек: `ss -tlnp | grep 8443` без 0.0.0.0:8443.
socat TCP-LISTEN:8443,bind=<xeon-tailnet-ip>,fork,reuseaddr TCP:127.0.0.1:8443
```

Caddyfile на VPS (punycode, не unicode). Запуск — `docker run`, не compose
(команда целиком в `ops/dual-access/README.md`):

```caddyfile
xn--80aubodcdd.xn--p1ai {
    reverse_proxy https://<xeon-tailnet-ip>:8443 {
        transport http {
            tls_insecure_skip_verify
        }
    }
}
```

Почему 8443, а не 8081: production-nginx держит host/origin-чеки,
лимиты тел и `/api`→`api:8091` (тот же апстрим, что у 8081, но с закалкой).
Без локальных сертов — fallback `reverse_proxy http://<xeon-tailnet-ip>:8081`.

Второй домен `larpcorp.ru` (+ `www.larpcorp.ru` → 301): отдельный Caddy-блок
на тот же upstream с `header_up Host {host}`; nginx `server_name` и
api `ALLOWED_*` расширены через `DTS_PUBLIC_EXTRA_HOSTS/ORIGIN`
(compose.production.yaml:17-18, Xeon `.env`); LE на оба имени auto.
Канон — `ops/dual-access/Caddyfile`.

## 2. DNS в Selectel (прямой режим)

- `A @ → <VPS-IP>` (apex голого домена).
- `A www` KEEP (Ф11, www остаётся).
- CNAME не создавать (он только для tunnel-режима).
- TTL 300 на переезд, после стабилизации — 3600; NS/MX не трогать.
- Откат — вернуть A из истории зоны Selectel.

Чек: `dig @8.8.8.8 +short xn--80aubodcdd.xn--p1ai A` и `dig @1.1.1.1 +short xn--80aubodcdd.xn--p1ai A` = IP VPS;
`curl -v https://ларпкорп.рф/ | grep -i issuer` → Let's Encrypt.

## 3. Smoke (оба пути, хост)

```sh
T=http://<xeon-tailnet-ip>:8080 P=https://ларпкорп.рф
curl -sS -o /dev/null -w "%{http_code}\n" $T/api/v1/health
curl -sS -o /dev/null -w "%{http_code}\n" $P/api/v1/health
diff <(curl -sS $T/api/v1/health) <(curl -sS $P/api/v1/health) && echo PARITY-OK
curl -sSI http://ларпкорп.рф | head -1
```

Жди: `200` с обоих входов, parity diff пуст, `http -> 308 https`.

## 4. Откат

```sh
docker stop caddy && docker rm caddy  # на VPS (запуск тоже docker-run, не compose)
systemctl stop xeon-proxy-8443    # хост-прокси на Xeon (юнит в ops/dual-access/)
ufw delete allow 80/tcp; ufw delete allow 443/tcp
```

Tailnet (`:8080`) остаётся, public закрыт.

## 5. Merge/deploy

Deploy ветки отдельно не нужен: PR в main → staging plain
(`deploy-staging.yml`). Чек: FF (`merge-base --is-ancestor`),
зелёный `validate`, прямой пуш в main запрещён (RULES.md п.5).

## 6. Tailnet-сторона (вердикт арены)

По умолчанию STUB-ONLY (`:8080` уже доступен с tailnet).
+web/api из tailnet — только хост-прокси tailnet-IP→loopback
или `ssh -L`, compose не трогать.

## 7. Отложенное

Preview на `compose.web.yaml` отложен: меняет общий путь всех PR,
решает команда. Полный QA-journey — на хосте с docker
(`scripts/qa-public-journey.js` требует TLS-стенд на `:8443`;
guard крутится по `QA_BASES`, дефолт без ослабления).
Проверка тракта до домена (Ф5): `tailscale ping` loss 0%, curl через Caddy
по IP с Host-хаком, ребут Xeon <5 мин самовосстановления.
Мониторинг (Ф13): юниты в `ops/dual-access/monitor-*`, timer 90с,
webhook только в EnvironmentFile 0600. www (Ф11): KEEP, влит в
`ops/dual-access/Caddyfile`. cert-renewal: SKIP (LE свежий до 01.2027, auto-renew в Caddy).

## 8. SSH-аудит VPS

```sh
sshd -T | grep -Ei '^(port 22|permitrootlogin no|passwordauthentication no)'
stat -c '%a %n' ~/.ssh ~/.ssh/authorized_keys  # жди 700 / 600
fail2ban-client status sshd  # enabled, maxretry 5, bantime 1h
```
- Порт 22 НЕ менять — ломает `firewall.yml` (`ufw allow 22/tcp`) и `wait_for: port: 22`.
- Только ключи: `PermitRootLogin no`, `PasswordAuthentication no`; проверки из живой сессии, вторую не закрывать.
Откат: lockout → только консоль хостера.
