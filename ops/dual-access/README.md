# dual-access — VPS edge для xn--80aubodcdd.xn--p1ai (ларпкорп.рф)

Origin: `https://<XEON-TAILNET-IP>:8443` (плейсхолдер; реальный IP только в рантайме).
Фаза плана: Ф3 (vps-notes/phases-dual-access.md).

```sh
docker run -d --name caddy --restart unless-stopped -p 80:80 -p 443:443 -p 443:443/udp \
  -v $PWD/ops/dual-access/Caddyfile:/etc/caddy/Caddyfile:ro \
  -v caddy_data:/data -v caddy_config:/config caddy:2-alpine
```

LE: `docker logs -f caddy` → жди `certificate obtained successfully`.
Проверка: `curl -sI https://xn--80aubodcdd.xn--p1ai | head -5`
Parity: `diff <(curl -sk https://xn--80aubodcdd.xn--p1ai/) <(curl -sk https://<XEON-TAILNET-IP>:8443/)`
Откат: `docker stop caddy && docker rm caddy` (DNS/origin без изменений; серты остаются в `caddy_data`).
Формат: `caddy fmt` правит только табы/скобки, не валидирует проксирование.
