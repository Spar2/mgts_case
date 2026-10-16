# Команды (ветки `ml` / `web`) — микрогайд

## Замысел владельца (что строим)
Каждая команда деплоит **свою ветку** в **свой изолированный каталог** staging-машины
(Xeon, Ubuntu, в tailnet). Main-стенд (`/opt/haklolarp-staging`, порт 8080) — не трогаем.
CI-раннеры ходят в tailnet по общему ephemeral-ключу и деплоят по SSH.
Результаты пушей и деплоев падают в командный Discord (`DISCORD_WEBHOOK_ML`, `DISCORD_WEBHOOK_WEB`).
Пока владелец спит — доводишь workflow своим агентом по образцу девопса
(`.github/workflows/deploy-staging.yml`).

## Что тебе уже дано
| Что | ML | Web |
|---|---|---|
| Ветка | `ml` | `web` |
| Каталог на стенде | `/opt/haklolarp-ml` | `/opt/haklolarp-web` |
| Compose-проект | `haklolarp-ml` | `haklolarp-web` |
| Порт | `8082` (8080 занят main-стендом, 8081 — за фронтом) | `8081` (8080 занят main-стендом, 8082 — за ML) |
| Deploy-заготовка | `.github/workflows/deploy-ml.yml` (пуш в `ml` = деплой) | `.github/workflows/deploy-web.yml` (пуш в `web` = деплой) |
| Discord | секрет `DISCORD_WEBHOOK_ML` | секрет `DISCORD_WEBHOOK_WEB` |
| Образец | `.github/workflows/deploy-staging.yml` | `.github/workflows/deploy-staging.yml` |

Доступы (выдаёт владелец): collaborator в репо + tailnet (одноразовый ключ:
`tailscale up --auth-key=<ключ> --hostname=<твоя-машина>`).

Учти про железо: Xeon E5-2696 v4 **без GPU** — только CPU-инференс/лёгкие сервисы,
тяжёлое обучение здесь не гоняем.

## Статус агента (2026-10-05)
1. `compose.ml.yaml` + `Dockerfile.ml` — FastAPI `haklolarp-ml-api`, порт `${APP_PORT:-8082}`.
2. `deploy-ml.yml`: pytest на GHA, на стенде `docker compose up --build` (без pull из registry).
3. Пайплайн: `fiber/` ingestion + denoise zoo. CSV/веса в git не входят.
4. Пуш в `ml` ещё нужно проверить по Discord и `curl :8082/health`.
5. Volume под модели/кеш — следующим шагом, когда появятся веса CNN.

## Задачи твоему агенту (ml)
1. Создай в корне репо `compose.ml.yaml`: сервис(ы) ML, проброс `8082`,
   `container_name` с префиксом `haklolarp-ml-`, `restart: unless-stopped`,
   `timeout-minutes` на долгие шаги с запасом.
2. В `deploy-ml.yml` раскомментируй compose-блок (pull/up с `-p haklolarp-ml -f compose.ml.yaml`).
3. Добавь шаги под ваш стек **до** деплоя: install → lint → test (лёгкие, без обучения).
   Долгие сборки кешируй (Docker layer cache / GHA cache), иначе ран будет висеть.
4. Запушь в `ml`: проверь Discord (пуш + деплой) и контейнер
   (`docker ps`, `curl http://127.0.0.1:8082/health` only from Xeon; tailnet `curl http://100.66.240.22:8082` NOT WORKING till rebind (loopback-only 127.0.0.1)).
5. Опционально: свой `TAILSCALE_AUTH_KEY_ML`, self-hosted runner, volume под
   модели/кеш (веса и датасеты в git не коммитить — только код).

## Задачи твоему агенту (web)
1. Создай в корне репо `compose.web.yaml`: сервис(ы) фронта, проброс `8081`,
   `container_name` с префиксом `haklolarp-web-`, `restart: unless-stopped`.
2. В `deploy-web.yml` раскомментируй compose-блок (pull/up с `-p haklolarp-web -f compose.web.yaml`).
3. Добавь шаги под ваш стек **до** деплоя: install → lint → test → build
   (образец шагов смотри в любом Node-проекте; падать ран должен до SSH).
4. Запушь в `web`: проверь Discord (пуш + деплой) и контейнер
   (`docker ps`, `curl http://127.0.0.1:8081/` only from Xeon; tailnet `curl http://100.66.240.22:8081` NOT WORKING till rebind (loopback-only 127.0.0.1)).
5. Опционально: свой `TAILSCALE_AUTH_KEY_WEB`, self-hosted runner, healthcheck в compose.

## Как везти изменения в main (PR-flow)
1. Работаешь в `ml`/`web`, пуш — стенд `/opt/haklolarp-ml` или `/opt/haklolarp-web` обновляется сам.
2. Открываешь PR в `main`: стартует gate (`validate`: yaml/compose/lint) и
   preview-окружение `/opt/haklolarp-previews/pr-N` на порту `9000+N`.
   Порты в compose публикуй через `${APP_PORT:-8082}` (ml) / `${APP_PORT:-8081}` (web)
   (как в корневом `compose.yaml`).
3. Gate зелёный + preview проверен → владелец мёржит (авто-блока нет:
   branch protection на приватном репо требует Pro, мёрж красных PR —
   на совести владельца) → `main` едет на staging.
4. Закрытие/мерж PR сносит preview автоматически.

## Правила
- Порты и `container_name`/`project` — только свои, чужие не занимать.
- Веса моделей, датасеты, кеши — в тома/кеш CI, не в git и не в образ без нужды.
- Тяжёлое (node_modules, dist) — в Docker-слоях/кеше CI, не в git.
- В `main` — только через PR владельца.
