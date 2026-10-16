# QUARANTINE — gate-правило selfhosted-миграции

Боец F5 (relay-переключатель + QUARANTINE). Фаза 1: notify переведены на
relay-переключатель без даунтайма. Инлайн-notify и массовый перевод
`runs-on` — отложены (фаза 2, wave-2).

## Gate-правило (guard-ref)

Разрешённые долгоживущие ветки: `main`, `data`, `web`, `ml`, `devops`.
Любой другой `github.ref_name` — fail, кроме внутренних PR (`pull_request`,
не форк): `github.ref_name` там вида `N/merge`, проходят как доверенные.
Форки (`fork == 'true'`) — всегда fail, даже при совпадении имени ветки
с allowlist (чужую ветку можно назвать `data`): форкам на self-hosted нельзя.

Composite action: `.github/actions/guard-ref/action.yml`

```yaml
- name: Guard branch ref
  uses: ./.github/actions/guard-ref
  with:
    ref_name: ${{ github.ref_name }}
    event_name: ${{ github.event_name }}
    fork: ${{ github.event.pull_request.head.repo.fork }}
```

Логика — `bash`: `fork == 'true'` → всегда `exit 1`; иначе
`main|data|web|ml|devops` → pass, иначе при `event_name == pull_request`
(internal PR, ref вида `N/merge`) → pass, иначе `exit 1`.

## Relay-переключатель notify (фаза 1, без даунтайма)

Во всех 5 notify (`notify-data/ml/web/devops/runs.yml`) задан job-env:

```yaml
env:
  NOTIFY_VIA_RELAY: ${{ vars.NOTIFY_VIA_RELAY }}
  RELAY_URL: ${{ vars.RELAY_URL }}
```

- Пусто (по умолчанию) → legacy: прямой Discord-webhook как сейчас
  (`sarisia/actions-status-discord`, `ack_no_webhook: true`).
- Задано (`vars.NOTIFY_VIA_RELAY != ''` **и** `vars.RELAY_URL != ''`)
  → `POST $RELAY_URL` с `Authorization: Bearer $RELAY_TOKEN`
  (`secrets.RELAY_TOKEN`), тело — JSON через `jq`, `curl --fail`.
- Фолбэк без даунтайма: legacy-шаг идёт при
  `env.NOTIFY_VIA_RELAY == '' || env.RELAY_URL == ''`,
  relay-шаг — только когда заданы оба. Если `RELAY_URL` не задан,
  шлётся прямой webhook как раньше.

Контракт relay: `POST $RELAY_URL`, bearer из `secrets.RELAY_TOKEN`.

## Deferred-список (фаза 2, НЕ трогать в фазе 1)

1. `deploy-reusable.yml` — инлайн-шаг `Discord notify`
   (`if: always() && inputs.inline_notify`) — deferred, фаза 2.
2. `preview.yml` — инлайн-шаги `Discord notify (preview ready)` и
   `Discord notify (preview removed)` — deferred, фаза 2.
3. Wave-2 `runs-on` потребление guard — массовый перевод джобов на
   self-hosted + потребление `guard-ref` во всех workflows — deferred,
   wave-2 (пример задела: `pr-gate.yml validate`, `lint-data.yml lint`).
4. F3 review — комментарий-трэд бойца F3: ссылка добавляется бойцом F3
   в PR-обсуждении; этот документ фиксирует точку связи
   (F3-коммент: см. PR comments / fighter F3 thread — `F3-COMMENT-LINK: TBD by F3`).
   <!-- F3-COMMENT-LINK: TBD by F3 -->

## F6: покрытие wave-2 (platform-env) + FROZEN-вердикт

| Workflow / job | Вердикт | runs-on | guard-ref | Почему |
|---|---|---|---|---|
| `pr-gate.yml validate` | ЕДЕТ (задел wave-2, уже) | `[self-hosted, team-devops]` | да (после checkout) | PR-gate, внутренние PR `N/merge` проходят как доверенные |
| `pr-gate.yml platform-env` | ЕДЕТ (F6) | `[self-hosted, team-devops]` | да (после checkout, F6) | Тот же триггер `pull_request → main`; `check_platform_env.py` — чистый python+pyyaml, без секретов tailnet/KVM |
| `lint-data.yml lint` | ЕДЕТ (задел wave-2, уже) | self-hosted | да | Эталон потребления guard вне pr-gate |
| `preview.yml` (gate/preview/cleanup) | FROZEN — остаётся hosted | `ubuntu-latest` | НЕТ, не ставить | wake-chicken-egg: джоба будит Xeon через KVM, если он спит — на спящем Xeon будить некому; cleanup намеренно ungated (снос чужих preview при close форков) — guard с `fork == fail` его сломает; инлайн Discord-notify — deferred фаза 2 |
| `dev-clone.yml` (vars/upsert/cleanup) | FROZEN — остаётся hosted | `ubuntu-latest` | НЕТ, не ставить | wake-chicken-egg (тот же probe+wake через KVM); триггер `push → dev-*` вне guard-allowlist (`main/data/web/ml/devops`) — guard завалит все легитимные пуши; инлайн Discord-notify — deferred фаза 2 |
| `notify-*` (data/ml/web/devops/runs) | FROZEN — остаются hosted | `ubuntu-latest` | НЕТ, не ставить | curl-дёшево: один `jq`+`curl` без сборки, выгоды от self-hosted ноль; Discord-блок: relay-переключатель фазы 1 уже покрывает перевод без даунтайма; `notify-runs` на `workflow_run` — вешать его на self-hosted добавляет нагрузку раннеру ради уведомлений |

Итого guard-покрытие после F6: `pr-gate.yml` — 2/2 джобов
(`validate` + `platform-env`), эталон `lint-data.yml`, `web-checks.yml`
— см. `git grep guard-ref`. Всё остальное из таблицы — FROZEN по причинам
выше, не мигрировать без пересмотра gate-правила (dev-*) и wake-архитектуры.
