# Sol → Muse → Grok: рабочий порядок

## Зачем три роли

Sol изучает задачу, задаёт критерий результата и принимает изменения. Muse через OpenCode выполняет ограниченное поручение. Grok Build независимо проверяет предоставленный код. Решение о включении изменения принимает Sol после воспроизведения и тестов. Два одинаковых ответа моделей не заменяют проверку.

Это мост через CLI, а не встроенные субагенты Codex. Sol сам запускает команды в ходе активной сессии. Фоновая работа после завершения сессии и бесконечный автономный цикл не настроены.

## Проверенная конфигурация

- OpenCode CLI `opencode-ai@1.19.0`, отдельно от desktop-приложения.
- Worker: `opencode-go/muse-spark-1.3-contributor` через существующий вход OpenCode Go.
- Grok Build `1.0.46`, reviewer: `grok-4.7`, существующий вход grok.com.
- Python 3.12+; Git; npm. Ключи в репозиторий и чат не передавать.

OpenCode Go Contributor допускает использование запросов/ответов для обучения Meta. Поэтому в пилоте используется опубликованный код, без исходных приватных CSV, `.env`, заявок и журналов пользователей. Условия: [OpenCode Go](https://opencode.ai/docs/go/). Для приватной разработки сначала выбрать провайдера с подходящими условиями обработки данных.

## Установка и первый запуск

Команды ниже выполнять из checkout `web`. Вместо `PUBLISHED_SHA` взять проверенный опубликованный commit; `../agent-pilot-web` должен быть новым путём. Не копировать рабочую папку целиком.

```powershell
npm install --prefix .runtime/agent-tools --no-audit --no-fund opencode-ai@1.19.0
git worktree add --detach ../agent-pilot-web PUBLISHED_SHA
Copy-Item scripts/agents/opencode.audit.json ../agent-pilot-web/opencode.json
```

Если Go ещё не подключён, выполните `opencode auth login` через установленный CLI и выберите OpenCode Go; секрет вводится в CLI. Вход desktop и CLI может использовать общий локальный store, но это проверяется командой `auth list`, а не чтением файла ключей.

```powershell
$pilotCli = (Resolve-Path .runtime/agent-tools/node_modules/opencode-ai/bin/opencode.exe).Path
python scripts/agent-pilot.py muse --checkout ../agent-pilot-web --prompt scripts/agents/muse-smoke.txt --executable $pilotCli --output .runtime/agent-pilot/smoke.ndjson --timeout 120
python scripts/agent-pilot.py muse --checkout ../agent-pilot-web --prompt scripts/agents/muse-audit.txt --executable $pilotCli --output .runtime/agent-pilot/audit.ndjson --timeout 300
```

Windows-путь executable выше проверен на этой машине; на Linux использовать фактический бинарник установленного платформенного пакета. Runner передаёт аргументы без shell.

Ожидаемый контрольный ответ: `schema_version=dts-method-result-v1`, один ряд для `denoised`, путь `docs/web/RESULT-HANDOFF.md`. Сам факт успешного exit code недостаточен: runner требует завершённый ответ и проверяет неизменность байтов файлов. Конфигурация OpenCode должна точно совпадать с проверенным шаблоном.

## Независимая проверка Grok

```powershell
grok models
python scripts/agent-review-bundle.py --checkout ../agent-pilot-web --output .runtime/grok-review.txt
python scripts/agent-pilot.py grok --checkout ../agent-pilot-web --prompt .runtime/grok-review.txt --executable C:/Users/recos/.grok/bin/grok.exe --output .runtime/agent-pilot/grok.json --timeout 300
```

Путь Grok заменить своим. При отсутствии входа — `grok login`. Reviewer получает только выбранные опубликованные фрагменты, без ответа Muse: это уменьшает зависимость суждений. Инструменты отключены, число ходов ограничено тремя. Не отключать запреты ради зависшего запуска.

## Что сохраняется

В `.runtime/agent-pilot` остаются исходный ответ, stderr, извлечённый ответ `.answer.md` и `.proof.json`: commit, длительность, код выхода, fingerprint до/после, наличие завершённого ответа. `.runtime` игнорируется Git. Сырые ответы могут содержать исходный код и не являются публичным отчётом. Публикуется только проверенная сводка.

Таймаут — незавершённая задача, а не согласие или успешный аудит. Runner не повторяет запрос автоматически. Он не подтверждает стоимость, научную точность и отсутствие доступа вне процесса; запреты OpenCode — ограничения инструментов, а не отдельная OS-песочница. Для недоверенного приватного кода нужен отдельный контейнер/пользователь без секретов.

## Поручения на реализацию

Текущий шаблон намеренно допускает только аудит: `read/glob/grep`, запрет shell, редактирования, MCP и внешних путей. Он не предназначен для незаметного включения режима записи. Реализацию делать следующим отдельным циклом:

1. Sol фиксирует проблему, воспроизведение и проверяемый критерий; выбирает один модуль и отдельную ветку от проверенного SHA.
2. Для Muse создаётся отдельная конфигурация с записью только в перечисленные файлы. Команды установки, миграции, тестов и публикации остаются у Sol; доступ ко всему репозиторию не выдаётся автоматически.
3. Muse возвращает diff, объяснение и ограничения. Sol сверяет реальные изменения с поручением, самостоятельно запускает регрессионный тест и нужные проверки.
4. Grok видит diff и достаточный контекст; замечания снова воспроизводятся. Sol исправляет или отклоняет их с причиной.
5. Sol сохраняет проверенный commit. Push/merge/deploy — отдельные действия в рамках разрешения пользователя, worker их не выполняет.

На хакатоне ограничивать цикл временем и числом проверяемых задач. Первая цель — подключить настоящий результат Захара и показать воспроизводимый пользовательский путь. Новые функции без данных не должны вытеснять интеграцию и презентацию.

Документация CLI: [OpenCode agents](https://opencode.ai/docs/agents/), [permissions](https://opencode.ai/docs/permissions/), [Grok headless](https://docs.x.ai/build/cli/headless-scripting).
