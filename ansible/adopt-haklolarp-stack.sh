#!/usr/bin/env bash
# Один раз: scp-каталог /opt/haklolarp-stack уступает место git-клону ветки data,
# тома проекта haklolarp-stack не удаляются. Пустой проект haklolarp-data
# (неудачный второй стек) снимается только если его тома нулевого размера.
# Повторный запуск каталог не перекладывает и down живого стека не делает.
# Аргумент restore — после клона скопировать CSV из .prev в DATA_DIR.
set -euo pipefail

# HAKLO_STACK_ROOT allows the same branches to run against a temp directory.
STACK="${HAKLO_STACK_ROOT:-/opt/haklolarp-stack}"
PREV="${STACK}.prev"
CSV_DEST="${STACK}/data"

# На Xeon у docker volume inspect нет UsageData.Size. Пустой том — ноль файлов.
# Образ берём локальный, чтобы не тянуть новый.
volume_file_count() {
  local vol="$1" img count
  img=$(docker inspect -f '{{.Image}}' haklolarp-stack-clickhouse-1 2>/dev/null || true)
  if [ -z "$img" ]; then
    img=$(docker images -q 'clickhouse/clickhouse-server' | head -n 1)
  fi
  if [ -z "$img" ]; then
    echo 'adopt: нет локального образа clickhouse, размер тома не проверить.' >&2
    exit 1
  fi
  count=$(docker run --rm --network none --entrypoint /bin/sh -v "${vol}:/v:ro" "$img" -c 'find /v -type f | wc -l')
  printf '%s' "$count" | tr -d '[:space:]'
}

assert_port_owner() {
  local ids id proj ports
  ids=$(docker ps -q --filter publish=8123)
  ports=$(docker ps --format '{{.Ports}}')
  if printf '%s\n' "$ports" | grep -F '8123' >/dev/null; then
    if [ -z "$ids" ]; then
      echo 'adopt: 8123 виден в docker ps, фильтр publish=8123 пуст. Ничего не остановлено.' >&2
      exit 1
    fi
  fi
  # shellcheck disable=SC2086 # id из docker ps -q, пробелов внутри нет
  for id in $ids; do
    proj=$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$id")
    echo "adopt: 8123 container ${id} project=${proj}"
    if [ "$proj" != "haklolarp-stack" ]; then
      echo "adopt: 127.0.0.1:8123 занят проектом '${proj}', ожидался haklolarp-stack. Ничего не остановлено." >&2
      exit 1
    fi
  done
  # ids — вывод docker ps -q, по одному id в строке, пробелов внутри нет.
}

drop_empty_side_project() {
  local vols vol size ids
  vols=$(docker volume ls -q --filter name=haklolarp-data)
  ids=$(docker ps -aq --filter label=com.docker.compose.project=haklolarp-data)
  if [ -z "$vols" ] && [ -z "$ids" ]; then
    echo 'adopt: проект haklolarp-data уже отсутствует'
    return 0
  fi
  # shellcheck disable=SC2086 # имя тома из docker volume ls -q
  for vol in $vols; do
    size=$(volume_file_count "$vol")
    echo "adopt: volume ${vol} files=${size}"
    if [ "$size" != "0" ]; then
      echo "adopt: том ${vol} содержит ${size} файлов, down --volumes не делаю." >&2
      exit 1
    fi
  done
  if [ -f /opt/haklolarp-data/stack/compose.yaml ]; then
    docker compose -p haklolarp-data -f /opt/haklolarp-data/stack/compose.yaml down --volumes --remove-orphans
  else
    if [ -n "$ids" ]; then
      # shellcheck disable=SC2086
      docker rm -f $ids
    fi
    if [ -n "$vols" ]; then
      # shellcheck disable=SC2086
      docker volume rm $vols
    fi
  fi
  echo 'adopt: пустой проект haklolarp-data снят'
}

restore_csv() {
  local owner="${1:-}"
  local src="${PREV}/data"
  if [ ! -d "$src" ]; then
    echo 'adopt: CSV в prev нет, копирование пропускаю'
    exit 0
  fi
  mkdir -p "$CSV_DEST"
  if [ -n "$(find "$CSV_DEST" -type f -print -quit)" ]; then
    echo 'adopt: CSV в DATA_DIR уже есть, повторно не копирую'
    exit 0
  fi
  cp -a "${src}/." "$CSV_DEST/"
  if [ -n "$owner" ]; then
    chown -R "${owner}:${owner}" "$CSV_DEST"
  fi
  echo "adopt: CSV скопированы в ${CSV_DEST}"
}

if [ "${1:-}" = "restore" ]; then
  restore_csv "${2:-}"
  exit 0
fi

echo '== docker ps =='
docker ps -a --format 'table {{.Names}}\t{{.Ports}}\t{{.Status}}\t{{.Label "com.docker.compose.project"}}'
echo '== docker volume ls =='
docker volume ls

assert_port_owner
drop_empty_side_project

if [ -d "${STACK}/.git" ]; then
  echo "adopt: ${STACK} уже git, каталог не перекладываю"
  exit 0
fi

# Empty or missing directory: .prev may already hold the scp tree.
# The clone fills the path. A non-empty non-git directory next to .prev is left alone.
if [ ! -d "$STACK" ] || [ -z "$(find "$STACK" -mindepth 1 -print -quit)" ]; then
  echo "adopt: ${STACK} пуст или отсутствует, его заполнит клон"
  exit 0
fi

if [ -e "$PREV" ]; then
  echo "adopt: ${PREV} уже есть, а текущий каталог без .git. Ничего не перекладываю." >&2
  exit 1
fi

if [ -f "${STACK}/compose.yaml" ]; then
  docker compose -p haklolarp-stack -f "${STACK}/compose.yaml" stop
fi
ids=$(docker ps -q --filter label=com.docker.compose.project=haklolarp-stack)
if [ -n "$ids" ]; then
  # shellcheck disable=SC2086
  docker stop $ids
fi
mv "$STACK" "$PREV"
echo "adopt: scp-дерево сохранено в ${PREV}, тома не удалялись"
