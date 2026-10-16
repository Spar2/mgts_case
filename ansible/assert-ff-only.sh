#!/usr/bin/env bash
# Если клон уже есть, HEAD должен быть предком origin/<ветка>.
# Иначе plain-деплой умрёт на git pull --ff-only. reset --hard не делаем.
# Нет .git — выходим: клон создаст следующий шаг.
set -euo pipefail
repo=${1:?repo}
branch=${2:?branch}
if [ ! -d "$repo/.git" ]; then
  echo "ff: no git, skip"
  exit 0
fi
git -C "$repo" fetch origin "$branch"
if git -C "$repo" merge-base --is-ancestor HEAD "origin/$branch"; then
  echo "ff: HEAD is an ancestor of origin/$branch"
  exit 0
fi
echo "ff: HEAD не предок origin/$branch. git pull --ff-only в plain-деплое не пройдёт. reset --hard не делаю." >&2
exit 1
