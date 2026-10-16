#!/usr/bin/env bash
# Гоняет ansible/assert-ff-only.sh на локальных git-репозиториях, без SSH.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
script="$here/assert-ff-only.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

git -C "$work" init --bare origin.git >/dev/null
git init "$work/seed" >/dev/null
git -C "$work/seed" config user.email test@example.com
git -C "$work/seed" config user.name test
echo base > "$work/seed/file"
git -C "$work/seed" add file
git -C "$work/seed" commit -m base >/dev/null
git -C "$work/seed" branch -M data
git -C "$work/seed" remote add origin "$work/origin.git"
git -C "$work/seed" push -u origin data >/dev/null
git -C "$work/origin.git" symbolic-ref HEAD refs/heads/data
rm -rf "$work/live"
git clone "$work/origin.git" "$work/live" >/dev/null
git -C "$work/live" checkout data >/dev/null
git -C "$work/live" config user.email test@example.com
git -C "$work/live" config user.name test

if "$script" "$work/live" data >/dev/null; then
  echo "ok ancestor"
else
  echo "FAIL ancestor should pass" >&2
  exit 1
fi

echo extra > "$work/live/extra"
git -C "$work/live" add extra
git -C "$work/live" commit -m extra >/dev/null
if "$script" "$work/live" data >/dev/null; then
  echo "FAIL local commit should fail" >&2
  exit 1
fi
echo "ok local commit rejected"

rm -rf "$work/empty"
mkdir -p "$work/empty"
if "$script" "$work/empty" data >/dev/null; then
  echo "ok no git skipped"
else
  echo "FAIL missing git should skip" >&2
  exit 1
fi
