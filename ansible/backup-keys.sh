#!/usr/bin/env bash
# Бэкап ключевого материала админа (devops).
# Собирает приватные ключи из ~/.ssh в tar, шифрует age на публичный
# recipient из .age-recipient, кладёт в ~/haklolarp-keys/, проверяет
# расшифровкой. Секретов в скрипте нет.
# Использование: bash ansible/backup-keys.sh
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
RECIPIENT="$(cat "$SCRIPT_DIR/.age-recipient")"
BACKUP_DIR="$HOME/haklolarp-keys"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

KEYS=(xeon_staging haklolarp_deploy runner_admin kvm_wake)
DATE="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BACKUP_DIR/keys-backup-$DATE.tar.gz.age"

mkdir -p "$BACKUP_DIR"
mkdir -p "$STAGE/keys"
for k in "${KEYS[@]}"; do
  cp "$HOME/.ssh/$k" "$HOME/.ssh/$k.pub" "$STAGE/keys/"
done

{
  echo "haklolarp keys backup $DATE (UTC)"
  echo "recipient: $RECIPIENT"
  echo ""
  for k in "${KEYS[@]}"; do
    echo "$k: $(ssh-keygen -lf "$STAGE/keys/$k.pub")"
  done
} > "$STAGE/keys/MANIFEST.txt"

tar -czf "$STAGE/keys.tar.gz" -C "$STAGE" keys
if command -v age >/dev/null 2>&1; then
  AGE_BIN=age
elif [ -x "$HOME/.local/bin/age" ]; then
  AGE_BIN="$HOME/.local/bin/age"
else
  echo 'age не найден: релиз с github.com/FiloSottile/age' >&2
  exit 1
fi
"$AGE_BIN" -r "$RECIPIENT" -o "$OUT" "$STAGE/keys.tar.gz"

# Проверка: расшифровать во временный каталог и сверить список файлов.
if [ -f "$HOME/.config/age/backup.key" ]; then
  "$AGE_BIN" -d -i "$HOME/.config/age/backup.key" -o "$STAGE/keys-check.tar.gz" "$OUT"
  tar -tzf "$STAGE/keys-check.tar.gz" | sort > "$STAGE/list-check.txt"
  tar -tzf "$STAGE/keys.tar.gz" | sort > "$STAGE/list-orig.txt"
  if cmp -s "$STAGE/list-check.txt" "$STAGE/list-orig.txt"; then
    echo "VERIFY_OK $OUT"
  else
    echo 'расшифровка не совпала!' >&2
    exit 1
  fi
else
  echo "без приватного ключа проверка пропущена: $OUT"
fi

# Ротация: держать последние 5 бандлов.
ls -t "$BACKUP_DIR"/keys-backup-*.tar.gz.age | tail -n +6 | xargs -r rm -f
echo "BACKUP_OK $OUT"
