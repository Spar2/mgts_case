#!/usr/bin/env bash
# Dual-access monitor Ф13: health обоих входов + parity; 2 фейла подряд -> Discord.
# Webhook только из EnvironmentFile 0600, в репо его нет.
set -eu
S=/tmp/monitor-dual-access.fails
F=$(cat "$S" 2>/dev/null || echo 0)
TCODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$TAILNET_BASE/" 2>/dev/null) || TCODE=000
P=$(curl -fsS --max-time 10 "$PUBLIC_BASE/api/v1/health" 2>/dev/null) || P=""
if [ "$TCODE" = 200 ] && [ -n "$P" ]; then echo 0 >"$S"; exit 0; fi
F=$((F + 1)); echo "$F" >"$S"
if [ "$F" -ge 2 ]; then
  echo 0 >"$S"
  if [ -n "${MONITOR_WEBHOOK:-}" ]; then
    curl -fsS --max-time 10 -X POST -H 'Content-Type: application/json' \
      -d '{"content":"dual-access FAIL"}' "$MONITOR_WEBHOOK" || echo 'dual-access FAIL (webhook POST failed)' >&2
  else
    echo 'dual-access FAIL (no webhook; logged only)' >&2
  fi
fi
