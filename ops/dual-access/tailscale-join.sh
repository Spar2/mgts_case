#!/usr/bin/env bash
# Tailscale join VPS (Ф5): ключ только из env или файла, не из репо/команд.
#   TAILSCALE_AUTH_KEY=<key> ./ops/dual-access/tailscale-join.sh
#   ./ops/dual-access/tailscale-join.sh /path/to/keyfile
set -eu
KEY="${TAILSCALE_AUTH_KEY:-}"
[ -z "$KEY" ] && [ $# -ge 1 ] && KEY="$(cat "$1")"
[ -z "$KEY" ] && { echo "need TAILSCALE_AUTH_KEY or key file" >&2; exit 1; }
set +x
tailscale up --auth-key="$KEY"
tailscale status
KEY=""; unset KEY
