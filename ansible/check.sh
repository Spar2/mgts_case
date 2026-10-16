#!/usr/bin/env bash
# Локальный гейт перед пушем: как CI (ansible-lint.yml),
# плюс syntax-check плейбуков и установка пинованных коллекций.
# NOTE: data-stand.yml покрыт везде — здесь, в ansible-lint.yml и в pr-gate.yml; локальный гейт = CI + syntax-check.
# Зелёный check.sh = можно пушить. Красный = чиним локально, в ветку не пушим.
set -euo pipefail
cd "$(dirname "$0")"
VENV_BIN="${ANSIBLE_VENV_BIN:-$HOME/.venv/ansible/bin}"
export PATH="$VENV_BIN:$PATH"
echo '== collections (pinned) =='
ansible-galaxy collection install -r requirements.yml
echo '== syntax-check =='
for play in playbooks/site.yml playbooks/deploy.yml playbooks/runners.yml playbooks/kvm.yml playbooks/data-stand.yml; do
  ansible-playbook --syntax-check "$play"
done
echo '== ansible-lint =='
for play in playbooks/site.yml playbooks/deploy.yml playbooks/runners.yml playbooks/kvm.yml playbooks/data-stand.yml; do
  ansible-lint --offline "$play"
done
echo '== shellcheck =='
if command -v shellcheck >/dev/null 2>&1; then
  SHELLCHECK_BIN=shellcheck
elif [ -x "$HOME/.local/bin/shellcheck" ]; then
  SHELLCHECK_BIN="$HOME/.local/bin/shellcheck"
else
  echo 'shellcheck не найден: pacman -S shellcheck или релиз с github.com/koalaman/shellcheck' >&2
  exit 1
fi
"$SHELLCHECK_BIN" -S warning ./*.sh
echo ALL_GREEN
