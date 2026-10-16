#!/usr/bin/env bash
# Drives ansible/adopt-haklolarp-stack.sh with a stub docker and a temp root.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
script="$here/adopt-haklolarp-stack.sh"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/bin"
cat > "$work/bin/docker" << 'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$work/bin/docker"
export PATH="$work/bin:$PATH"
phrase='Ничего не перекладываю'

fail() { echo "FAIL $*" >&2; exit 1; }

run() {
  set +e
  out=$(HAKLO_STACK_ROOT="$work/stack" "$script" 2>&1)
  rc=$?
  set -e
  printf '%s\n' "$out"
  return "$rc"
}

rm -rf "$work/stack" "$work/stack.prev"
mkdir -p "$work/stack.prev"
if out=$(run); then
  printf '%s\n' "$out" | grep -F "$phrase" && fail "A-missing hard stop"
  echo "ok A-missing"
else
  fail "A-missing rc"
fi

rm -rf "$work/stack"
mkdir -p "$work/stack" "$work/stack.prev"
if out=$(run); then
  printf '%s\n' "$out" | grep -F "$phrase" && fail "A-empty hard stop"
  echo "ok A-empty"
else
  fail "A-empty rc"
fi

rm -rf "$work/stack"
mkdir -p "$work/stack.prev" "$work/stack"
echo keep > "$work/stack/keep.txt"
set +e
out=$(run)
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  fail "B-nonempty should fail"
fi
if [ "$(cat "$work/stack/keep.txt")" != "keep" ]; then
  fail "B-nonempty mutated the live dir"
fi
echo "ok B-nonempty"
