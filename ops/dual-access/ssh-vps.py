"""VPS SSH helper: quoteless exec (pipeline strips quotes).

Usage (no quotes anywhere):
  python ops/dual-access/ssh-vps.py root@77.91.114.128 echo VPS-SSH-OK
  python ops/dual-access/ssh-vps.py root@77.91.114.128 curl -sk URL
Remote part = argv[2:] joined by spaces. Key file on this admin box only.
"""
import subprocess
import sys

HOST = sys.argv[1] if len(sys.argv) > 1 else 'root@77.91.114.128'
REMOTE = ' '.join(sys.argv[2:]) or 'echo VPS-SSH-OK'
KEY = 'C:/Users/werd0/.ssh/dual-access-temp'

r = subprocess.run(
    ['ssh', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
     HOST, REMOTE],
    capture_output=True, text=True, timeout=180,
)
print(r.stdout, end='')
print(r.stderr, end='')
raise SystemExit(r.returncode)
