"""Xeon SSH helper: quoteless exec (pipeline strips quotes).

Usage (no quotes anywhere; password only via argv, never on disk):
  python ops/dual-access/ssh-xeon.py 192.168.1.123 USER PASS echo XEON-SSH-OK
Remote part = argv[4:] joined by spaces. Requires paramiko on admin box.
"""
import sys

import paramiko

HOST, USER, PWD = sys.argv[1], sys.argv[2], sys.argv[3]
REMOTE = ' '.join(sys.argv[4:]) or 'echo XEON-SSH-OK'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PWD, timeout=20)
_, out, err = c.exec_command(REMOTE, timeout=180)
code = out.channel.recv_exit_status()
sys.stdout.buffer.write(out.read())
sys.stdout.buffer.write(err.read())
sys.stdout.buffer.flush()
c.close()
raise SystemExit(code)
