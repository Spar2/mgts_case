# FROZEN: KVM/WoL — не трогать (always-on)
import os
import shutil

os.makedirs('/usr/local/bin', exist_ok=True)
shutil.copy('/tmp/xeon-wake-only', '/usr/local/bin/xeon-wake-only')
os.chmod('/usr/local/bin/xeon-wake-only', 0o755)
os.makedirs('/root/.ssh', mode=0o700, exist_ok=True)

pub = open('/tmp/kvm_wake.pub').read().strip()
line = 'restrict,command="/usr/local/bin/xeon-wake-only" ' + pub + '\n'
ak = '/root/.ssh/authorized_keys'
cur = open(ak).read() if os.path.exists(ak) else ''
if 'xeon-wake-only' not in cur:
    with open(ak, 'a') as f:
        f.write(line)
os.chmod(ak, 0o600)
print('INSTALL_OK')
