"""Atomic heartbeat publication, including Windows read/replace contention."""

import os
import time
import uuid
from pathlib import Path


def publish(path: Path, content: str):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                # A Windows reader may hold the destination briefly. Permanent
                # permission failures still propagate; stale readiness is unsafe.
                if os.name != "nt" or attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)
