#!/bin/sh
# Thin wrapper for systemd (journal captures stdout/stderr).
set -eu
exec python3 /opt/ci/poll/poller.py
