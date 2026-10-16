"""Разовая сверка CH/Redis: truncate/report/muts/streams/purge.
Вызов: python3 verify.py truncate|report|muts|streams|purge [dts_id] [--yes]."""

import os
import re
import sys
import urllib.request

CH = "http://127.0.0.1:8123/"
STREAMS = ("dts:profiles", "dts:events")
GROUP = "ingesters"
PURGE_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
USAGE = "usage: python3 verify.py truncate|report|muts|streams|purge [dts_id] [--yes]"


def q(text):
    url = CH + "?query=" + urllib.request.quote(text)
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode()


def confirmed(args):
    return "--yes" in args or os.environ.get("CONFIRM") == "1"


def _mini_resp():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
    from miniresp import MiniResp

    return MiniResp("127.0.0.1", 6379, timeout=10.0)


def pending_total():
    """Сумма pending по всем стримам через XPENDING (miniresp xpending_summary)."""
    from miniresp import RedisError

    r = _mini_resp()
    try:
        total = 0
        for stream in STREAMS:
            try:
                n, _, _, _ = r.xpending_summary(stream, GROUP)
            except RedisError:
                continue  # стрим/группа отсутствует — pending нет
            total += n
        return total
    finally:
        r.close()


def do_truncate(args):
    if not confirmed(args):
        print(
            "refuse: truncate wipes haklolarp.profiles_raw + haklolarp.events;"
            " re-run with --yes or CONFIRM=1",
            file=sys.stderr,
        )
        return 2
    try:
        pending = pending_total()
    except Exception as e:
        print(f"refuse: preflight xpending failed: {e}", file=sys.stderr)
        return 3
    if pending != 0:
        print(
            f"refuse: streams pending={pending} != 0; drain ingesters first",
            file=sys.stderr,
        )
        return 3
    print(q("TRUNCATE TABLE haklolarp.profiles_raw"))
    print(q("TRUNCATE TABLE haklolarp.events"))
    print("truncated")
    return 0


def do_report():
    print(
        q(
            "SELECT dts_id, channel_id, count(), min(ts), max(ts), max(segment_id)"
            " FROM haklolarp.profiles_raw GROUP BY dts_id, channel_id"
        )
    )
    print(q("SELECT count() FROM haklolarp.profiles_raw"))
    return 0


def do_muts():
    print(
        q(
            "SELECT mutation_id, is_done, parts_to_do FROM system.mutations"
            " WHERE database = 'haklolarp' AND table = 'profiles_raw'"
        )
    )
    return 0


def do_streams():
    from miniresp import RedisError

    r = _mini_resp()
    try:
        for stream in STREAMS:
            try:
                print(f"stream={stream} len={r.xlen(stream)}")
                total, first, last, consumers = r.xpending_summary(stream, GROUP)
                print(f"  pending={total} first={first} last={last} consumers={consumers}")
            except RedisError as e:
                print(f"stream={stream} absent: {e}")
                continue
            try:
                groups = r.xinfo_groups(stream)
            except RedisError as e:
                print(f"  groups: {e}")
                continue
            for g in groups:
                print(
                    f"  group={g.get('name')} lag={g.get('lag')} "
                    f"pending={g.get('pending')} last-id={g.get('last-delivered-id')}"
                )
    finally:
        r.close()
    return 0


def do_purge(args):
    positional = [a for a in args if a != "--yes"]
    if len(positional) != 1:
        print(USAGE, file=sys.stderr)
        return 2
    dts = positional[0]
    if not PURGE_RE.match(dts):
        print(
            f"refuse: bad dts_id {dts!r}; allowlist ^[A-Za-z0-9._:-]+$",
            file=sys.stderr,
        )
        return 2
    # Allowlist исключает кавычку; замена '' — защита в глубину.
    safe = dts.replace("'", "''")
    print(q(f"ALTER TABLE haklolarp.profiles_raw DELETE WHERE dts_id = '{safe}'"))
    print("purge issued")
    return 0


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    if len(argv) < 2 or argv[1] in ("-h", "--help", "help"):
        print(USAGE, file=sys.stderr)
        return 2
    mode, rest = argv[1], argv[2:]
    if mode == "truncate":
        return do_truncate(rest)
    elif mode == "report":
        return do_report()
    elif mode == "muts":
        return do_muts()
    elif mode == "streams":
        return do_streams()
    elif mode == "purge":
        return do_purge(rest)
    print(f"{USAGE}\nunknown mode {mode!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
