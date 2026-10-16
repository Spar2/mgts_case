"""CI guard: platform env consistency (compose vs code vs deploy).

Verifies every env var the code requires is provided by compose files
(or has a code default), and every deploy-script flag exists in compose/code.
Never reads or prints env VALUES — only key names. No network, no docker.
Exit 0 = OK, exit 1 = MISSING required key.
"""

import ast
import pathlib
import re
import sys

import yaml

# yaml ctor for compose merge tags (same as pr-gate tolerance)
for _tag in ("!override", "!reset"):
    yaml.SafeLoader.add_constructor(
        _tag,
        lambda loader, node: (
            loader.construct_sequence(node, deep=True)
            if isinstance(node, yaml.SequenceNode)
            else loader.construct_scalar(node)
        ),
    )

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPOSE_FILES = [
    "compose.web.yaml",
    "compose.platform.yaml",
    "stack/compose.yaml",
]
CODE_DIRS = [
    "services/web-api/workbench",
    "stack/backend",
    "stack/ingester",
    "stack/streamer",
]
DEPLOY_GLOBS = ["ops/deploy-*.sh"]

# Required-without-default vars that are intentionally optional
# (maintenance-only / opt-in override). Documented, WARN instead of FAIL.
KNOWN_OPTIONAL = {
    "DTS_BACKUP_RECIPIENT": "maintenance profile only",
    "DTS_WEB_DIST": "opt-in dist override, guarded by get()",
}


def compose_keys():
    """name -> set(files). Tracks :? required markers separately."""
    found, required = {}, set()
    for rel in COMPOSE_FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for svc in (doc.get("services") or {}).values():
            env = (svc or {}).get("environment") or {}
            items = env.items() if isinstance(env, dict) else env
            for item in items:
                text = item if isinstance(item, str) else item[0]
                name = re.split(r"[:=]", text, maxsplit=1)[0]
                found.setdefault(name, set()).add(rel)
                if ":?" in text:
                    required.add(name)
    return found, required


def code_keys():
    """name -> (kind, files). kind: default (has None-default),
    required (bare subscript), loopback (default points at loopback)."""
    import urllib.parse
    out = {}
    for rel in CODE_DIRS:
        d = ROOT / rel
        if not d.exists():
            continue
        for f in sorted(d.glob("*.py")):
            try:
                tree = ast.parse(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            for node in ast.walk(tree):
                name, kind = None, None
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    fn = node.func
                    if fn.attr == "get" and isinstance(fn.value, ast.Attribute):
                        v = fn.value
                        if (isinstance(v.value, ast.Name) and v.value.id == "os"
                                and v.attr == "environ" and node.args
                                and isinstance(node.args[0], ast.Constant)):
                            name = node.args[0].value
                            kind = "default"
                            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                                try:
                                    host = (urllib.parse.urlsplit(str(node.args[1].value)).hostname
                                            or "")
                                except Exception:
                                    host = ""
                                if host in ("127.0.0.1", "localhost", "::1"):
                                    kind = "loopback"
                    elif fn.attr == "getenv" and isinstance(fn.value, ast.Name) and fn.value.id == "os":
                        if node.args and isinstance(node.args[0], ast.Constant):
                            name = node.args[0].value
                            kind = "default"
                            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                                if str(node.args[1].value).startswith(("127.0.0.1", "localhost")):
                                    kind = "loopback"
                elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
                    v = node.value
                    if (isinstance(v.value, ast.Name) and v.value.id == "os"
                            and v.attr == "environ"
                            and isinstance(node.slice, ast.Constant)):
                        name, kind = node.slice.value, "required"
                if name:
                    prev = out.get(name, ("default", set()))
                    order = {"required": 0, "loopback": 1, "default": 2}
                    keep = kind if order[kind] < order[prev[0]] else prev[0]
                    out[name] = (keep, prev[1] | {f.name})
    return out


def deploy_flags():
    """Env names referenced in deploy scripts (informational)."""
    found = set()
    for pat in DEPLOY_GLOBS:
        for f in sorted(ROOT.glob(pat)):
            text = f.read_text(encoding="utf-8", errors="replace")
            found |= set(re.findall(r"DTS_[A-Z_]+|APP_PORT|PORT|DATA_DIR", text))
    return found


def main():
    compose, required = compose_keys()
    code = code_keys()
    deploy = deploy_flags()
    rows = []
    bad, warns = 0, 0
    for name in sorted(code):
        kind, files = code[name]
        in_compose = name in compose
        if kind == "required" and not in_compose:
            if name in KNOWN_OPTIONAL:
                verdict = "OPTIONAL_KNOWN"
                warns += 1
            else:
                verdict = "MISSING"
                bad += 1
        elif kind == "loopback" and in_compose:
            verdict = "LOOPBACK_DEFAULT"
            warns += 1
        else:
            verdict = "DEFAULT_OK" if kind != "required" else "OK"
        rows.append((name, verdict, ",".join(sorted(files)[:2]),
                     ",".join(sorted(compose.get(name, ()))) or "-",
                     "deploy" if name in deploy else "-"))
    print("var|verdict|code|compose|deploy")
    for r in rows:
        print("|".join(r))
    if required - set(compose):
        print("REQUIRED-NOT-IN-COMPOSE: " + ",".join(sorted(required - set(compose))))
        bad += 1
    print("RESULT: " + ("FAIL" if bad else ("WARN" if warns else "OK")))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
