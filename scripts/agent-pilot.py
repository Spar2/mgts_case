"""Run bounded external-agent audits; Sol verifies findings before integration.

Credentials stay in each CLI's existing account store. Outputs stay in .runtime.
No shell interpolation, auto-approval, publication or merge is performed.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def fingerprint(checkout):
    """Include bytes: porcelain alone misses edits to an already modified file."""
    digest = hashlib.sha256()
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=checkout,
    ).split(b"\0")
    for name in sorted(set(names) - {b""}):
        path = checkout / name.decode("utf-8")
        digest.update(name)
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()


def completed_answer(backend, raw_output):
    if backend == "muse":
        events = [
            json.loads(line) for line in raw_output.splitlines() if line.startswith("{")
        ]
        if any(e.get("type") == "error" for e in events):
            raise ValueError("Provider error event")
        texts = [e["part"]["text"] for e in events if e.get("type") == "text"]
        stops = [e for e in events if e.get("part", {}).get("reason") == "stop"]
        answer = texts[-1] if texts and stops else ""
    else:
        data = json.loads(raw_output)
        if (
            data.get("is_error")
            or data.get("error")
            or data.get("stopReason") != "end_turn"
        ):
            raise ValueError("Incomplete or failed Grok turn")
        answer = data.get("text", "")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("No completed answer")
    return answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["muse", "grok"])
    p.add_argument("--checkout", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--executable", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=300)
    a = p.parse_args()
    if not 1 <= a.timeout <= 600:
        p.error("Timeout must be between 1 and 600 seconds")
    checkout = a.checkout.resolve()
    output = a.output.resolve()
    if ".runtime" not in output.parts:
        p.error("Agent transcripts must stay in a private .runtime directory")
    if output.exists():
        p.error("Use a new output filename for each run; never mix receipts")
    prompt = a.prompt.read_text(encoding="utf-8")
    base_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    before = fingerprint(checkout)
    if a.backend == "muse":
        template = Path(__file__).parent / "agents" / "opencode.audit.json"
        config_path = checkout / "opencode.json"
        if not config_path.is_file() or json.loads(
            config_path.read_text(encoding="utf-8")
        ) != json.loads(template.read_text(encoding="utf-8")):
            p.error(
                "Checkout must use the exact reviewed read-only OpenCode configuration"
            )
        command = [
            str(a.executable.resolve()),
            "run",
            "--pure",
            "--agent",
            "dts-auditor",
            "--model",
            "opencode-go/muse-spark-1.3-contributor",
            "--format",
            "json",
            "--title",
            "DTS bounded audit",
            prompt,
        ]
    else:
        # Grok receives a coordinator-curated source bundle in the prompt only.
        command = [
            str(a.executable.resolve()),
            "--prompt-file",
            str(a.prompt.resolve()),
            "--verbatim",
            "--tools",
            "",
            "--deny",
            "Read",
            "--deny",
            "Bash",
            "--deny",
            "Edit",
            "--deny",
            "MCPTool",
            "--output-format",
            "json",
            "--max-turns",
            "3",
            "--permission-mode",
            "dontAsk",
            "--no-subagents",
            "--disable-web-search",
            "--model",
            "grok-4.7",
            "--reasoning-effort",
            "low",
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        result = subprocess.run(
            command, cwd=checkout, capture_output=True, timeout=a.timeout
        )
        output.write_bytes(result.stdout)
        output.with_suffix(".stderr.log").write_bytes(result.stderr)
        code = result.returncode
    except subprocess.TimeoutExpired as exc:
        output.write_bytes(exc.stdout or b"")
        output.with_suffix(".stderr.log").write_bytes(exc.stderr or b"")
        code = 124
    after = fingerprint(checkout)
    # CLI exit zero alone does not prove a provider returned an answer.
    answer = ""
    if code == 0:
        try:
            raw_output = output.read_text(encoding="utf-8")
            answer = completed_answer(a.backend, raw_output)
        except (ValueError, KeyError, TypeError):
            code = 65
    proof = {
        "backend": a.backend,
        "base_sha": base_sha,
        "exit_code": code,
        "seconds": round(time.monotonic() - started, 2),
        "checkout_unchanged": before == after,
        "content_fingerprint_before": before,
        "content_fingerprint_after": after,
        "completed_answer": bool(answer),
    }
    if answer:
        output.with_suffix(".answer.md").write_text(answer, encoding="utf-8")
    output.with_suffix(".proof.json").write_text(
        json.dumps(proof, indent=2), encoding="utf-8"
    )
    print(json.dumps(proof))
    if before != after:
        raise SystemExit("Read-only audit changed checkout; coordinator must inspect")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
