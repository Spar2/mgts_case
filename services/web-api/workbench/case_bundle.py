"""Read bounded ZIP data in memory; never extract untrusted filenames."""

import hashlib
import json
import zipfile


def read_bundle(path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("ZIP exceeds 16 MiB")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        required = {"manifest.json", "evidence.json", "evidence.html", "README.txt"}
        if (
            len(names) != len(set(names))
            or not required <= set(names)
            or set(names) - required - {"run.json"}
        ):
            raise ValueError("Unexpected or duplicate bundle files")
        if (
            any(
                entry.flag_bits & 1
                or entry.file_size > 8 * 1024 * 1024
                or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                for entry in entries
            )
            or sum(entry.file_size for entry in entries) > 16 * 1024 * 1024
        ):
            raise ValueError("Bundle files exceed limits or are encrypted")
        files = {entry.filename: archive.read(entry) for entry in entries}
    manifest = json.loads(files.pop("manifest.json"))
    if manifest.get("schema_version") != "dts-case-bundle-v1" or set(
        manifest["files"]
    ) != set(files):
        raise ValueError("Unsupported or incomplete manifest")
    for name, payload in files.items():
        item = manifest["files"][name]
        if (
            item["bytes"] != len(payload)
            or item["sha256"] != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError("Bundle file differs from manifest")
    report = json.loads(files["evidence.json"])
    if report.get("case", {}).get("id", manifest["case_id"]) != manifest["case_id"]:
        raise ValueError("Case identity differs from manifest")
    run = json.loads(files["run.json"]) if "run.json" in files else None
    if (
        report["source_sha256"] != manifest["source_sha256"]
        or (report.get("analysis", {}).get("run_id") or report["profile"].get("processing_run_id")) != manifest["run_id"]
    ):
        raise ValueError("Manifest identity differs from report")
    report_sha = hashlib.sha256(files["evidence.json"]).hexdigest()
    marker = f'<meta name="dts-evidence-sha256" content="{report_sha}">'.encode()
    if marker not in files["evidence.html"]:
        raise ValueError("HTML is not bound to this report")
    from .cases import digest, encoded

    if digest(encoded({"evidence": report, "run": run})) != manifest["snapshot_sha256"]:
        raise ValueError("Snapshot differs from manifest")
    answer = report.get("expert_answer")
    review = manifest.get("review")
    if bool(answer) != bool(review) or (
        answer
        and any(
            answer.get(k) != review.get(k)
            for k in ("account_id", "review_id", "queue_sha256", "revision")
        )
    ):
        raise ValueError("Expert revision differs from manifest")
    return report, run, manifest
