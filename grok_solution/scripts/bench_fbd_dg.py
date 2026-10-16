"""KID synth-vs-real по энкодерам DG/MIXSTYLE/RANDCONV/STATIC.

Один запуск смотрит artifacts/eval/s3_{dg,mixstyle,randconv,static}*.
Новый U-Net получает per-file KID (v8/v10/v11 test, seed 0). Это диагностика:
числа не выбирают модель и не калибруют детектор на реальных CSV.
Повтор каждые 10 минут делает внешний планировщик. FD_∞, KL и BN-шум не считаются.

Пример:
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /workspace/repo/run/.venv/bin/python scripts/bench_fbd_dg.py
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "artifacts" / "eval"
OUT_PATH = EVAL_DIR / "fbd.json"
DOC_PATH = ROOT / "docs" / "TZ_A_FBD.md"
EVENTS_PATH = Path("/workspace/grok_logs/_events.log")

FAMILIES = ("dg", "mixstyle", "randconv", "static")
VERSIONS = ("v8", "v10", "v11")
REAL_NAMES = ("1", "2", "3", "4", "5", "6")
FBD5_END = "<!-- FBD5_END -->"
FBD6_START = "<!-- FBD6_START -->"
FBD6_END = "<!-- FBD6_END -->"
GLOB_TEXT = "artifacts/eval/s3_{dg,mixstyle,randconv,static}*"

PROTOCOL = {
    "versions": list(VERSIONS),
    "splits": "test",
    "with_val": False,
    "n_synth_per_stratum": 16,
    "n_real_per_stratum": 120,
    "patch_t_x": [96, 128],
    "batch": 32,
    "seed": 0,
    "balance": "equal n_zone and n_quiet, round-robin across files",
    "kernel": "unbiased polynomial MMD degree 3, gamma=1/d, coef0=1 (Binkowski 2018)",
    "real_policy": "MGTS csv evaluation only; no BN, threshold, or probe fit on real",
    "selection": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(eval_dir: Path) -> list[dict]:
    """Файлы .pt семейств и .pt внутри одноимённых каталогов. Прочие суффиксы мимо."""
    found = []
    seen = set()
    if not eval_dir.is_dir():
        return found
    for path in sorted(eval_dir.iterdir(), key=lambda p: p.name.lower()):
        family = _family_of(path.name)
        if family is None:
            continue
        if path.is_file():
            candidates = [path] if path.suffix.lower() == ".pt" else []
        elif path.is_dir():
            candidates = [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() == ".pt"]
        else:
            candidates = []
        for cand in candidates:
            try:
                key = str(cand.resolve())
            except OSError:
                key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            found.append({"family": family, "path": cand})
    return found


def _family_of(name: str):
    low = name.lower()
    for family in FAMILIES:
        if low.startswith(f"s3_{family}"):
            return family
    return None


def pending_items(found: list[dict], known_shas: set[str]) -> list[dict]:
    pending = []
    for item in found:
        path = item["path"]
        try:
            digest = sha256_file(path)
            size = path.stat().st_size
        except OSError:
            continue
        if digest in known_shas:
            continue
        pending.append({**item, "sha256": digest, "size": int(size)})
    return pending


def known_shas(block: dict) -> set[str]:
    shas = set()
    for row in list(block.get("models") or []) + list(block.get("skipped") or []):
        digest = row.get("sha256")
        if digest:
            shas.add(digest)
    return shas


def is_partial_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "eof",
        "truncated",
        "ran out of input",
        "pytorchstreamreader",
        "failed reading zip",
        "invalid load key",
        "unpicklingerror",
        "pickle data was truncated",
    )
    return any(marker in text for marker in markers)


def _strip_module(state: dict) -> dict:
    if any(str(key).startswith("module.") for key in state):
        return {str(key)[len("module."):] if str(key).startswith("module.") else key: value for key, value in state.items()}
    return state


def _trim_encoder_prefix(state: dict) -> dict | None:
    state = _strip_module(state)
    if "e1.0.weight" in state:
        return state
    for key in state:
        text = str(key)
        if not text.endswith("e1.0.weight"):
            continue
        prefix = text[: -len("e1.0.weight")]
        trimmed = {str(other)[len(prefix):]: value for other, value in state.items() if str(other).startswith(prefix)}
        if "e1.0.weight" in trimmed:
            return trimmed
    return None


def unwrap_unet_state(obj, depth: int = 0):
    """Достаёт state_dict DTSUNet из обёрток state_dict/model/unet и префикса module."""
    if depth > 4:
        return None
    if hasattr(obj, "state_dict") and not isinstance(obj, dict):
        try:
            obj = obj.state_dict()
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    direct = _trim_encoder_prefix(obj)
    if direct is not None:
        return direct
    for key in ("state_dict", "model", "unet", "state", "weights"):
        if key in obj:
            found = unwrap_unet_state(obj[key], depth + 1)
            if found is not None:
                return found
    return None


def _ensure_root_on_path():
    import sys
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_encoder(path: Path):
    """Грузит энкодер. partial — файл ещё пишется, sha не запоминаем. not_unet — sha запоминаем."""
    _ensure_root_on_path()
    import torch
    torch.set_num_threads(2)
    from dts.fbd import BottleneckEncoder
    from dts.models import DTSUNet

    try:
        size_before = path.stat().st_size
    except OSError as exc:
        return None, "partial", _short(exc)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if is_partial_error(exc):
            return None, "partial", _short(exc)
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc2:
            return None, "partial", _short(exc2)
    try:
        if path.stat().st_size != size_before:
            return None, "partial", "size changed during read"
    except OSError as exc:
        return None, "partial", _short(exc)
    state = unwrap_unet_state(obj)
    if state is None:
        return None, "not_unet", "no e1.0.weight"
    try:
        width = int(state["e1.0.weight"].shape[0])
        model = DTSUNet(w=width)
    except Exception as exc:
        return None, "not_unet", _short(exc)
    own = model.state_dict()
    n_model = len(own)
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    if missing or unexpected:
        del model
        return None, "key_mismatch", (
            f"strict missing={len(missing)} unexpected={len(unexpected)} model={n_model}"
        )
    try:
        incompat = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        del model
        return None, "key_mismatch", f"strict shape model={n_model} {_short(exc)}"
    if list(incompat.missing_keys) or list(incompat.unexpected_keys):
        del model
        return None, "key_mismatch", (
            f"strict missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)} model={n_model}"
        )
    model.eval()
    enc = BottleneckEncoder(model, width, str(path))
    enc.load_info = {
        "strict": True,
        "n_loaded": n_model,
        "n_model": n_model,
        "missing": 0,
        "unexpected": 0,
    }
    return enc, "ok", ""


def state_max_abs(path_a: Path, path_b: Path):
    """max |Δ| по совпавшим тензорам двух чекпойнтов. None, если наборы ключей разные."""
    import torch

    def read(path: Path):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        return unwrap_unet_state(obj)

    left, right = read(path_a), read(path_b)
    if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
        return None
    gap = 0.0
    for key, tensor in left.items():
        other = right[key]
        if not hasattr(tensor, "shape") or tuple(tensor.shape) != tuple(other.shape):
            return None
        gap = max(gap, float((tensor.detach().float() - other.detach().float()).abs().max()))
    return round(gap, 6)


def _short(exc: BaseException) -> str:
    line = str(exc).splitlines()
    return (line[-1] if line else type(exc).__name__)[:200]


def rel_ckpt(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _num(value, nd=6):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, nd)


def compact_summary(family: str, path: Path, digest: str, summary: dict, seconds: float) -> dict:
    per = {}
    for ver, rows in (summary.get("per_file") or {}).items():
        per[ver] = {}
        for name, row in rows.items():
            per[ver][str(name)] = {
                "kid": _num(row.get("kid")),
                "n": row.get("n"),
                "n_zone": row.get("n_zone"),
                "n_quiet": row.get("n_quiet"),
                "zone_fraction": _num(row.get("zone_fraction"), 4),
            }
    balance = {}
    for ver, block in (summary.get("real") or {}).items():
        balance[ver] = {
            "n": block.get("n"),
            "n_zone": block.get("n_zone"),
            "n_quiet": block.get("n_quiet"),
            "zone_fraction": _num(block.get("zone_fraction"), 4),
        }
    return {
        "family": family,
        "checkpoint": rel_ckpt(path),
        "sha256": digest,
        "width": int(summary["width"]),
        "feature_dim": int(summary["feature_dim"]),
        "real_kid": {ver: _num(value) for ver, value in (summary.get("real_kid") or {}).items()},
        "real_balance": balance,
        "per_file": per,
        "rank_far_to_near": summary.get("rank_far_to_near") or {},
        "seconds": round(float(seconds), 1),
        "real_calibrated": False,
        "role": "diagnostic_not_selection",
    }


def apply_update(payload: dict, update: dict) -> dict:
    """Меняет только dg_models. Остальные ключи payload остаются как были."""
    prev = dict(payload.get("dg_models") or {})
    models = {}
    for row in prev.get("models") or []:
        if row.get("sha256"):
            models[row["sha256"]] = row
    for row in update.get("models") or []:
        models[row["sha256"]] = row
    skipped = {}
    for row in prev.get("skipped") or []:
        if row.get("sha256"):
            skipped[row["sha256"]] = row
    for row in update.get("skipped") or []:
        skipped[row["sha256"]] = row
    done = {row.get("checkpoint") for row in models.values()} | {row.get("checkpoint") for row in skipped.values()}
    retries = {}
    for row in list(prev.get("retries") or []) + list(update.get("retries") or []):
        retries[(row.get("checkpoint"), row.get("size"))] = row
    retries = {key: row for key, row in retries.items() if row.get("checkpoint") not in done}
    seen = set(prev.get("seen_families") or [])
    seen.update(update.get("seen_families") or [])
    watch = dict(prev.get("watch") or {})
    incoming = dict(update.get("watch") or {})
    if "started_utc" not in watch and "started_utc" not in incoming:
        incoming["started_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    watch.update(incoming)
    payload["dg_models"] = {
        "role": "diagnostic_not_selection",
        "selection": False,
        "real_calibrated": False,
        "watch": watch,
        "protocol": PROTOCOL,
        "seen_families": [family for family in FAMILIES if family in seen],
        "models": sorted(models.values(), key=lambda row: (row.get("family", ""), row.get("checkpoint", ""))),
        "skipped": sorted(skipped.values(), key=lambda row: (row.get("family", ""), row.get("checkpoint", ""))),
        "retries": [retries[key] for key in sorted(retries, key=lambda item: (str(item[0]), item[1] or 0))],
    }
    return payload


def _fmt(value) -> str:
    if value is None:
        return "н/д"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _rank_text(names) -> str:
    return " > ".join(str(name) for name in names) if names else "н/д"


def render_section(block: dict, found: list[dict]) -> str:
    lines = ["## DG-модели", ""]
    lines.append(
        "Диагностика, не выбор. KID synth-vs-real считается по каждому реальному файлу на bottleneck "
        "энкодера, когда в `artifacts/eval` появляется чекпойнт `s3_dg*`, `s3_mixstyle*`, `s3_randconv*` "
        "или `s3_static*` (файл `.pt` или `.pt` внутри такого каталога). Протокол per-file: v8, v10 и v11, "
        "только test, 16 зонных и 16 тихих патчей на синтетический файл, 120+120 на кабель, патч 96×128, "
        "batch 32, seed 0. Сторож проверяет каталог каждые 10 минут. Реальные CSV читаются только в оценке. "
        "Абсолютный KID каждой ширины остаётся в своей таблице."
    )
    lines.append("")
    models = list(block.get("models") or [])
    present = {item["family"] for item in found}
    present.update(row.get("family") for row in models)
    present.update(row.get("family") for row in block.get("skipped") or [])
    if not models:
        lines.append(
            "Пока нет чекпойнтов. Как только появится U-Net с весом `e1.0.weight`, сюда допишется per-file KID."
        )
        lines.append("")
    for row in models:
        ckpt = row.get("checkpoint") or ""
        lines.append(f"### {row.get('family')}: `{Path(ckpt).name}`")
        lines.append("")
        sha = str(row.get("sha256") or "")
        load = row.get("load") or {}
        load_text = ""
        if load.get("n_loaded") is not None:
            load_text = (
                f" Загрузка strict={str(bool(load.get('strict'))).lower()}, "
                f"ключей {load.get('n_loaded')}/{load.get('n_model')}."
            )
        lines.append(
            f"sha256 `{sha[:16]}`, ширина {row.get('width')}, размерность {row.get('feature_dim')}. "
            f"Таблица описывает этот энкодер.{load_text}"
        )
        same = row.get("same_state_dict_as")
        if same:
            lines.append(
                f"Все тензоры state dict совпадают с `{same}` (max |Δ| = 0). "
                "Отличие только в байтах файла."
            )
        lines.append("")
        lines.append("| файл | KID v8 | KID v10 | KID v11 |")
        lines.append("|---|---:|---:|---:|")
        order = list((row.get("rank_far_to_near") or {}).get("v8") or [])
        for ver_rows in (row.get("per_file") or {}).values():
            for name in ver_rows:
                if name not in order:
                    order.append(name)
        per = row.get("per_file") or {}
        for name in order:
            cells = [_fmt(((per.get(ver) or {}).get(name) or {}).get("kid")) for ver in VERSIONS]
            lines.append(f"| {name}.csv | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")
        ranks = row.get("rank_far_to_near") or {}
        bits = [f"{ver} {_rank_text(ranks.get(ver))}" for ver in VERSIONS]
        lines.append("Дальний → ближний: " + "; ".join(bits) + ".")
        lines.append("")
        pooled = row.get("real_kid") or {}
        balance = (row.get("real_balance") or {}).get("v8") or {}
        n_text = ""
        if balance.get("n") is not None:
            n_text = f" (n={balance.get('n')}, доля зон {_fmt(balance.get('zone_fraction'))})"
        lines.append(
            "Пул KID v8/v10/v11 = "
            + " / ".join(_fmt(pooled.get(ver)) for ver in VERSIONS)
            + n_text
            + "."
        )
        lines.append("")
    missing = [family for family in FAMILIES if family not in present]
    if missing and models:
        names = ", ".join(f"`s3_{family}*`" for family in missing)
        lines.append(f"В этом обходе ещё нет файлов: {names}.")
        lines.append("")
    skipped = list(block.get("skipped") or [])
    if skipped:
        lines.append("Без энкодера DTSUNet KID не считался:")
        lines.append("")
        for row in skipped:
            lines.append(f"- `{row.get('checkpoint')}` ({row.get('family')}), sha256 `{str(row.get('sha256') or '')[:16]}`.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def splice_fbd6(doc: str, section: str) -> str:
    block = FBD6_START + "\n\n" + section.rstrip() + "\n\n" + FBD6_END + "\n"
    if FBD6_START in doc and FBD6_END in doc:
        start = doc.index(FBD6_START)
        end = doc.index(FBD6_END) + len(FBD6_END)
        if end < len(doc) and doc[end] == "\n":
            end += 1
        return doc[:start] + block + doc[end:]
    if FBD5_END in doc:
        insert = doc.index(FBD5_END) + len(FBD5_END)
        if insert < len(doc) and doc[insert] == "\n":
            insert += 1
        return doc[:insert] + "\n" + block + doc[insert:]
    if doc and not doc.endswith("\n"):
        doc += "\n"
    return doc + "\n" + block


def dump_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".fbd6partial")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def append_event(path: Path, body: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%H:%M")
    line = f"{stamp} {body}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    return line


def _retry_known(block: dict, checkpoint: str, size: int) -> bool:
    for row in block.get("retries") or []:
        if row.get("checkpoint") == checkpoint and row.get("size") == size:
            return True
    return False


def publish(out_path: Path, doc_path: Path, events_path: Path, update: dict, found: list[dict], log_lines: list[str]):
    # Перечитываем JSON прямо перед записью: длинный KID не должен затереть чужие ключи.
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    keys = list(payload.keys())
    apply_update(payload, update)
    missing = [key for key in keys if key not in payload]
    if missing:
        raise RuntimeError("dg publish dropped fbd.json keys: " + ", ".join(missing))
    dump_json(out_path, payload)
    doc = doc_path.read_text(encoding="utf-8") if doc_path.is_file() else ""
    doc_path.write_text(splice_fbd6(doc, render_section(payload["dg_models"], found)), encoding="utf-8")
    for body in log_lines:
        append_event(events_path, body)
    return payload["dg_models"]


def _import_bench():
    import sys
    root = str(ROOT)
    scripts = str(ROOT / "scripts")
    if root not in sys.path:
        sys.path.insert(0, root)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import bench_fbd
    return bench_fbd


def score_checkpoint(enc, name: str) -> dict:
    import torch
    torch.set_num_threads(2)
    bench = _import_bench()
    packs, real_packs = bench.collect_multi(
        [(name, enc)], list(VERSIONS), list(REAL_NAMES),
        PROTOCOL["n_synth_per_stratum"], PROTOCOL["n_real_per_stratum"],
        (96, 128), PROTOCOL["seed"], PROTOCOL["batch"], 0, with_val=False,
    )
    return bench.summarize_kid(name, enc, packs[name], real_packs[name], list(VERSIONS), PROTOCOL["seed"])


def _watch_update(seen: list[str], logged: bool) -> dict:
    watch = {
        "logged": True,
        "interval": "10m",
        "glob": GLOB_TEXT,
    }
    if not logged:
        watch["started_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"watch": watch, "seen_families": seen}


def _kid_log(row: dict) -> str:
    per = (row.get("per_file") or {}).get("v8") or {}
    order = list((row.get("rank_far_to_near") or {}).get("v8") or [])
    for name in per:
        if name not in order:
            order.append(name)
    kids = ",".join(f"{name}:{_fmt(per.get(name, {}).get('kid'))}" for name in order)
    pooled = row.get("real_kid") or {}
    pooled_text = "/".join(_fmt(pooled.get(ver)) for ver in VERSIONS)
    rank = ">".join(str(name) for name in (row.get("rank_far_to_near") or {}).get("v8") or [])
    ckpt = Path(str(row.get("checkpoint") or "")).name
    load = row.get("load") or {}
    loaded = ""
    if load.get("n_loaded") is not None:
        loaded = (
            f" strict={str(bool(load.get('strict'))).lower()} "
            f"loaded={load.get('n_loaded')}/{load.get('n_model')} "
            f"missing={load.get('missing')} unexpected={load.get('unexpected')}"
        )
    versus = row.get("weight_max_abs_vs_dg")
    versus_text = "" if versus is None else f" weight_max_abs_vs_dg={versus}"
    tag = os.environ.get("FBD6_TAG", "DONE")
    if tag not in {"DONE", "FIX"}:
        tag = "DONE"
    return (
        f"FBD6_{tag} family={row.get('family')} ckpt={ckpt} dim={row.get('feature_dim')} "
        f"width={row.get('width')} sha256={str(row.get('sha256') or '')[:16]}{loaded}{versus_text} "
        f"rank_v8={rank or 'na'} pooled={pooled_text} kid_v8={kids}"
    )


def main() -> int:
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    _ensure_root_on_path()
    if not OUT_PATH.is_file():
        raise SystemExit(f"нет {OUT_PATH}")
    found = discover(EVAL_DIR)
    seen = sorted({item["family"] for item in found})
    payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    block = payload.get("dg_models") or {}
    pending = pending_items(found, known_shas(block))
    doc = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.is_file() else ""
    logged = bool((block.get("watch") or {}).get("logged"))
    section_ok = FBD6_START in doc and FBD6_END in doc
    if not pending and logged and section_ok:
        print(f"[fbd6] idle found={len(found)} scored={len(block.get('models') or [])}", flush=True)
        return 0
    if not logged or not section_ok:
        lines = [] if logged else ["FBD6_WATCH pending=dg,mixstyle,randconv,static interval=10m"]
        block = publish(OUT_PATH, DOC_PATH, EVENTS_PATH, _watch_update(seen, logged), found, lines)
        print("[fbd6] watch section written", flush=True)
    for item in pending:
        path = item["path"]
        checkpoint = rel_ckpt(path)
        try:
            if path.stat().st_size != item["size"]:
                _note_retry(block, found, seen, checkpoint, item["size"], "size changed")
                block = json.loads(OUT_PATH.read_text(encoding="utf-8"))["dg_models"]
                continue
        except OSError as exc:
            _note_retry(block, found, seen, checkpoint, item["size"], _short(exc))
            block = json.loads(OUT_PATH.read_text(encoding="utf-8"))["dg_models"]
            continue
        enc, status, detail = load_encoder(path)
        if status == "partial":
            _note_retry(block, found, seen, checkpoint, item["size"], detail or "partial")
            block = json.loads(OUT_PATH.read_text(encoding="utf-8"))["dg_models"]
            continue
        if status != "ok":
            reason = "strict" if status == "key_mismatch" else "not_unet"
            row = {
                "family": item["family"],
                "checkpoint": checkpoint,
                "sha256": item["sha256"],
                "reason": reason,
                "detail": detail,
                "real_calibrated": False,
            }
            print(f"[fbd6] skip {checkpoint} {reason} {detail}", flush=True)
            tag = "FBD6_FIX" if reason == "strict" else "FBD6_SKIP"
            block = publish(
                OUT_PATH, DOC_PATH, EVENTS_PATH,
                {"skipped": [row], "seen_families": seen},
                found,
                [f"{tag} family={item['family']} ckpt={path.name} reason={reason} sha256={item['sha256'][:16]} {detail}"],
            )
            continue
        import time
        t0 = time.perf_counter()
        name = f"{item['family']}:{path.stem}"
        dg_path = EVAL_DIR / "s3_dg_unet.pt"
        versus = None
        if dg_path.is_file() and path.resolve() != dg_path.resolve():
            versus = state_max_abs(path, dg_path)
        print(f"[fbd6] score {checkpoint} width={enc.width} dim={enc.dim} strict={enc.load_info}", flush=True)
        summary = score_checkpoint(enc, name)
        row = compact_summary(item["family"], path, item["sha256"], summary, time.perf_counter() - t0)
        row["load"] = dict(getattr(enc, "load_info", {}) or {})
        if versus is not None:
            row["weight_max_abs_vs_dg"] = versus
        del enc
        print(f"[fbd6] done {checkpoint} pooled={row['real_kid']}", flush=True)
        block = publish(
            OUT_PATH, DOC_PATH, EVENTS_PATH,
            {"models": [row], "seen_families": seen},
            found,
            [_kid_log(row)],
        )
    return 0


def _note_retry(block: dict, found: list[dict], seen: list[str], checkpoint: str, size: int, detail: str):
    fresh = json.loads(OUT_PATH.read_text(encoding="utf-8")).get("dg_models") or block
    if _retry_known(fresh, checkpoint, size):
        print(f"[fbd6] retry quiet {checkpoint} size={size}", flush=True)
        return
    print(f"[fbd6] retry {checkpoint} size={size} {detail}", flush=True)
    publish(
        OUT_PATH, DOC_PATH, EVENTS_PATH,
        {"retries": [{"checkpoint": checkpoint, "size": int(size), "reason": "partial", "detail": detail}], "seen_families": seen},
        found,
        [f"FBD6_RETRY ckpt={Path(checkpoint).name} size={int(size)} reason=partial"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
