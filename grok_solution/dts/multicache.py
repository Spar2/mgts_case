"""Списки mmap-тайлов по нескольким версиям кэша.

Корень версии — artifacts/cache/<ver> и artifacts/mm/<ver>, тот же контракт,
что у data.mmap_items для одного v8. Пустой сплит не молчит: его видно по длине.
"""
from __future__ import annotations
from pathlib import Path
from dts import data


def version_roots(ver: str, artifact_root: str | Path = "artifacts") -> tuple[Path, Path]:
    root = Path(artifact_root)
    return root / "cache" / ver, root / "mm" / ver


def load_splits(versions, splits=("train", "val"), artifact_root: str | Path = "artifacts") -> dict:
    out = {s: [] for s in splits}
    for ver in versions:
        cache, mm = version_roots(ver, artifact_root)
        for split in splits:
            out[split].extend(data.mmap_items(cache, split, mm))
    return out
