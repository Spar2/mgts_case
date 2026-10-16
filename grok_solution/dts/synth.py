"""Версионно-независимый загрузчик синтетики dts_synth_vN.

Ожидаемая структура корня: h5/<split>/seqNNNN.h5 (+ labels.csv, sequences.csv).
Версию берём из имени dts_id ("synthv8-...") или из meta; новые версии (v9, v10)
подхватываются без правок кода, если формат h5 тот же (noisy, mask_event, mask_nuisance, x, t_unix).
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

HOT = (1, 2, 3, 4)
COLD = (5, 6)
EVENT_NAMES = {0: "background", 1: "tracing", 2: "hotspot", 3: "section_heating", 4: "slow_warming", 5: "coldspot", 6: "cooling"}


def discover(root: str | Path) -> dict:
    root = Path(root)
    seqs = []
    for f in sorted(root.glob("h5/*/seq*.h5")):
        seqs.append(dict(seq_id=f.stem, split=f.parent.name, path=str(f)))
    if not seqs:
        raise FileNotFoundError(f"нет h5/<split>/seq*.h5 в {root}")
    labels = pd.read_csv(root / "labels.csv") if (root / "labels.csv").exists() else None
    seqmeta = pd.read_csv(root / "sequences.csv") if (root / "sequences.csv").exists() else None
    version = "unknown"
    if labels is not None and len(labels):
        m = re.match(r"synth(v\d+)", str(labels["dts_id"].iloc[0]))
        version = m.group(1) if m else version
    return dict(root=str(root), version=version, seqs=seqs, labels=labels, sequences=seqmeta)


def read_seq(path: str | Path) -> dict:
    import h5py
    with h5py.File(path, "r") as f:
        noisy = f["noisy"][:].astype(np.float32)
        out = dict(noisy=noisy, x=f["x"][:].astype(np.float64), t=f["t_unix"][:].astype(np.float64),
                   mask_event=f["mask_event"][:], mask_nuisance=f["mask_nuisance"][:],
                   meta=json.loads(f.attrs.get("meta", "{}")))
    out["valid"] = np.isfinite(noisy) & (noisy > -299)
    return out
