"""Чтение реальных файлов МГТС (dts_id, channel_id, timestamp, points=[[x_m, T_C],...]).
Только для ОЦЕНКИ/инференса: модели на них не обучаются и не калибруются."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def load_csv(path) -> dict:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"], utc=False, format="mixed")
    tz_missing = getattr(ts.dt, "tz", None) is None
    if tz_missing:
        t = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()  # наивное время = UTC (метка tz_missing)
    else:
        t = (ts.dt.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    rows = [np.asarray(json.loads(p), dtype=np.float64) for p in df["points"]]
    # общая ось x: объединение по первому самому длинному профилю
    ref = max(rows, key=len)[:, 0]
    T = np.full((len(rows), len(ref)), np.nan, np.float32)
    for i, r in enumerate(rows):
        if len(r) == len(ref) and np.allclose(r[:, 0], ref):
            T[i] = r[:, 1]
        else:
            T[i] = np.interp(ref, r[:, 0], r[:, 1], left=np.nan, right=np.nan)
    order = np.argsort(t, kind="stable"); t = t[order]; T = T[order]
    keep = np.r_[True, np.diff(t) > 0]; t = t[keep]; T = T[keep]
    valid = np.isfinite(T) & (T > -299)
    return dict(name=Path(path).stem, x=ref, t=t, T=T, valid=valid, tz_missing=bool(tz_missing), sha256=sha256(path),
                channel_id=str(df["channel_id"].iloc[0]), t_iso0=str(ts.iloc[0]))
