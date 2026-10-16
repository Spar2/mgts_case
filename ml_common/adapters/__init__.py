"""Ленивые адаптеры. Импорт пакета не тянет torch."""
from __future__ import annotations

METHODS = ("fiber", "ens_gen2", "larpathon")


def get_analyzer(name: str, **kwargs):
    key = (name or "").strip()
    if key == "fiber":
        from ml_common.adapters.fiber import FiberAnalyzer

        return FiberAnalyzer()
    if key == "larpathon":
        from ml_common.adapters.larpathon import LarpathonAnalyzer

        return LarpathonAnalyzer(**kwargs)
    if key in {"ens_gen2", "grok_gen2"}:
        from ml_common.adapters.grok_gen2 import GrokGen2Analyzer

        return GrokGen2Analyzer(**kwargs)
    raise ValueError(f"unknown model {name!r}; expected one of {METHODS}")
