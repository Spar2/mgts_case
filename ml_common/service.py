"""Точка входа для fiber.api: model= выбирает адаптер, по умолчанию API её не вызывает."""
from __future__ import annotations

from ml_common.adapters import get_analyzer


def predict_field(model: str, field: dict, **kwargs) -> dict:
    return get_analyzer(model, **kwargs).predict(field)
