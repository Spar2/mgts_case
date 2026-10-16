"""Scores for detectors. Synthetic truth is known. Real windows are witnesses."""

from fiber.benchmark.cards import MODEL_CARDS
from fiber.benchmark.events import alarm_fraction, evaluate_events

__all__ = ["MODEL_CARDS", "alarm_fraction", "evaluate_events"]
