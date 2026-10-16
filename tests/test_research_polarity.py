"""Focus weight and polarity term are the loss the research fit applies."""
import inspect
from pathlib import Path

import torch

from fiber.pipeline import PipelineConfig, analyze_field
from fiber.research_cnn import (
    COOLING_IDS,
    FOCUS_EXPERT_WEIGHT,
    HARD_NEGATIVE_IDS,
    ResearchResidualCNN,
    fit_rows,
    research_batch_loss,
)
from fiber.weak_labels import EXPERT_WEIGHT
import numpy as np


def _row(seed_id, split_code, weight, class_name, presence, origin="expert_seed"):
    return {
        "id": seed_id,
        "record": "1",
        "origin": origin,
        "class_name": class_name,
        "split_code": split_code,
        "weight": weight,
        "expert_confirmed": origin == "expert_seed",
        "rows_half_open": [0, 28],
        "x_m": [1.0, 5.0],
        "presence": presence,
        "notes": "draft",
    }


def _contribution(row, logits):
    return float(research_batch_loss(
        logits,
        torch.tensor([row["class_index"]]),
        torch.tensor([row["loss_weight"]], dtype=torch.float32),
        torch.tensor([row["polarity_target"]], dtype=torch.float32),
    ))


def test_focus_weight_polarity_loss_and_analyze_field(monkeypatch):
    assert PipelineConfig().threshold_z == 3.5
    assert FOCUS_EXPERT_WEIGHT > EXPERT_WEIGHT
    train_source = (Path(__file__).resolve().parents[1] / "scripts" / "train_research_cnn.py").read_text(encoding="utf-8")
    assert "research_batch_loss(" in train_source
    rows = [
        _row("quiet-auto", 0, 1.0, "Background", None, origin="quiet"),
        _row("review-00034", 0, EXPERT_WEIGHT, "Local_Heating", "yes"),
        _row("holdout-auto", 2, 1.0, "Background", None, origin="quiet"),
        _row("guard-auto", -1, 1.0, "Local_Cooling", None, origin="rule"),
        _row("next-0006", -1, 1.0, "Local_Cooling", "yes"),
        _row("next-0010", -1, 1.0, "Local_Cooling", "yes"),
        _row("next-0002", 2, EXPERT_WEIGHT, "Background", "no"),
        _row("review-00223", 0, EXPERT_WEIGHT, "Local_Heating", "yes"),
    ]
    kept = fit_rows(rows)
    by_id = {row["id"]: row for row in kept}
    assert "holdout-auto" not in by_id
    assert "guard-auto" not in by_id
    heating = by_id["review-00034"]
    automatic = by_id["quiet-auto"]
    assert heating["class_name"] == "Local_Heating"
    assert heating["loss_weight"] == EXPERT_WEIGHT
    assert automatic["loss_weight"] == 1.0
    for seed_id in HARD_NEGATIVE_IDS & {"next-0006", "next-0010"}:
        assert by_id[seed_id]["presence"] == "no"
        assert by_id[seed_id]["class_name"] == "Background"
        assert by_id[seed_id]["polarity_target"] == 0.0
        assert by_id[seed_id]["loss_weight"] > heating["loss_weight"]
        assert by_id[seed_id]["loss_weight"] > automatic["loss_weight"]
    for seed_id in COOLING_IDS:
        assert by_id[seed_id]["class_name"] == "Local_Cooling"
        assert by_id[seed_id]["polarity_target"] == -1.0
        assert by_id[seed_id]["loss_weight"] > heating["loss_weight"]
        assert by_id[seed_id]["loss_weight"] > automatic["loss_weight"]
    logits = torch.zeros(1, 6)
    hard_loss = _contribution(by_id["next-0006"], logits)
    cool_loss = _contribution(by_id["next-0002"], logits)
    heat_loss = _contribution(heating, logits)
    auto_loss = _contribution(automatic, logits)
    assert hard_loss > heat_loss
    assert cool_loss > heat_loss
    assert hard_loss > auto_loss
    assert cool_loss > auto_loss
    signed = torch.zeros(1, 6)
    signed[0, heating["class_index"]] = 1.5
    signed[0, by_id["next-0002"]["class_index"]] = -1.25
    hot_target = research_batch_loss(
        signed,
        torch.tensor([heating["class_index"]]),
        torch.tensor([heating["loss_weight"]]),
        torch.tensor([1.0]),
    )
    cold_target = research_batch_loss(
        signed,
        torch.tensor([heating["class_index"]]),
        torch.tensor([heating["loss_weight"]]),
        torch.tensor([-1.0]),
    )
    assert float(hot_target) != float(cold_target)
    try:
        research_batch_loss(torch.zeros(1, 7), torch.tensor([0]), torch.tensor([1.0]), torch.tensor([0.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("a seventh score was accepted as a class")
    scores = ResearchResidualCNN(n_class=6)(torch.zeros(2, 1, 32, 64))
    assert tuple(scores.shape) == (2, 6)
    source = inspect.getsource(analyze_field)
    assert "research_cnn" not in source
    assert "research_residual_cnn" not in source
    real_open = open

    def guard(path, *args, **kwargs):
        name = str(path).replace("\\", "/")
        if "research_residual_cnn" in name or "research_cnn" in name:
            raise AssertionError(name)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guard)
    hot = np.full((12, 100), 22.0, dtype=np.float32)
    hot[:, 45:51] += 6.0
    found = analyze_field(hot, np.arange(100, dtype=np.float64), np.arange(12, dtype=np.float64) * 60.0)
    assert found["manifest"]["clean"] == "median_x3"
    assert found["manifest"]["detector"] == "spatial_z_halo"
    assert found["manifest"]["config"]["threshold_z"] == 3.5
    assert found["manifest"]["trained_classifier_operational"] is False
