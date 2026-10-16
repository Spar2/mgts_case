"""Research residual 2D CNN on real residual crops.

Supervised targets are weak-label rows plus the expert seeds. Automatic
holdout (split 2) and guard or mixed (split -1) rows stay out of the fit.
An expert seed stays in the loss even when its own window is holdout or
mixed, and it is not the checkpoint set. Cooling experts and presence=no
experts use a heavier loss weight than an ordinary expert Local_Heating
card. The batch loss adds a stronger hot/cold polarity term on the six
class scores. Polarity is not a class. The saved weights are
serve=research. analyze_field does not load them.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from fiber.classify.rules import LABELS
from fiber.research_context import context_evidence
from fiber.weak_labels import EXPERT_SEED_BY_ID, EXPERT_WEIGHT

CROP_H = 32
CROP_W = 64
RESIDUAL_SCALE_C = 5.0
EVAL_TIME = 28
EVAL_SPACE = 40
EVAL_TIME_STRIDE = 28
EVAL_SPACE_STRIDE = 40
FIT_SPLIT_CODES = (0, 1)
LABEL_INDEX = {name: index for index, name in enumerate(LABELS)}
# Ordinary expert Local_Heating stays at EXPERT_WEIGHT. Cooling cards and
# presence=no cards are heavier. Polarity is a signed term on the six
# scores, not a seventh class.
FOCUS_EXPERT_WEIGHT = 40.0
POLARITY_LOSS_SCALE = 2.0
HARD_NEGATIVE_IDS = frozenset({"next-0005", "next-0006", "next-0007", "next-0010"})
COOLING_IDS = frozenset({"review-00223", "next-0002"})
PREVIOUS_FIT_MATCHES = 10
PREVIOUS_FIT_CARDS = 15
PREVIOUS_FIT_SHA256 = "fed62abe36d24bdce9a07be14677506e41606dc4491f00bd3f80462cd87f4e51"
# Published context comparison before the ask-seed refit. Not this fit's score.
HISTORICAL_CONTEXT_MATCHES = 15
HISTORICAL_CONTEXT_CARDS = 22
HISTORICAL_CONTEXT_MATCH_LABEL = "15/22"
HISTORICAL_CONTEXT_SHA256 = "144dea50ec6aaad4985afb1bbab0fdb4275b9dc89a094cd9403fa04ab31e4216"
# Fixed °C scale for the context branch. Not fit per card.
CONTEXT_SCALE_C = 10.0
CONTEXT_NAMES = ("rectangle_median_c", "far_cable_median_c", "halo_inside_fraction")
CONTEXT_ARCHITECTURE = "residual2d_context"


def context_vector(
    rectangle_median_c: float | None,
    far_cable_median_c: float | None,
    halo_inside_fraction: float | None,
) -> np.ndarray:
    """Three context numbers beside the residual crop.

    An absent far-cable median stays non-finite. It is not replaced by the
    rectangle median. The halo fraction is the share of the 8–40 m ring
    inside the rectangle.
    """
    if rectangle_median_c is None or halo_inside_fraction is None:
        raise ValueError("rectangle median and halo fraction are required")
    rect = float(rectangle_median_c)
    halo = float(halo_inside_fraction)
    if not math.isfinite(rect) or not math.isfinite(halo):
        raise ValueError("rectangle median and halo fraction must be finite")
    if far_cable_median_c is None:
        far = float("nan")
    else:
        far = float(far_cable_median_c)
        if not math.isfinite(far):
            raise ValueError("a present far-cable median must be finite")
    vector = np.array([rect, far, halo], dtype=np.float32)
    if far_cable_median_c is None and np.isfinite(vector[1]):
        raise ValueError("absent far side was filled")
    if far_cable_median_c is None and vector[1] == vector[0]:
        raise ValueError("absent far side was filled with the rectangle median")
    return vector


def context_model_input(vector: np.ndarray) -> np.ndarray:
    """Scale the three numbers. A missing far side becomes 0 with present 0.

    The far slot is not copied from the rectangle median.
    """
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (3,):
        raise ValueError("context vector is rectangle, far cable, halo fraction")
    rect = float(vector[0])
    far = float(vector[1])
    halo = float(vector[2])
    present = math.isfinite(far)
    far_value = far / CONTEXT_SCALE_C if present else 0.0
    if not present and np.isfinite(far):
        raise ValueError("absent far side was filled")
    return np.asarray(
        [rect / CONTEXT_SCALE_C, far_value, halo, 1.0 if present else 0.0],
        dtype=np.float32,
    )


def read_context_crop(field: dict, x_m, time_s, rows_half_open, x_span):
    """Residual crop plus the three numbers from the shipped context reader."""
    t0, t1 = int(rows_half_open[0]), int(rows_half_open[1])
    x0, x1 = float(x_span[0]), float(x_span[1])
    config = field["manifest"]["config"]
    measured = context_evidence(
        field,
        x_m,
        time_s,
        rows_half_open=(t0, t1),
        x_span=(x0, x1),
        inner_m=float(config["inner_m"]),
        outer_m=float(config["outer_m"]),
    )
    vector = context_vector(
        measured["rectangle_median_c"],
        measured["far_cable_median_c"],
        measured["halo_inside_fraction"],
    )
    c0, c1 = meter_columns(np.asarray(x_m, dtype=np.float64), x0, x1)
    crop = residual_crop(
        np.asarray(field["clean"]),
        np.asarray(field["score_available"]),
        t0,
        t1,
        c0,
        c1,
    )
    return crop, vector, measured


PREVIOUS_NETWORK_CLASS = {
    "review-00034": "Local_Heating",
    "review-00098": "Local_Heating",
    "review-00187": "Local_Heating",
    "review-00223": "Local_Heating",
    "next-0001": "Background",
    "next-0002": "Background",
    "next-0003": "Local_Heating",
    "next-0004": "Local_Heating",
    "next-0005": "Background",
    "next-0006": "Local_Heating",
    "next-0007": "Background",
    "next-0008": "Local_Heating",
    "next-0009": "Local_Heating",
    "next-0010": "Local_Cooling",
    "next-0011": "Local_Heating",
}

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _focus_expert(seed: dict) -> bool:
    """Cooling experts and presence=no experts take the heavier loss weight."""
    return (
        str(seed.get("presence")) == "no"
        or seed.get("class_name") == "Local_Cooling"
        or str(seed.get("id")) in HARD_NEGATIVE_IDS
        or str(seed.get("id")) in COOLING_IDS
    )


def expert_loss_weight(seed: dict, stated: float) -> float:
    """Ordinary expert weight, or the heavier focus weight.

    A presence=no card and a Local_Cooling card sit strictly above an
    ordinary expert Local_Heating card that states EXPERT_WEIGHT.
    """
    weight = max(float(stated), float(EXPERT_WEIGHT))
    if _focus_expert(seed):
        weight = max(weight, float(FOCUS_EXPERT_WEIGHT))
    return float(weight)


def polarity_target_value(class_name: str) -> float:
    """Hot +1, cold -1, otherwise 0. Not a class name."""
    if class_name not in LABEL_INDEX:
        raise ValueError(f"class_name must be one of {LABELS}")
    if class_name == "Local_Cooling":
        return -1.0
    if class_name == "Local_Heating":
        return 1.0
    return 0.0


def research_batch_loss(class_logits, class_index, sample_weight, polarity_target):
    """Loss the research fit applies to one batch.

    Class cross-entropy is multiplied by ``sample_weight``. The polarity
    term is ``POLARITY_LOSS_SCALE`` times that weight times the squared
    error of ``tanh(logit_hot - logit_cold)`` against the polarity target.
    Six class scores stay six scores.
    """
    if torch is None:
        raise RuntimeError("torch is required for the research objective")
    logits = class_logits
    if logits.ndim != 2 or int(logits.shape[1]) != len(LABELS):
        raise ValueError("class scores must be a batch of the six class names")
    weight = sample_weight.reshape(-1).float()
    target = polarity_target.reshape(-1).float()
    index = class_index.reshape(-1).long()
    if not (weight.shape[0] == logits.shape[0] == target.shape[0] == index.shape[0]):
        raise ValueError("batch lengths must match")
    ce = torch.nn.functional.cross_entropy(logits, index, reduction="none")
    margin = torch.tanh(
        logits[:, LABEL_INDEX["Local_Heating"]] - logits[:, LABEL_INDEX["Local_Cooling"]]
    )
    polarity = (margin - target) ** 2
    per_example = weight * ce + (POLARITY_LOSS_SCALE * weight) * polarity
    return per_example.mean()


def fit_rows(rows: list[dict], extra_seeds: dict | None = None) -> list[dict]:
    """Keep automatic train/development rows and every known expert seed.

    Automatic split 2 and -1 are dropped. An expert seed is kept even when
    its window is holdout or mixed. Class, note, and bounds come from the
    seed table, so a hard-negative note cannot become Local_Cooling. An
    ordinary expert keeps at least EXPERT_WEIGHT. A cooling expert and a
    presence=no expert take FOCUS_EXPERT_WEIGHT, which is heavier.
    ``extra_seeds`` adds expert rows that are not in the previous 15. The
    previous 15 still come from ``EXPERT_SEED_BY_ID``. An unknown
    ``expert_seed`` id still raises.
    """
    extras = extra_seeds or {}
    kept = []
    for row in rows:
        code = int(row["split_code"])
        row_id = str(row["id"])
        seed = EXPERT_SEED_BY_ID.get(row_id)
        if seed is None:
            seed = extras.get(row_id)
        if seed is not None:
            class_name = seed["class_name"]
            weight = expert_loss_weight(seed, float(row.get("weight", EXPERT_WEIGHT)))
            origin = "expert_seed"
            notes = seed["notes"]
            presence = seed["presence"]
            expert_confirmed = True
            record = seed["record"]
            rows_half_open = [int(seed["rows_half_open"][0]), int(seed["rows_half_open"][1])]
            x_m = [float(seed["x_m"][0]), float(seed["x_m"][1])]
        else:
            if str(row.get("origin")) == "expert_seed":
                raise ValueError(f"unknown expert seed {row.get('id')}")
            if code not in FIT_SPLIT_CODES:
                continue
            class_name = row["class_name"]
            weight = float(row["weight"])
            origin = str(row["origin"])
            notes = row.get("notes")
            presence = row.get("presence")
            expert_confirmed = bool(row.get("expert_confirmed"))
            record = str(row["record"])
            rows_half_open = [int(row["rows_half_open"][0]), int(row["rows_half_open"][1])]
            x_m = [float(row["x_m"][0]), float(row["x_m"][1])]
        if class_name not in LABEL_INDEX:
            raise ValueError(f"class_name must be one of {LABELS}")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("loss weight must be a positive finite number")
        kept.append({
            "id": str(row["id"]),
            "record": record,
            "origin": origin,
            "class_name": class_name,
            "class_index": LABEL_INDEX[class_name],
            "split_code": code,
            "loss_weight": weight,
            "polarity_target": polarity_target_value(class_name),
            "rows_half_open": rows_half_open,
            "x_m": x_m,
            "expert_confirmed": expert_confirmed,
            "presence": presence,
            "notes": notes,
        })
    return kept


def loss_and_checkpoint_masks(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Expert seeds join the loss. The checkpoint is automatic development only.

    Automatic split 0 is the loss. Automatic split 1 is the checkpoint.
    An expert card is never a checkpoint row, including one whose split is 1.
    """
    loss = np.zeros(len(rows), dtype=bool)
    checkpoint = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        expert = row.get("origin") == "expert_seed" or str(row["id"]) in EXPERT_SEED_BY_ID
        code = int(row["split_code"])
        if expert or code == 0:
            loss[index] = True
        if code == 1 and not expert:
            checkpoint[index] = True
    return loss, checkpoint


def meter_columns(x_m: np.ndarray, left: float, right: float) -> tuple[int, int]:
    c0 = int(np.searchsorted(x_m, float(left) - 1e-3, side="left"))
    c1 = int(np.searchsorted(x_m, float(right) + 1e-3, side="right"))
    c0 = max(0, min(c0, len(x_m) - 1))
    c1 = max(c0 + 1, min(c1, len(x_m)))
    return c0, c1


def resize_nearest(patch: np.ndarray, out_h: int = CROP_H, out_w: int = CROP_W) -> np.ndarray:
    height, width = patch.shape
    rows = np.linspace(0, height - 1, out_h).astype(np.int32)
    cols = np.linspace(0, width - 1, out_w).astype(np.int32)
    return np.ascontiguousarray(patch[rows][:, cols])


def residual_crop(
    clean: np.ndarray,
    valid: np.ndarray,
    t0: int,
    t1: int,
    c0: int,
    c1: int,
) -> np.ndarray | None:
    """median_x3 crop minus its own valid median, resized to 32x64.

    The scale is a fixed 5 °C. Empty crops return None. This is the research
    input, not an alarm.
    """
    if t1 <= t0 or c1 <= c0:
        return None
    patch = np.asarray(clean[t0:t1, c0:c1], dtype=np.float32)
    mask = np.asarray(valid[t0:t1, c0:c1], dtype=bool) & np.isfinite(patch)
    if patch.size == 0 or int(mask.sum()) * 2 < patch.size:
        return None
    center = float(np.median(patch[mask]))
    residual = np.zeros(patch.shape, dtype=np.float32)
    residual[mask] = patch[mask] - center
    scaled = np.clip(residual / RESIDUAL_SCALE_C, -8.0, 8.0)
    return resize_nearest(scaled)


def class_mix(names: list[str]) -> dict:
    counts = {name: 0 for name in LABELS}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return {"n": len(names), "counts": counts}


def baseline_overlap(hits: np.ndarray, positive: np.ndarray) -> dict:
    """Jaccard of positive network windows and windows that meet a zone.

    `hits[i]` is true when window i intersects an operational zone.
    `positive[i]` is true when the network call is not Background.
    """
    hits = np.asarray(hits, dtype=bool)
    positive = np.asarray(positive, dtype=bool)
    if hits.shape != positive.shape:
        raise ValueError("hits and positive must match")
    n_positive = int(positive.sum())
    n_zone = int(hits.sum())
    intersection = int(np.count_nonzero(hits & positive))
    union = n_positive + n_zone - intersection
    overlap = float(intersection / union) if union else 0.0
    hit_fraction = float(intersection / n_positive) if n_positive else 0.0
    return {
        "n_windows": int(hits.size),
        "n_positive": n_positive,
        "n_zone_windows": n_zone,
        "intersection": intersection,
        "union": union,
        "overlap_jaccard": overlap,
        "positive_hit_fraction": hit_fraction,
    }


def window_hits(windows: np.ndarray, zones: np.ndarray) -> np.ndarray:
    """True when a half-open [t0, t1, c0, c1) window meets any zone box."""
    windows = np.asarray(windows, dtype=np.int32)
    zones = np.asarray(zones, dtype=np.int32)
    hits = np.zeros(len(windows), dtype=bool)
    if len(windows) == 0 or len(zones) == 0:
        return hits
    for start in range(0, len(windows), 2048):
        chunk = windows[start:start + 2048]
        time = (chunk[:, None, 0] < zones[None, :, 1]) & (zones[None, :, 0] < chunk[:, None, 1])
        space = (chunk[:, None, 2] < zones[None, :, 3]) & (zones[None, :, 2] < chunk[:, None, 3])
        hits[start:start + len(chunk)] = (time & space).any(axis=1)
    return hits


if nn is not None:
    class _ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.act = nn.GELU()

        def forward(self, x):
            return x + self.conv2(self.act(self.conv1(self.act(x))))

    class ResearchResidualCNN(nn.Module):
        """Residual 2D classifier. Six class scores. Not an alarm."""

        def __init__(self, n_class: int = 6):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU())
            self.block1 = _ResidualBlock(16)
            self.down = nn.Conv2d(16, 32, 3, stride=2, padding=1)
            self.block2 = _ResidualBlock(32)
            self.head = nn.Linear(32, n_class)
            self.act = nn.GELU()

        def forward(self, x):
            hidden = self.block1(self.stem(x))
            hidden = self.block2(self.act(self.down(hidden)))
            pooled = hidden.mean(dim=(2, 3))
            return self.head(pooled)

    def encode_context(context):
        """Map the three raw numbers onto the context branch.

        Column 1 of the raw input is the far-cable median. When that value
        is missing it stays out of the branch: the far slot is 0 and the
        present bit is 0. The rectangle median is not written into that slot.
        """
        if context.ndim != 2 or int(context.shape[1]) != 3:
            raise ValueError("context must be rectangle median, far-cable median, halo fraction")
        rect = context[:, 0] / CONTEXT_SCALE_C
        far = context[:, 1]
        halo = context[:, 2]
        present = torch.isfinite(far)
        far_value = torch.where(present, far / CONTEXT_SCALE_C, torch.zeros_like(far))
        return torch.stack([rect, far_value, halo, present.float()], dim=1)

    class ResearchContextCNN(nn.Module):
        """Residual crop plus rectangle, far cable, and halo fraction.

        Six class scores. serve=research. Not an alarm.
        """

        def __init__(self, n_class: int = 6):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU())
            self.block1 = _ResidualBlock(16)
            self.down = nn.Conv2d(16, 32, 3, stride=2, padding=1)
            self.block2 = _ResidualBlock(32)
            self.act = nn.GELU()
            self.context = nn.Sequential(nn.Linear(4, 32), nn.GELU())
            self.head = nn.Linear(64, n_class)

        def forward(self, crop, context):
            hidden = self.block1(self.stem(crop))
            hidden = self.block2(self.act(self.down(hidden)))
            pooled = hidden.mean(dim=(2, 3))
            extra = self.context(encode_context(context))
            return self.head(torch.cat([pooled, extra], dim=1))
else:  # pragma: no cover
    ResearchResidualCNN = None  # type: ignore[misc, assignment]
    ResearchContextCNN = None  # type: ignore[misc, assignment]
    encode_context = None  # type: ignore[misc, assignment]


def load_research_cnn(path: Path, device: str | None = None):
    """Load serve=research weights. Refuses an operational blob."""
    if torch is None or ResearchResidualCNN is None:
        raise RuntimeError("torch is required to load the research CNN")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob.get("serve") != "research":
        raise ValueError("research CNN serve must be research")
    if blob.get("operational") is True or blob.get("trained_classifier_operational") is True:
        raise ValueError("research CNN is not operational")
    if blob.get("architecture") != "residual2d":
        raise ValueError("research CNN architecture must be residual2d")
    model = ResearchResidualCNN(n_class=int(blob.get("n_class", 6)))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob, device


def load_research_context_cnn(path: Path, device: str | None = None):
    """Load the context research weights. Refuses an operational blob."""
    if torch is None or ResearchContextCNN is None:
        raise RuntimeError("torch is required to load the research context CNN")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob.get("serve") != "research":
        raise ValueError("research context CNN serve must be research")
    if blob.get("operational") is True or blob.get("trained_classifier_operational") is True:
        raise ValueError("research context CNN is not operational")
    if blob.get("architecture") != CONTEXT_ARCHITECTURE:
        raise ValueError("research context CNN architecture must be residual2d_context")
    if int(blob.get("n_class", 6)) != len(LABELS):
        raise ValueError("research context CNN must score the six class names")
    model = ResearchContextCNN(n_class=len(LABELS))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob, device
