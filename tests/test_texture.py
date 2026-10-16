"""Texture cables keep painted labels and keep two events out of one crop."""

import numpy as np

from fiber.classify.plant import LABEL_INDEX
from fiber.classify.texture import TEXTURE_KINDS, TEXTURE_TRAIN_SEEDS, plant_texture
from fiber.models.patch_classifier import SPACE_CROP


def test_texture_recipe_is_frozen():
    assert len(TEXTURE_KINDS) == 3
    assert TEXTURE_KINDS[0]["amp"] == 0.35
    assert TEXTURE_KINDS[0]["noise"] == 0.18
    assert TEXTURE_KINDS[0]["periods"] == (24, 48, 96)
    assert TEXTURE_KINDS[1]["periods"] == (16, 32, 64)
    assert TEXTURE_KINDS[2]["periods"] == (8, 16, 32)
    assert TEXTURE_TRAIN_SEEDS[0] == 1_200_000
    assert len(TEXTURE_TRAIN_SEEDS) == 24
    assert 1_100_000 not in TEXTURE_TRAIN_SEEDS


def test_painted_texture_labels_are_the_generator_class():
    scene = plant_texture(1_200_007)
    assert scene["label_source"] == "generator"
    assert scene["texture_index"] == 1_200_007 % 3
    for event in scene["events"]:
        mask = scene["event_id"] == event["id"]
        assert mask.any()
        assert np.all(scene["labels"][mask] == LABEL_INDEX[event["label"]])
    assert not scene["labels"][~scene["valid"]].any()
    again = plant_texture(1_200_007)
    assert np.array_equal(scene["field"], again["field"])
    assert np.array_equal(scene["labels"], again["labels"])


def test_one_crop_holds_at_most_one_texture_event():
    for seed in (1_200_000, 1_200_001, 1_200_002):
        scene = plant_texture(seed)
        columns = []
        for x in range(scene["event_id"].shape[1]):
            columns.append({int(item) for item in np.unique(scene["event_id"][:, x]) if int(item) != 0})
        for x0 in range(0, scene["event_id"].shape[1] - SPACE_CROP + 1):
            union: set[int] = set()
            for x in range(x0, x0 + SPACE_CROP):
                union |= columns[x]
            assert len(union) <= 1
