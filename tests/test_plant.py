import numpy as np

from fiber.classify.plant import (
    KIND_TO_LABEL,
    LABEL_INDEX,
    LABELS,
    SoftmaxSceneModel,
    plant_scene,
    rule_scene_label,
    scene_features,
)
from fiber.models.patch_classifier import TIME_CROP, crop_deviation


def test_planted_labels_sit_on_valid_cells_only():
    for kind, name in KIND_TO_LABEL.items():
        scene = plant_scene(kind, seed=1)
        assert scene["label"] == name
        assert not scene["labels"][~scene["valid"]].any()
        if kind == "background":
            assert scene["labels"].sum() == 0
        else:
            assert set(np.unique(scene["labels"][scene["valid"]])) <= {0, LABEL_INDEX[name]}
            assert (scene["labels"] == LABEL_INDEX[name]).any()


def test_crop_deviation_removes_the_local_median():
    scene = plant_scene("background", seed=2)
    patch = crop_deviation(scene["field"], scene["valid"], 0, 0)
    assert patch.shape == (TIME_CROP, 40)
    assert abs(float(np.median(patch))) < 1.0


def test_rules_name_extended_events_and_drop_a_single_cell():
    for seed in range(3):
        assert rule_scene_label(plant_scene("local_heating", seed)) == "Local_Heating"
        assert rule_scene_label(plant_scene("local_cooling", seed)) == "Local_Cooling"
        assert rule_scene_label(plant_scene("periodic", seed)) == "Periodic_Thermal"
        assert rule_scene_label(plant_scene("broad_shift", seed)) == "Broad_Shift"
        assert rule_scene_label(plant_scene("background", seed)) == "Background"
        assert rule_scene_label(plant_scene("sensor_fault", seed)) == "Background"


def test_softmax_agrees_with_the_generator_on_new_seeds():
    def bundle(seeds):
        rows = []
        target = []
        for seed in seeds:
            for kind in KIND_TO_LABEL:
                scene = plant_scene(kind, seed)
                rows.append(scene_features(scene))
                target.append(LABEL_INDEX[scene["label"]])
        return np.stack(rows), np.array(target, dtype=np.int64)

    model = SoftmaxSceneModel()
    train_x, train_y = bundle(range(8))
    model.fit(train_x, train_y)
    test_x, test_y = bundle(range(8, 12))
    predicted = model.predict(test_x)
    recalls = []
    for name in LABELS:
        mask = [LABELS[int(index)] == name for index in test_y]
        hit = sum(pred == name for pred, keep in zip(predicted, mask, strict=True) if keep)
        recalls.append(hit / sum(mask))
    assert min(recalls) >= 0.75


def test_patch_classifier_outputs_six_logits():
    torch = __import__("pytest").importorskip("torch")
    from fiber.models.patch_classifier import PatchClassifier

    net = PatchClassifier()
    out = net(torch.zeros(2, 1, 28, 40))
    assert tuple(out.shape) == (2, 6)
