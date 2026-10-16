from fiber.benchmark.channels import score_local_channel
from fiber.benchmark.synthetic import local_scene


def test_clean_channels_keep_planted_hot_and_cold():
    scene = local_scene()
    for method in ("identity", "median_x3", "tv_x", "bilateral_x"):
        block = score_local_channel(method, scene)
        assert block["hot"]["event_recall"] == 1.0
        assert block["cold"]["event_recall"] == 1.0
        assert block["dead_tail_alarms"] == 0
        assert block["hot"]["predicted_invalid_cells"] == 0
