"""Refine stays inside the dilation and does not invent a zone."""

import numpy as np

from fiber.anomaly.detect import spatial_background, spatial_z
from fiber.anomaly.refine import dilate_mask, refine_zones, zones_to_mask
from fiber.anomaly.zones import extract_zones
from fiber.benchmark.events import evaluate_events
from fiber.benchmark.synthetic import local_scene_var
from fiber.denoise import apply_method
from fiber.pipeline import PipelineConfig


def _axes(shape):
    rows, cols = shape
    return np.arange(cols, dtype=np.float64), np.arange(rows, dtype=np.float64) * 60.0


def test_half_max_edge_is_within_one_sample_of_the_step():
    fine = np.zeros((20, 40), dtype=np.float32)
    fine[:, 10:20] = 4.0
    hot = np.zeros(fine.shape, dtype=bool)
    hot[:, 8:22] = True
    cold = np.zeros(fine.shape, dtype=bool)
    valid = np.ones(fine.shape, dtype=bool)
    x_m, time_s = _axes(fine.shape)
    refined, cold_out, _zones = refine_zones(
        hot, cold, fine, np.zeros_like(fine), valid, x_m, time_s, mode="half_max",
    )
    cols = np.flatnonzero(refined.any(axis=0))
    assert cols[0] >= 9 and cols[0] <= 11
    assert cols[-1] >= 18 and cols[-1] <= 20
    assert not cold_out.any()
    assert np.all(refined <= dilate_mask(hot, valid, x_m))


def test_region_grow_stays_inside_the_dilation():
    fine = np.zeros((20, 40), dtype=np.float32)
    fine[:, 10:20] = 4.0
    hot = np.zeros(fine.shape, dtype=bool)
    hot[:, 8:22] = True
    valid = np.ones(fine.shape, dtype=bool)
    x_m, time_s = _axes(fine.shape)
    refined, _cold, _zones = refine_zones(
        hot, np.zeros_like(hot), fine, np.zeros_like(fine), valid, x_m, time_s, mode="region_grow",
    )
    assert np.all(refined <= dilate_mask(hot, valid, x_m))


def test_empty_and_invalid_leave_a_bool_mask():
    shape = (4, 6)
    x_m, time_s = _axes(shape)
    hot, cold, zones = refine_zones(
        np.zeros(shape, dtype=bool),
        np.zeros(shape, dtype=bool),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
        np.ones(shape, dtype=bool),
        x_m,
        time_s,
    )
    assert hot.dtype == bool and cold.dtype == bool
    assert not hot.any() and zones == []
    assert not np.isnan(hot.astype(float)).any()

    fine = np.ones((6, 8), dtype=np.float32)
    fine[2, 3] = np.nan
    valid = np.ones(fine.shape, dtype=bool)
    valid[2, 3] = False
    mask = np.zeros(fine.shape, dtype=bool)
    mask[1:5, 1:6] = True
    x_m, time_s = _axes(fine.shape)
    refined, _cold, _zones = refine_zones(
        mask, np.zeros_like(mask), fine, np.zeros_like(fine), valid, x_m, time_s, mode="half_max",
    )
    assert refined.dtype == bool
    assert not refined[2, 3]
    assert not np.isnan(refined.astype(float)).any()


def test_ramp0_iou_does_not_fall():
    scene = local_scene_var(0, amp_hot=6.0, amp_cold=-5.0, width=7, noise=0.0, ramp=0)
    clean = apply_method("median_x3", scene["clean"], scene["valid"])
    score, background, _scale = spatial_z(clean, scene["valid"], scene["x_m"])
    config = PipelineConfig()
    iso = [str(float(stamp)) for stamp in scene["time_s"]]
    zones = extract_zones(
        score, scene["valid"], scene["x_m"], iso, scene["time_s"],
        thresh=config.threshold_z,
        min_width_m=config.min_width_m,
        min_profiles=config.min_profiles,
        min_cells=config.min_cells,
        max_gap_s=config.max_gap_s,
        contrast_c=clean.astype(np.float64) - np.asarray(background, dtype=np.float64),
    )
    hot = zones_to_mask(zones, scene["clean"].shape, scene["x_m"], scene["time_s"], polarity="hot")
    cold = zones_to_mask(zones, scene["clean"].shape, scene["x_m"], scene["time_s"], polarity="cold")
    fine_bg = spatial_background(scene["clean"], scene["valid"], scene["x_m"])
    for mode in ("half_max", "region_grow"):
        refined_hot, refined_cold, _records = refine_zones(
            hot, cold, scene["clean"], fine_bg, scene["valid"], scene["x_m"], scene["time_s"], mode=mode,
        )
        assert np.all(refined_hot <= dilate_mask(hot, scene["valid"], scene["x_m"]))
        assert np.all(refined_cold <= dilate_mask(cold, scene["valid"], scene["x_m"]))
        for truth, before, after in (
            (scene["hot"], hot, refined_hot),
            (scene["cold"], cold, refined_cold),
        ):
            before_iou = evaluate_events(truth, before, scene["time_s"], scene["x_m"], scene["valid"])["support_iou"]
            after_iou = evaluate_events(truth, after, scene["time_s"], scene["x_m"], scene["valid"])["support_iou"]
            assert before_iou is not None and after_iou is not None
            assert after_iou + 1e-9 >= before_iou
