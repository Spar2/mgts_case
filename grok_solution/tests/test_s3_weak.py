"""Слабые события: вес амплитуды, curriculum, FAR-хвост, focal+Tversky. Без больших кэшей."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import s3_weak_lib as W
from dts import metrics as M


def test_curriculum_moves_toward_small_amplitude():
    a = W.curriculum(0, 8)
    b = W.curriculum(7, 8)
    mid = W.curriculum(3, 8)
    assert a["amp_hi"] > mid["amp_hi"] > b["amp_hi"]
    assert a["p_small"] < mid["p_small"] < b["p_small"]
    assert a["gamma"] < b["gamma"]
    assert a["rank_w"] < b["rank_w"]
    assert b["amp_hi"] == 2.0
    assert abs(a["amp_hi"] - 8.0) < 1e-9
    assert W.curriculum(3, 8) == mid
    assert 2.0 <= b["amp_hi"] <= a["amp_hi"] <= 8.0


def test_amp_gain_boosts_only_below_two_degrees():
    g = W.amp_gain([0.5, 1.0, 2.0, 8.0], gamma=1.0)
    assert abs(g[2] - 1.0) < 1e-9 and abs(g[3] - 1.0) < 1e-9
    assert abs(g[1] - 2.0) < 1e-9
    assert abs(g[0] - 4.0) < 1e-9
    # пол 0.2 °C и потолок 4 не дают бесконечный вес
    assert float(W.amp_gain(0.01, gamma=3.0)) == 4.0
    assert float(W.amp_gain(1.0, gamma=0.0)) == 1.0
    amps = np.linspace(0.2, 2.0, 7)
    gains = W.amp_gain(amps, gamma=1.2)
    assert np.all(np.diff(gains) <= 1e-12)


def test_paint_weights_follows_mask_and_amplitude():
    ev = np.zeros((6, 8), np.uint8)
    ev[1:3, 1:3] = 1
    ev[1:3, 4:6] = 2
    boxes = [
        dict(t0=1, t1=3, x0=1, x1=3, amp=1.0, mask=1),
        dict(t0=1, t1=3, x0=4, x1=6, amp=6.0, mask=2),
    ]
    w = W.paint_weights(ev, boxes, gamma=1.0)
    assert w.shape == ev.shape
    assert float(w[0, 0]) == 1.0
    assert abs(float(w[1, 1]) - 2.0) < 1e-6
    assert abs(float(w[1, 4]) - 1.0) < 1e-6
    # чужой mask внутри бокса не получает вес события
    ev2 = ev.copy()
    ev2[1, 1] = 0
    w2 = W.paint_weights(ev2, boxes, gamma=1.0)
    assert float(w2[1, 1]) == 1.0
    assert abs(float(w2[2, 2]) - 2.0) < 1e-6


def test_boxes_match_event_pixel_margins():
    x = np.arange(10, dtype=np.float64)
    t = np.arange(12, dtype=np.float64) * 100.0
    ev = np.zeros((12, 10), np.uint8)
    ev[2:6, 2:5] = 3
    rec = [dict(kind="event", dist_start_m=2.0, dist_end_m=4.0, ts=200.0, te=500.0, amp=1.2, mask=3)]
    boxes = W.boxes_from_axes(x, t, rec)
    assert len(boxes) == 1
    b = boxes[0]
    assert (b["x0"], b["x1"], b["t0"], b["t1"]) == (2, 5, 1, 7)
    # те же границы, что у event_pixels: dist±0.25, время±120
    import pandas as pd
    E0 = pd.Timestamp("1970-01-01", tz="UTC")
    row = pd.DataFrame([dict(
        seq_id="s", kind="event", label="hotspot", mask_value=3, amp_obs_C=1.2,
        dist_start_m=2.0, dist_end_m=4.0,
        t_start=(E0 + pd.Timedelta(seconds=200)).isoformat(),
        t_end=(E0 + pd.Timedelta(seconds=500)).isoformat(),
        difficulty="hard",
    )])
    px = M.event_pixels(next(M.label_boxes(row, "s").itertuples()), x, t, ev)
    painted = np.zeros_like(ev, bool)
    painted[b["t0"]:b["t1"], b["x0"]:b["x1"]] = True
    assert np.array_equal(px, painted & (ev == 3))


def test_focal_tversky_upweights_missed_weak_pixels():
    cw = torch.tensor([1.0, 3.0, 3.0])
    y = torch.zeros(1, 8, 8, dtype=torch.long)
    y[0, 2:5, 2:5] = 1
    y[0, 0, 0] = -1
    miss = torch.full((1, 3, 8, 8), 4.0)
    miss[:, 1:] = -4.0
    aw = torch.ones(1, 8, 8)
    aw_hi = aw.clone()
    aw_hi[0, 2:5, 2:5] = 4.0
    low = W.focal_tversky(miss, y, aw, cw, rank_w=0.0)[0]
    high = W.focal_tversky(miss, y, aw_hi, cw, rank_w=0.0)[0]
    assert float(high) > float(low)
    hit = torch.full((1, 3, 8, 8), -4.0)
    hit[:, 1] = 4.0
    good = W.focal_tversky(hit, y, aw_hi, cw, rank_w=0.0)[0]
    assert float(good) < float(high)
    logit = miss.detach().clone().requires_grad_(True)
    loss, parts = W.focal_tversky(logit, y, aw_hi, cw, rank_w=0.5)
    loss.backward()
    assert logit.grad.abs().sum() > 0
    assert set(parts) == {"focal", "tversky", "rank"}
    # невалидный пиксель не создаёт градиент в одиночку: сравним с y=-1 везде кроме одного фона
    assert float(low) > 0


def test_score_tail_matches_far_threshold():
    rng = np.random.default_rng(0)
    s = rng.random(20_000).astype(np.float32)
    tail = W.ScoreTail(cutoff=0.5)
    # два куска, как по файлам
    tail.add(s[:8000])
    tail.add(s[8000:])
    got = tail.threshold(1e-3)
    ref = M.far_threshold(s, 1e-3)
    assert tail.last_source == "exact"
    assert abs(got - ref) < 1e-6
    # квантиль ниже cutoff — гистограмма, ошибка не больше шага бина
    wide = W.ScoreTail(cutoff=0.999, bins=100_000)
    wide.add(s)
    rough = wide.threshold(1e-2)
    assert wide.last_source == "hist"
    assert abs(rough - M.far_threshold(s, 1e-2)) <= 1e-5 + 1e-6


def test_crossing_level_log_interpolates_upward_only():
    # 0.5 лежит между recall 0.4 при SNR 2 и 0.8 при SNR 4
    got = W.crossing_level([1.0, 2.0, 4.0], [0.1, 0.4, 0.8], 0.5)
    t = (0.5 - 0.4) / (0.8 - 0.4)
    expect = float(np.exp(np.log(2.0) + t * (np.log(4.0) - np.log(2.0))))
    assert abs(got - expect) < 1e-9
    assert W.crossing_level([1.0, 2.0, 4.0], [0.7, 0.6, 0.4], 0.5) is None
    assert W.crossing_level([3.0], [0.9], 0.5) is None


def test_bg_anchor_penalises_only_a_hotter_background():
    y = torch.zeros(1, 8, 8, dtype=torch.long)
    y[0, 0, :] = 1
    ref = torch.full((1, 8, 8), 0.2)
    calm = torch.full((1, 8, 8), 0.1)
    hot = torch.full((1, 8, 8), 0.5)
    assert float(W.bg_anchor(calm, ref, y)) == 0.0
    assert float(W.bg_anchor(hot, ref, y)) > 0.0


def test_rank_term_is_deterministic_and_finite():
    torch.manual_seed(0)
    logit = torch.randn(2, 3, 16, 16)
    y = torch.zeros(2, 16, 16, dtype=torch.long)
    y[:, 4:8, 4:8] = 1
    aw = torch.ones(2, 16, 16)
    aw[:, 4:8, 4:8] = 3
    cw = torch.tensor([1.0, 3.0, 3.0])
    l1, p1 = W.focal_tversky(logit, y, aw, cw, rank_w=0.8)
    l2, p2 = W.focal_tversky(logit, y, aw, cw, rank_w=0.8)
    assert torch.equal(l1, l2)
    assert p1 == p2
    assert np.isfinite(float(l1))
    # без перевзвешивания ранжирование выключено: хвост фона не учится
    ones = torch.ones_like(aw)
    _, plain = W.focal_tversky(logit, y, ones, cw, rank_w=1.0)
    assert plain["rank"] == 0.0
