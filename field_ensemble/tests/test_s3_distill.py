"""Дистилляция: симплекс s/h, потеря, отбор файлов, запрет записи в учителя, ворота победы."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dts import distill, metrics as M
import s3_sel_core as sel


def test_probs_roundtrip_and_simplex():
    p_bg = torch.tensor([0.2, 0.7, 0.0])
    p_hot = torch.tensor([0.5, 0.2, 0.4])
    p_cold = torch.tensor([0.3, 0.1, 0.6])
    s = 1.0 - p_bg
    h = p_hot - p_cold
    recon = distill.probs_from_sh(s, h)
    assert recon.shape == (3, 3)
    assert torch.allclose(recon.sum(0), torch.ones(3), atol=1e-5)
    assert torch.allclose(recon[0], p_bg, atol=1e-5)
    assert torch.allclose(recon[1], p_hot, atol=1e-5)
    assert torch.allclose(recon[2], p_cold, atol=1e-5)
    # float16-шум: h шире, чем позволяет s. Доли не отрицательные и снова на симплексе.
    noisy = distill.probs_from_sh(torch.tensor([0.2, 0.9]), torch.tensor([0.8, -1.4]))
    assert torch.all(noisy >= 0)
    assert torch.allclose(noisy.sum(0), torch.ones(2), atol=1e-5)


def test_map_axis_matches_logits():
    s = torch.rand(2, 5, 6)
    h = (torch.rand(2, 5, 6) - 0.5) * s
    p = distill.probs_from_sh(s, h)
    assert p.shape == (2, 3, 5, 6)
    assert torch.allclose(p.sum(1), torch.ones(2, 5, 6), atol=1e-5)
    assert torch.allclose(1.0 - p[:, 0], s, atol=1e-5)


def test_loss_lower_when_student_matches_teacher():
    torch.manual_seed(0)
    teacher = torch.softmax(torch.randn(2, 3, 8, 8), dim=1)
    s = 1.0 - teacher[:, 0]
    h = teacher[:, 1] - teacher[:, 2]
    recon = distill.probs_from_sh(s, h)
    assert torch.allclose(recon, teacher, atol=1e-5)
    y = teacher.argmax(dim=1)
    good = teacher.clamp(min=1e-6).log().detach().requires_grad_(True)
    bad = torch.zeros_like(good)
    loss_good, parts_good = distill.distill_loss(good, teacher, y, temperature=2.0)
    loss_bad, _parts_bad = distill.distill_loss(bad, teacher, y, temperature=2.0)
    assert float(loss_good) < float(loss_bad)
    assert parts_good["kl"] < 1e-4
    y_ign = torch.full_like(y, -1)
    loss_ign, parts_ign = distill.distill_loss(good, teacher, y_ign)
    assert float(loss_ign) == 0.0
    assert parts_ign == dict(kl=0.0, bce=0.0, hard=0.0)
    loss_ign.backward()
    assert torch.all(good.grad == 0)


def test_hard_term_is_light_and_finite():
    logits = torch.zeros(1, 3, 4, 4, requires_grad=True)
    teacher = torch.softmax(torch.randn(1, 3, 4, 4), dim=1)
    y = torch.zeros(1, 4, 4, dtype=torch.long)
    full, parts = distill.distill_loss(logits, teacher, y, w_hard=0.25)
    bare, parts_bare = distill.distill_loss(logits, teacher, y, w_hard=0.0)
    assert torch.isfinite(full)
    assert abs(float(full - bare) - 0.25 * parts["hard"]) < 1e-5
    assert parts_bare["hard"] == parts["hard"]
    assert distill.STUDENT_WIDTH <= distill.WIDTH_MAX == 24


def test_pixel_counts_match_confusion():
    s = np.array([[0.1, 0.9], [0.2, 0.05]], np.float32)
    ev = np.array([[1, 0], [0, 0]])
    nu = np.array([[0, 1], [0, 0]])
    valid = np.ones_like(ev, bool)
    got = distill.pixel_counts(s, ev, nu, valid, 0.5)
    pred = (s >= 0.5) & valid
    truth = ev > 0
    tp, fp, fn = M.confusion(pred, truth, valid)
    assert (got["tp"], got["fp"], got["fn"]) == (tp, fp, fn)
    assert got["far_n"] == 0 and got["far_d"] == 2
    frames = [dict(ev=s[ev > 0], nu=s[(ev == 0) & (nu > 0)], bg=s[(ev == 0) & (nu == 0)])]
    stats = distill.iou_from_groups(frames, far=0.5)
    assert 0.0 <= stats["iou"] <= 1.0
    assert stats["n_bg"] == 2


def test_mean_scores_is_equal_average():
    a = (np.array([0.0, 1.0], np.float32), np.array([0.2, -0.2], np.float32))
    b = (np.array([0.4, 0.6], np.float32), np.array([0.0, 0.4], np.float32))
    s, h = distill.mean_scores([a, b])
    assert np.allclose(s, [0.2, 0.8])
    assert np.allclose(h, [0.1, 0.1])


def test_teacher_row_round_robin_skips_huge():
    rows = []
    for ver in ("v8", "v10"):
        for tmpl, n in (("F2", 3), ("F5", 1)):
            for i in range(n):
                rows.append(dict(ver=ver, template=tmpl, stem=f"{ver}-{tmpl}-{i}", n_pix=1000 + i, proxy=False))
    rows.append(dict(ver="v8", template="F2", stem="huge", n_pix=10**9, proxy=False))
    picked = distill.select_teacher_rows(rows, k=6, max_pixels=2_200_000)
    assert len(picked) == 6
    assert all(r["n_pix"] <= 2_200_000 for r in picked)
    assert all(r["stem"] != "huge" for r in picked)
    keys = [(r["ver"], r["template"]) for r in picked[:4]]
    assert len(set(keys)) == 4
    train, proxy = distill.split_train_proxy(picked, min_proxy=2, min_train=4)
    assert len(proxy) == 2 and len(train) == 4
    assert distill.is_proxy("v8", "seq0000") == distill.is_proxy("v8", "seq0000")


def test_safe_output_refuses_teacher_and_release(tmp_path):
    root = ROOT
    ok = distill.assert_safe_output("artifacts/eval/s3_distill_v8.json", root)
    assert ok.name == "s3_distill_v8.json"
    outside = distill.assert_safe_output(tmp_path / "x.json", root)
    assert outside.name == "x.json"
    for bad in (
        "artifacts/stage1/unet_v810ft.pt",
        "models_release/det0_unet.pt",
        "models_release_dom/unet_dom.pt",
        "artifacts/pred/v8/ens_u2ft_tcn_lgbm/val/seq0006.npz",
    ):
        try:
            distill.assert_safe_output(bad, root)
        except RuntimeError:
            continue
        raise AssertionError(bad)


def test_promotion_needs_every_gate():
    good = dict(
        student_val=0.390,
        winner_val_recomputed=distill.WINNER_VAL,
        nested_delta5=0.004,
        v10=distill.WINNER_V10,
        v11=distill.WINNER_V11,
        winner_test_recomputed=distill.WINNER_TEST,
        winner_v10_recomputed=distill.WINNER_V10,
        winner_v11_recomputed=distill.WINNER_V11,
    )
    assert distill.promotion_decision(good)["beats"] is True
    low_val = dict(good, student_val=distill.WINNER_VAL + 5e-5)
    assert distill.promotion_decision(low_val)["beats"] is False
    low_nested = dict(good, nested_delta5=0.0029)
    assert distill.promotion_decision(low_nested)["beats"] is False
    assert distill.promotion_decision(low_nested)["reason"].find("nested") >= 0
    bad_v10 = dict(good, v10=distill.WINNER_V10 - 0.01)
    assert distill.promotion_decision(bad_v10)["beats"] is False
    bad_protocol = dict(good, winner_val_recomputed=0.30)
    got = distill.promotion_decision(bad_protocol)
    assert got["beats"] is False and got["protocol_ok"] is False
    missing_nested = dict(good)
    missing_nested["nested_delta5"] = None
    assert distill.promotion_decision(missing_nested)["beats"] is False
    far = dict(good, student_val=0.30, nested_delta5=-0.02)
    assert distill.promotion_decision(far)["close"] is False
    off_v11 = dict(good, winner_v11_recomputed=distill.WINNER_V11 - 0.01)
    assert distill.promotion_decision(off_v11)["beats"] is False


def test_nested_fold_api_matches_sel2():
    frames = []
    for level in (0.05, 0.2, 0.8, 0.95):
        frames.append(dict(
            ev_m=np.array([[0.99, 0.4]], np.float32),
            nu_m=np.zeros((1, 2), np.float32),
            bg_m=np.full((1, 400), level, np.float32),
            n_bg=400,
            cooling=[],
        ))
    assert sel.sequence_folds(4, 2) == [[0, 2], [1, 3]]
    parts = [sel.evaluate_frames([frames[i] for i in fold], np.array([1.0])) for fold in sel.sequence_folds(4, 2)]
    micro = sel.micro_iou(parts)
    assert 0.0 <= micro <= 1.0
    verdict = sel.nested_verdict({"2": micro, "5": micro})
    assert verdict["accept"] is (micro > sel.NESTED_BAR)
    assert sel.NESTED_BAR == pytest_bar()


def pytest_bar():
    return 0.3867 + 0.003
