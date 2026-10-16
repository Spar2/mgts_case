"""ST2: эмбеддинги энкодера и GroupKFold без утечки групп."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("s3_st2_embed", ROOT / "scripts" / "s3_st2_embed.py")
st2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st2)

from dts import events as E


def test_scalar_pack_matches_taxonomy_widths():
    img = np.zeros((3, 3, 32, 64), np.float32)
    img[:, 0, :, 20:28] = 1.5
    base = np.zeros((3, len(E.SCALAR_NAMES)), np.float32)
    ctx = np.zeros((3, len(E.CONTEXT_NAMES)), np.float32)
    packed = st2.pack_scalars(base, img, ctx)
    assert packed.shape == (3, len(E.SCALAR_NAMES) + len(E.SHAPE_NAMES) + len(E.CONTEXT_NAMES))
    assert np.isfinite(packed).all()


def test_unet_embed_shape_is_deterministic_and_sees_flip():
    torch = pytest.importorskip("torch")
    from dts.models import DTSUNet

    torch.manual_seed(0)
    net = DTSUNet(w=4)
    net.eval()
    x = torch.zeros(2, 3, 32, 64)
    x[0, 0, :, :8] = 3
    x[1, 0, :, 40:] = 3
    with torch.no_grad():
        a = st2.unet_embed(net, x)
        b = st2.unet_embed(net, x)
        flipped = st2.unet_embed(net, torch.flip(x, [3]))
    # e2=8, e3=16, e4=32 каналов, три пула (среднее, std, центр−край)
    assert a.shape == (2, 3 * (8 + 16 + 32))
    assert torch.allclose(a, b)
    assert not torch.allclose(a, flipped)


def test_group_folds_cover_episodes_without_overlap():
    groups = np.array([f"seq{i // 3:04d}" for i in range(30)])
    idx = np.arange(30)
    folds = st2.iter_group_folds(idx, groups, n_splits=5)
    assert len(folds) == 5
    seen = []
    for tr, va in folds:
        assert not set(groups[tr]) & set(groups[va])
        assert len(np.intersect1d(tr, va)) == 0
        seen.append(va)
    covered = np.sort(np.concatenate(seen))
    assert np.array_equal(covered, idx)


def test_score_perfect_prediction_and_present_macro():
    y = np.array([0, 1, 2, 3, 4, 5, 6, 0])
    prob = np.eye(7, dtype=np.float32)[y]
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    r = st2.score_predictions(y, y, prob, groups, E.CLASSES)
    assert r["macro_f1"] == pytest.approx(1.0)
    assert r["coarse_macro_f1"] == pytest.approx(1.0)
    assert r["n"] == 8
    assert st2.present_macro(np.array([0, 0, 1]), np.array([0, 1, 1])) < 1.0


def test_fusion_fit_is_deterministic_and_emits_probabilities():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    n = 40
    img = rng.normal(size=(n, 3, 32, 64)).astype(np.float32)
    scal = rng.normal(size=(n, 35)).astype(np.float32)
    unet = rng.normal(size=(n, 12)).astype(np.float32)
    unet_flip = unet + 0.1
    y = np.array([i % 7 for i in range(n)], np.int64)
    groups = np.array([f"g{i // 10}" for i in range(n)])
    idx = np.arange(n)

    def once(seed):
        net, mu_s, sd_s, mu_u, sd_u, info = st2.fit_embed_fusion(
            img, scal, unet, unet_flip, y, groups, idx,
            seed=seed, epochs=2, patience=2, min_epochs=2, batch=16, use_inner=False, n_classes=7)
        prob = st2.predict_probs(net, img, scal, unet, unet_flip, mu_s, sd_s, mu_u, sd_u, idx)
        return prob, info

    p1, info = once(0)
    p2, _ = once(0)
    assert p1.shape == (n, 7)
    assert np.allclose(p1.sum(1), 1, atol=1e-5)
    assert np.allclose(p1, p2, atol=1e-6)
    assert info["epochs_ran"] == 2
    assert info["n_inner"] == 0


def test_crops_from_cache_keep_synth_label(tmp_path):
    cache = tmp_path / "cache" / "val"
    cache.mkdir(parents=True)
    x = np.arange(40) * 0.25
    t = np.arange(30) * 120.0
    X = np.zeros((3, 30, 40), np.float32)
    X[2] = 1
    X[0, 8:18, 10:18] = 2
    np.savez(cache / "seq0007.npz", X=X, x=x, t=t, sigma=np.ones(40, np.float32))
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "seq_id,t_start,t_end,dist_start_m,dist_end_m,label\n"
        "seq0007,1970-01-01T00:16:00+00:00,1970-01-01T00:36:00+00:00,2.5,4.5,hotspot\n"
        "seq0007,1970-01-01T00:16:00+00:00,1970-01-01T00:40:00+00:00,6.0,6.4,tracing\n"
    )
    (tmp_path / "sequences.csv").write_text("seq_id,template\nseq0007,F5\n")
    block = st2.crops_from_cache(tmp_path / "cache", labels, ("val",))
    assert block["img"].shape[0] == 2
    assert block["img"].shape[1:] == (3, 32, 64)
    assert block["y"].tolist() == [E.CLASSES.index("Local_Heating"), E.CLASSES.index("Contact_Trace")]
    assert set(block["groups"]) == {"seq0007"}
    assert block["template"].tolist() == ["F5", "F5"]
    assert block["ctx"].shape == (2, len(E.CONTEXT_NAMES))
