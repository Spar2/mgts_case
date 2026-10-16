"""FBD: шум увеличивает Фреше-расстояние, seed фиксирует выборку и бутстреп."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import fbd, metrics

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models_release" / "det0_unet.pt"


def test_frechet_zero_on_identical_cloud():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 6))
    same = fbd.frechet_distance(a, a.copy(), shrink=False)
    assert same["fbd"] < 1e-6
    assert fbd.frechet_diag(a, a) < 1e-8


def test_fbd_grows_with_noise():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(180, 8))
    eps = rng.normal(size=ref.shape)
    small = fbd.frechet_distance(ref, ref + 0.15 * eps, shrink=False)["fbd"]
    large = fbd.frechet_distance(ref, ref + 1.8 * eps, shrink=False)["fbd"]
    assert large > small
    assert fbd.frechet_diag(ref, ref + 1.8 * eps) > fbd.frechet_diag(ref, ref + 0.15 * eps)


def test_bootstrap_and_seed_are_deterministic():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(50, 5))
    b = a + rng.normal(scale=0.4, size=a.shape)
    r1 = fbd.bootstrap_fbd(a, b, n_boot=12, seed=7, shrink=False)
    r2 = fbd.bootstrap_fbd(a, b, n_boot=12, seed=7, shrink=False)
    r3 = fbd.bootstrap_fbd(a, b, n_boot=12, seed=8, shrink=False)
    assert r1["fbd"] == r2["fbd"] == r3["fbd"]
    assert r1["ci95"] == r2["ci95"]
    assert r1["ci95"][0] <= r1["ci95"][1]
    assert np.isfinite(r1["fbd"])
    assert fbd.stable_seed(0, "v8", "seq") == fbd.stable_seed(0, "v8", "seq")
    assert fbd.stable_seed(0, "v8") != fbd.stable_seed(0, "v10")


def test_sample_patches_deterministic_and_avoids_zones():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3, 80, 90)).astype(np.float32)
    X[:, :, 40:] = 7
    valid = np.ones((80, 90), bool)
    ban = np.zeros((80, 90), bool)
    ban[:, 40:] = True
    a = fbd.sample_patches(X, valid, n=6, patch=(16, 24), seed=11, ban=ban)
    b = fbd.sample_patches(X, valid, n=6, patch=(16, 24), seed=11, ban=ban)
    c = fbd.sample_patches(X, valid, n=6, patch=(16, 24), seed=12, ban=ban)
    assert a.shape == (6, 3, 16, 24)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(a[:, 0] != 7)


def test_segment_ids_split_on_gap():
    t = np.array([0, 120, 240, 1000, 1120], np.float64)
    assert list(fbd.segment_ids(t, 300)) == [0, 0, 0, 1, 1]


def test_far_hist_matches_exact_quantile():
    rng = np.random.default_rng(4)
    scores = rng.random(80000)
    edges = np.linspace(0, 1, 20001)
    hist, _ = np.histogram(scores, bins=edges)
    exact = metrics.far_threshold(scores, 1e-3)
    approx = fbd.far_threshold_hist(hist, edges, 1e-3)
    assert abs(exact - approx) < 1e-4


def test_kl_separates_zone_from_quiet_and_is_deterministic():
    field = np.zeros((40, 30), np.float32)
    field[8:18, 4:12] = 0.85
    valid = np.ones_like(field, bool)
    ev = np.zeros(field.shape, np.uint8)
    ev[8:18, 4:12] = 1
    t = np.arange(40) * 120.0
    specs = fbd.zone_specs(ev, valid, t, min_cells=12, min_quiet_profiles=10)
    r1 = fbd.kl_zone_quiet(field, specs)
    r2 = fbd.kl_zone_quiet(field, specs)
    assert r1["hot"] == r2["hot"]
    summary = fbd.kl_summary(r1["hot"])
    assert summary["n"] == 1 and summary["mean"] > 0.5
    flat = fbd.kl_zone_quiet(np.zeros_like(field), specs)
    assert fbd.kl_summary(flat["hot"])["mean"] == pytest.approx(0.0, abs=1e-9)
    # холодная зона отдельно, горячей нет
    ev2 = np.zeros_like(ev)
    ev2[8:18, 4:12] = 5
    specs_c = fbd.zone_specs(ev2, valid, t, min_cells=12)
    cold = fbd.kl_summary(fbd.kl_zone_quiet(field, specs_c)["cold"])
    assert cold["n"] == 1 and cold["mean"] > 0.5


def test_selector_verdict_threshold():
    assert fbd.selector_verdict(0.61, 0.6) == "selector"
    assert fbd.selector_verdict(0.9, 0.4) == "diagnostics"
    assert fbd.selector_verdict(None, 0.9) == "diagnostics"


def test_encoder_fbd_grows_with_input_noise_and_is_deterministic():
    torch = pytest.importorskip("torch")
    if not CKPT.is_file():
        pytest.skip("det0_unet.pt is not in models_release")
    enc = fbd.BottleneckEncoder.load(CKPT)
    assert enc.width == 24 and enc.dim == 192
    rng = np.random.default_rng(5)
    n, th, tw = 16, 32, 64
    t = np.linspace(0, 1, th, dtype=np.float32)
    x = np.linspace(0, 1, tw, dtype=np.float32)
    clean = np.zeros((n, 3, th, tw), np.float32)
    for i in range(n):
        wave = np.sin(2 * np.pi * (1.2 + 0.15 * i) * t)[:, None] * np.cos(2 * np.pi * (0.8 + 0.1 * i) * x)[None, :]
        clean[i, 0] = np.clip(3.0 * wave, -12, 12)
        clean[i, 1] = np.clip(2.0 * wave, -12, 12)
        clean[i, 2] = 1
    eps = rng.normal(size=clean[:, :2].shape).astype(np.float32)
    small = clean.copy()
    large = clean.copy()
    small[:, :2] = np.clip(clean[:, :2] + 0.05 * eps, -12, 12)
    large[:, :2] = np.clip(clean[:, :2] + 3.0 * eps, -12, 12)
    f0 = enc.embed(clean, batch=8)
    f_small = enc.embed(small, batch=8)
    f_large = enc.embed(large, batch=8)
    again = enc.embed(clean, batch=8)
    assert np.allclose(f0, again)
    d_small = fbd.frechet_distance(f0, f_small, shrink=True)["fbd"]
    d_large = fbd.frechet_distance(f0, f_large, shrink=True)["fbd"]
    assert np.isfinite(d_small) and np.isfinite(d_large)
    assert d_large > d_small
    assert fbd.frechet_diag(f0, f_large) > fbd.frechet_diag(f0, f_small)
    assert torch is not None


def test_kid_grows_with_noise_and_matches_twice():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(160, 8))
    b = rng.normal(size=(160, 8))
    eps = rng.normal(size=b.shape)
    same = fbd.kid_mmd(a, b)
    mild = fbd.kid_mmd(a, b + 0.25 * eps)
    strong = fbd.kid_mmd(a, b + 2.0 * eps)
    assert np.isfinite(same) and np.isfinite(mild) and np.isfinite(strong)
    assert strong > mild > same
    assert fbd.kid_mmd(a, b + 0.25 * eps) == mild


def test_fd_infinity_ranks_shift_and_is_deterministic():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(120, 4))
    b = rng.normal(size=(120, 4))
    c = rng.normal(size=(120, 4)) + 1.5
    same = fbd.fd_infinity(a, b, seed=3, n_sizes=4, n_repeats=4, shrink=False)
    shift = fbd.fd_infinity(a, c, seed=3, n_sizes=4, n_repeats=4, shrink=False)
    again = fbd.fd_infinity(a, b, seed=3, n_sizes=4, n_repeats=4, shrink=False)
    assert same["fd_inf"] is not None and shift["fd_inf"] is not None
    assert shift["fd_inf"] > same["fd_inf"] + 1.0
    assert same["fd_inf"] == again["fd_inf"]
    assert same["r2"] > 0.5
    too_small = fbd.fd_infinity(a[:6], b[:6], seed=0, n_sizes=4, n_repeats=2, shrink=False)
    assert too_small["fd_inf"] is None


def test_domain_probe_separates_shift_not_matched_noise():
    rng = np.random.default_rng(13)

    def cloud(loc, seed):
        rr = np.random.default_rng(seed)
        feats, groups = [], []
        for g in range(8):
            feats.append(rr.normal(loc=loc, scale=1.0, size=(16, 6)))
            groups.extend([f"g{g}"] * 16)
        return np.concatenate(feats), groups

    a, ga = cloud(0.0, 1)
    b, gb = cloud(0.0, 2)
    c, gc = cloud(2.5, 3)
    near = fbd.domain_probe_auc(a, ga, b, gb, n_splits=4, seed=0)
    far = fbd.domain_probe_auc(a, ga, c, gc, n_splits=4, seed=0)
    assert near["auc_oof"] is not None and far["auc_oof"] is not None
    assert far["separation"] > 0.9
    assert near["separation"] < 0.75
    assert far["separation"] > near["separation"]
    again = fbd.domain_probe_auc(a, ga, c, gc, n_splits=4, seed=0)
    assert again["auc_oof"] == far["auc_oof"]


def test_balanced_assembly_matches_zone_fraction():
    rng = np.random.default_rng(14)

    def pack(nfiles, nz, nq):
        return {
            "ids": [f"f{i}" for i in range(nfiles)],
            "zone": [rng.normal(size=(nz, 3)) for _ in range(nfiles)],
            "quiet": [rng.normal(size=(nq, 3)) for _ in range(nfiles)],
        }

    pa, pb = pack(4, 10, 7), pack(3, 8, 9)
    fa, ga, fb, gb, info = fbd.assemble_balanced(pa, pb, seed=0)
    assert info["n_zone"] == 24
    assert info["n_quiet"] == 27
    assert len(fa) == len(fb) == info["n"] == 51
    assert len(ga) == len(fa) and len(gb) == len(fb)
    assert abs(info["zone_fraction"] - 24 / 51) < 1e-12
    assert len(set(ga)) > 1 and len(set(gb)) > 1
    fa2, _, fb2, _, info2 = fbd.assemble_balanced(pa, pb, seed=0)
    assert info2 == info
    assert np.array_equal(fa, fa2) and np.array_equal(fb, fb2)
    report = fbd.gap_report(fa, ga, fb, gb, seed=1, n_sizes=3, n_repeats=2)
    assert np.isfinite(report["kid"])
    assert report["n_a"] == report["n_b"] == 51


def test_kid_balanced_ranks_shifted_file_and_matches_fraction():
    rng = np.random.default_rng(21)

    def pack(nfiles, nz, nq, shift):
        return {
            "ids": [f"f{i}" for i in range(nfiles)],
            "zone": [rng.normal(loc=shift, size=(nz, 4)) for _ in range(nfiles)],
            "quiet": [rng.normal(loc=shift, size=(nq, 4)) for _ in range(nfiles)],
        }

    synth = pack(5, 12, 12, 0.0)
    near = pack(1, 12, 12, 0.0)
    far = pack(1, 12, 12, 2.0)
    kn = fbd.kid_balanced(synth, near, seed=0)
    kf = fbd.kid_balanced(synth, far, seed=0)
    assert kn["n_zone"] == 12 and kn["n_quiet"] == 12 and kn["n"] == 24
    assert abs(kn["zone_fraction"] - 0.5) < 1e-12
    assert kf["zone_fraction"] == kn["zone_fraction"]
    assert kf["kid"] > kn["kid"]
    again = fbd.kid_balanced(synth, far, seed=0)
    assert again["kid"] == kf["kid"]


def test_file_probe_separates_one_shifted_cable():
    rng = np.random.default_rng(22)

    def synth(shift):
        feats, groups = [], []
        for g in range(6):
            feats.append(rng.normal(loc=shift, size=(18, 5)))
            groups.extend([f"s{g}"] * 18)
        return np.concatenate(feats), groups

    a, ga = synth(0.0)
    near = rng.normal(size=(72, 5))
    far = rng.normal(loc=2.5, size=(72, 5))
    pn = fbd.file_domain_probe(a, ga, near, n_chunks=4, n_splits=4, seed=0)
    pf = fbd.file_domain_probe(a, ga, far, n_chunks=4, n_splits=4, seed=0)
    assert pf["n_folds"] >= 2 and pf["separation"] > 0.9
    assert pn["separation"] < 0.75
    assert pf["separation"] > pn["separation"]
    again = fbd.file_domain_probe(a, ga, far, n_chunks=4, n_splits=4, seed=0)
    assert again["auc_oof"] == pf["auc_oof"]


def test_origins_match_crops_and_colored_noise_grows():
    rng = np.random.default_rng(3)
    t_n, w_n = 48, 40
    T = rng.normal(size=(t_n, w_n)).astype(np.float32)
    valid = np.ones((t_n, w_n), bool)
    X = np.zeros((3, t_n, w_n), np.float32)
    X[2] = 1
    zone = np.zeros((t_n, w_n), bool)
    zone[8:12, 8:12] = True
    plain = fbd.sample_patches(X, valid, 4, (16, 16), seed=2, ban=zone)
    paired, origin = fbd.sample_patches(X, valid, 4, (16, 16), seed=2, ban=zone, return_origins=True)
    assert np.array_equal(plain, paired)
    assert np.array_equal(paired, fbd.crop_patches(X, origin, (16, 16)))
    zplain = fbd.sample_zone_patches(X, valid, zone, n=2, patch=(16, 16), seed=2, min_sep=2)
    zpair, zorigin = fbd.sample_zone_patches(X, valid, zone, n=2, patch=(16, 16), seed=2, min_sep=2, return_origins=True)
    assert np.array_equal(zplain, zpair)
    assert np.array_equal(zpair, fbd.crop_patches(X, zorigin, (16, 16)))
    base = fbd.channels_with_colored_noise(T, valid, seed=1, phi_t=0.32, phi_x=0.556, std=0.0)
    mild = fbd.channels_with_colored_noise(T, valid, seed=1, phi_t=0.32, phi_x=0.556, std=0.05)
    strong = fbd.channels_with_colored_noise(T, valid, seed=1, phi_t=0.32, phi_x=0.556, std=0.8)
    d_mild = float(np.mean((mild[:2] - base[:2]) ** 2))
    d_strong = float(np.mean((strong[:2] - base[:2]) ** 2))
    assert d_strong > d_mild > 0


def test_bn_eval_keeps_stats_and_real_refresh_is_refused():
    torch = pytest.importorskip("torch")
    path = ROOT / "models_release_dom" / "unet_dom.pt"
    if not path.is_file():
        pytest.skip("models_release_dom/unet_dom.pt is missing")
    enc = fbd.BottleneckEncoder.load(path)
    snap = fbd.bn_running_snapshot(enc.model)
    assert len(snap) >= 1
    rng = np.random.default_rng(0)
    patches = rng.normal(size=(8, 3, 32, 64)).astype(np.float32)
    enc.embed(patches, batch=4)
    assert fbd.bn_stats_max_abs(snap, fbd.bn_running_snapshot(enc.model)) == 0.0
    weight = enc.model.e1[0].weight.detach().clone()
    with pytest.raises(ValueError, match="real"):
        fbd.refresh_bn_synth(enc.model, patches, batch=4, domain="real")
    assert fbd.bn_stats_max_abs(snap, fbd.bn_running_snapshot(enc.model)) == 0.0
    fbd.refresh_bn_synth(enc.model, patches, batch=4, domain="synth")
    assert fbd.bn_stats_max_abs(snap, fbd.bn_running_snapshot(enc.model)) > 0
    assert torch.equal(weight, enc.model.e1[0].weight.detach())
    assert enc.model.training is False


def test_zone_patches_cover_a_zone_cell():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(3, 60, 70)).astype(np.float32)
    valid = np.ones((60, 70), bool)
    zone = np.zeros((60, 70), bool)
    zone[10:14, 20:24] = True
    X[:, 10:14, 20:24] = 5
    a = fbd.sample_zone_patches(X, valid, zone, n=3, patch=(16, 16), seed=4, min_sep=4)
    b = fbd.sample_zone_patches(X, valid, zone, n=3, patch=(16, 16), seed=4, min_sep=4)
    assert len(a) >= 1
    assert np.array_equal(a, b)
    assert np.all(np.any(a.reshape(len(a), -1) == 5, axis=1))
