"""Потери, мульти-кэш и AR(1)-аугментация. Дефолт train_stage1 не трогаем."""
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dts import losses, multicache, noise_aug


def test_parse_named_terms():
    assert losses.parse_spec("focal+tversky_ce") == ["focal", "tversky_ce"]
    assert losses.parse_spec("ce") == ["ce"]
    try:
        losses.parse_spec("dice")
        raise AssertionError("dice не из списка")
    except ValueError:
        pass


def test_focal_ignores_invalid_and_penalises_confident_error():
    cw = torch.tensor([1.0, 3.0, 3.0])
    logit = torch.tensor([[[[5.0, -5.0]]]], dtype=torch.float32).repeat(1, 3, 1, 1)
    # класс 0 уверенно на обоих пикселях; второй пиксель невалиден
    y = torch.tensor([[[0, -1]]])
    good = losses.focal(logit, y, cw)
    y_bad = torch.tensor([[[1, -1]]])
    bad = losses.focal(logit, y_bad, cw)
    assert good.ndim == 0 and float(good) < float(bad)
    # градиент доходит до логитов
    logit2 = logit.detach().clone().requires_grad_(True)
    losses.focal(logit2, y_bad, cw).backward()
    assert logit2.grad.abs().sum() > 0


def test_tversky_ce_lower_when_foreground_matches():
    cw = torch.tensor([1.0, 3.0, 3.0])
    y = torch.zeros(1, 4, 4, dtype=torch.long)
    y[0, 1:3, 1:3] = 1
    y[0, 0, 0] = -1
    hit = torch.full((1, 3, 4, 4), -4.0)
    hit[:, 1] = 4.0
    miss = torch.full((1, 3, 4, 4), 4.0)
    miss[:, 1:] = -4.0
    assert float(losses.compute("tversky_ce", hit, y, cw)) < float(losses.compute("tversky_ce", miss, y, cw))


def test_two_versions_mmap(tmp_path):
    for ver, val in (("va", 1.0), ("vb", 3.0)):
        for split in ("train", "val"):
            d = tmp_path / "cache" / ver / split
            d.mkdir(parents=True)
            x = np.full((3, 6, 7), val, np.float16)
            ev = np.zeros((6, 7), np.uint8)
            ev[1, 1] = 2
            np.savez(d / "seq0007.npz", X=x, ev=ev, valid=np.ones((6, 7), bool))
    loaded = multicache.load_splits(["va", "vb"], artifact_root=tmp_path)
    assert len(loaded["train"]) == 2 and len(loaded["val"]) == 2
    vals = sorted(float(np.asarray(it["X"])[0, 0, 0]) for it in loaded["train"])
    assert vals == [1.0, 3.0]
    # повторный вызов не переписывает npy и читает тот же y
    again = multicache.load_splits(["va"], splits=("train",), artifact_root=tmp_path)
    assert int(np.asarray(again["train"][0]["y"])[1, 1]) == 1


def test_ar1_leaves_valid_and_hits_std():
    rng = np.random.default_rng(0)
    x = np.zeros((6, 3, 48, 36), np.float32)
    x[:, 2] = 1
    y = noise_aug.add_channel_ar1(x, 0.4, 0.7, 0.3, rng)
    assert np.allclose(y[:, 2], 1)
    assert abs(float(y[:, 0].std()) - 0.4) < 0.05
    assert abs(float(y[:, 1].std()) - 0.4) < 0.05
    assert not np.allclose(y[:, 0], y[:, 1])
    assert float(np.max(np.abs(y[:, :2]))) <= 12
    same = noise_aug.add_channel_ar1(x, 0.0, 0.9, 0.9, rng)
    assert np.shares_memory(same, x) or np.allclose(same, x)
    assert np.allclose(same, x)


def test_recipe_parser_reads_lit_sentence():
    text = (
        "Рецепт noise-matched аугментации для агента ae, сетка 120 с × 0.25 м:\n"
        "До make_channels: белый шум генератора умножить на 0.953. "
        "Поверх этого окрасить поле AR(1): φ_t = 0.357 на шаге 120 с и φ_x = 0.736 "
        "на шаге 0.25 м (у synth сейчас φ_t = 0.170, φ_x = 0.345).\n"
        "На уже нормированном тайле ae, каналы 0 и 1: добавить такой же AR(1) со std = 0.743 "
        "в единицах σ. Канал valid не трогать.\n"
    )
    got = noise_aug.recipe_from_markdown(text)
    assert got == {"std": 0.743, "phi_t": 0.357, "phi_x": 0.736}
    assert noise_aug.recipe_from_markdown("нет рецепта") is None
    live = Path("/workspace/dts_solution/docs/EXPERIMENTS_LATE.md").read_text()
    assert noise_aug.recipe_from_markdown(live) == {"std": 0.743, "phi_t": 0.357, "phi_x": 0.736}


def test_train_stage1_help_keeps_defaults():
    root = Path(__file__).resolve().parents[1]
    src = (root / "scripts" / "train_stage1.py").read_text()
    assert "return ce + dice" in src
    r = subprocess.run(
        [sys.executable, "scripts/train_stage1.py", "--help"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert "--loss" in r.stdout and "--sets" in r.stdout
    assert 'ap.add_argument("--loss", default=None' in src
    assert 'ap.add_argument("--sets", default=None' in src
    assert 'ap.add_argument("--noise-std", type=float, default=0.0' in src
