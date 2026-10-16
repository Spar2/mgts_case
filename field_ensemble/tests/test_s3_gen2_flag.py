"""Флаг gen2 в dts.infer: по умолчанию выключен, вместе с mae/static не включается."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from dts import infer


THR = 0.8579896092424856


def test_gen2_flag_defaults_off():
    assert infer._GEN2 == ("off", "on")
    assert infer._MAE == ("off", "on")
    assert inspect.signature(infer.Analyzer.__init__).parameters["gen2"].default == "off"
    assert inspect.signature(infer.Analyzer.run).parameters["gen2"].default is None


def test_blend_with_gen2_is_equal_mean():
    score = np.array([0.2, 0.8], np.float32)
    hot = np.array([0.5, -0.5], np.float32)
    s_g = np.array([1.0, 0.0], np.float32)
    h_g = np.array([1.0, -1.0], np.float32)
    got_s, got_h = infer.blend_with_gen2(score, hot, s_g, h_g)
    assert np.allclose(got_s, 0.5 * (score + s_g))
    assert np.allclose(got_h, 0.5 * (hot + h_g))


def test_gen2_rejects_bad_and_mixed_flags():
    with pytest.raises(ValueError, match="gen2 must be"):
        infer.Analyzer(gen2="branch")
    with pytest.raises(ValueError, match="gen2=on"):
        infer.Analyzer(gen2="on", mae="on")
    with pytest.raises(ValueError, match="gen2=on"):
        infer.Analyzer(gen2="on", static="on")
    with pytest.raises(ValueError, match="mae must be"):
        infer.Analyzer(mae="branch")


def test_gen2_spec_threshold_and_ckpt():
    spec = infer.gen2_spec()
    assert spec["method"] == "ens_blend"
    assert spec["threshold"] == THR
    assert spec["width"] == 16
    ckpt = Path(spec["ckpt"])
    assert ckpt.name == "s3_gen2_unet.pt"
    assert ckpt.is_file()
    assert ckpt.parent.name == "models_release_gen2"
