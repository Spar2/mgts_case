"""Сторож DG: находит .pt, пропускает уже посчитанный sha, не трогает чужие ключи JSON."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    path = ROOT / "scripts" / "bench_fbd_dg.py"
    spec = importlib.util.spec_from_file_location("bench_fbd_dg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discover_finds_pt_and_ignores_json(tmp_path):
    mod = _mod()
    (tmp_path / "s3_dg_unet.pt").write_bytes(b"dg")
    (tmp_path / "s3_mixstyle_note.json").write_text("{}", encoding="utf-8")
    (tmp_path / "s3_other.pt").write_bytes(b"no")
    nested = tmp_path / "s3_randconv_run"
    nested.mkdir()
    (nested / "best.pt").write_bytes(b"rc")
    (nested / "log.json").write_text("{}", encoding="utf-8")
    (tmp_path / "s3_static_readme.txt").write_text("x", encoding="utf-8")
    found = mod.discover(tmp_path)
    families = sorted(item["family"] for item in found)
    assert families == ["dg", "randconv"]
    assert all(item["path"].suffix == ".pt" for item in found)


def test_pending_skips_known_sha(tmp_path):
    mod = _mod()
    path = tmp_path / "s3_static_a.pt"
    path.write_bytes(b"weights")
    found = mod.discover(tmp_path)
    digest = mod.sha256_file(path)
    assert mod.pending_items(found, {digest}) == []
    pending = mod.pending_items(found, set())
    assert len(pending) == 1
    assert pending[0]["sha256"] == digest
    assert pending[0]["family"] == "static"


def test_unwrap_state_dict_module_prefix_and_reject():
    mod = _mod()

    class Weight:
        def __init__(self, shape):
            self.shape = shape

    raw = {"e1.0.weight": Weight((16,))}
    assert mod.unwrap_unet_state(raw)["e1.0.weight"].shape[0] == 16
    wrapped = {"model": {"module.e1.0.weight": Weight((8,)), "module.head.weight": Weight((1,))}}
    state = mod.unwrap_unet_state(wrapped)
    assert "e1.0.weight" in state
    assert "head.weight" in state
    assert mod.unwrap_unet_state({"state_dict": {"foo": 1}}) is None


def test_load_encoder_strict_rejects_extra_key(tmp_path):
    import sys
    import torch
    sys.path.insert(0, str(ROOT))
    from dts.models import DTSUNet

    mod = _mod()
    model = DTSUNet(w=4)
    good = tmp_path / "s3_dg_tiny.pt"
    torch.save(model.state_dict(), good)
    enc, status, _detail = mod.load_encoder(good)
    assert status == "ok"
    assert enc.load_info == {"strict": True, "n_loaded": len(model.state_dict()), "n_model": len(model.state_dict()), "missing": 0, "unexpected": 0}
    bad_state = model.state_dict()
    bad_state["not_a_unet.weight"] = bad_state["e1.0.weight"].detach().clone()
    bad = tmp_path / "s3_mixstyle_extra.pt"
    torch.save(bad_state, bad)
    enc2, status2, detail2 = mod.load_encoder(bad)
    assert enc2 is None
    assert status2 == "key_mismatch"
    assert "unexpected=1" in detail2


def test_partial_error_is_retry_not_a_recorded_skip():
    mod = _mod()
    assert mod.is_partial_error(EOFError("ran out of input"))
    assert mod.is_partial_error(RuntimeError("PytorchStreamReader failed reading zip archive"))
    assert not mod.is_partial_error(RuntimeError("Weights only load failed"))


def test_splice_keeps_fbd5_and_replaces_only_fbd6():
    mod = _mod()
    doc = "before\n<!-- FBD5_END -->\n\nafter results stay\n"
    once = mod.splice_fbd6(doc, "## DG-модели\n\nпусто\n")
    assert "before" in once
    assert "<!-- FBD5_END -->" in once
    assert "after results stay" in once
    assert once.count("<!-- FBD6_START -->") == 1
    twice = mod.splice_fbd6(once, "## DG-модели\n\nесть модель\n")
    assert twice.count("<!-- FBD6_START -->") == 1
    assert "есть модель" in twice
    assert "пусто" not in twice
    assert "after results stay" in twice
    assert "<!-- FBD5_END -->" in twice


def test_merge_keeps_foreign_keys_and_accumulates_by_sha():
    mod = _mod()
    payload = {"pairs": {"v8": 1}, "bn_noise": {"real_calibrated": False}, "order": ["kid"]}
    mod.apply_update(payload, {"models": [{"sha256": "a", "family": "dg", "checkpoint": "artifacts/eval/s3_dg.pt"}]})
    mod.apply_update(payload, {"models": [{"sha256": "b", "family": "mixstyle", "checkpoint": "artifacts/eval/s3_mixstyle.pt"}]})
    assert payload["pairs"] == {"v8": 1}
    assert payload["bn_noise"]["real_calibrated"] is False
    assert payload["order"] == ["kid"]
    assert [row["sha256"] for row in payload["dg_models"]["models"]] == ["a", "b"]
    assert payload["dg_models"]["selection"] is False
    text = mod.render_section(payload["dg_models"], [])
    assert "DG-модели" in text
    assert "Диагностика, не выбор" in text
    assert "Победитель" not in text
    payload["dg_models"]["models"][0]["same_state_dict_as"] = "artifacts/eval/s3_dg_unet.pt"
    noted = mod.render_section(payload["dg_models"], [])
    assert "max |Δ| = 0" in noted
    assert "s3_dg_unet.pt" in noted
