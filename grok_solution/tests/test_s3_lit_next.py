"""LIT6: three non-U-Net pipelines, synth only, no training."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "S3_LIT_NEXT.md"
JS = ROOT / "artifacts" / "eval" / "s3_lit6.json"

ARXIV = ("2508.02889", "2405.14458", "2304.08069", "2501.17688", "2504.05662")


def test_papers_advice_and_ban():
    doc = DOC.read_text(encoding="utf-8")
    payload = json.loads(JS.read_text(encoding="utf-8"))
    assert payload["name"] == "LIT6"
    assert payload["model_trained"] is False
    assert payload["beats_winner"] is False
    assert payload["real_csv_role"] == "forbidden_even_unlabeled"
    assert payload["n_papers_primary"] == 4
    assert payload["winner"]["val_iou_far_1e-3"] == 0.38671235821914546
    assert payload["nested_bar"] == 0.3897
    assert {a["id"] for a in payload["advice"]} == {"F", "D", "C"}
    for paper in payload["papers"]:
        assert 2023 <= paper["year"] <= 2026
        assert paper["arxiv"] in doc
    for arxiv in ARXIV:
        assert arxiv in doc
    for advice in payload["advice"]:
        assert advice["real_train"] is False
        assert advice["id"] in doc
        assert "0.38671235821914546" in advice["adopt_if"]
        assert advice["cpu"]
    assert "даже без меток" in doc
    assert "0.38671235821914546" in doc
    assert "real_noise_psd" in doc
    f = next(a for a in payload["advice"] if a["id"] == "F")
    assert f["numbers"]["steps"] == 1
    assert f["numbers"]["load_reflect_vae"] is False
    d = next(a for a in payload["advice"] if a["id"] == "D")
    assert d["numbers"]["nms"] is False
    assert d["numbers"]["coco_weights"] is False
    assert d["numbers"]["rtdetr_r50_ap_s"] == 34.8
    c = next(a for a in payload["advice"] if a["id"] == "C")
    assert c["numbers"]["vertices_max"] == 8
    assert c["numbers"]["refine_steps"] == 1


def test_module_does_not_import_torch():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module.split(".")[0])
    assert "torch" not in mods
