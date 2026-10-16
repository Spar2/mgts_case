"""LIT5: single-source DG recipes, no target data, no training."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "S3_LIT_DG.md"
JS = ROOT / "artifacts" / "eval" / "s3_lit5.json"

ARXIV = ("2304.00424", "2307.09005", "2407.03654", "2506.10133")


def test_papers_advice_and_ban():
    doc = DOC.read_text(encoding="utf-8")
    payload = json.loads(JS.read_text(encoding="utf-8"))
    assert payload["name"] == "LIT5"
    assert payload["model_trained"] is False
    assert payload["beats_winner"] is False
    assert payload["real_csv_role"] == "forbidden_even_unlabeled"
    assert payload["n_papers_primary"] == 4
    assert payload["winner"]["val_iou_far_1e-3"] == 0.38671235821914546
    assert payload["winner"]["test_iou_far_1e-3"] == 0.4052762065230473
    assert payload["dom_caution"]["gap_decreased_vs_v810ft"] is False
    assert {a["id"] for a in payload["advice"]} == {"M", "R", "P"}
    for paper in payload["papers"]:
        assert 2023 <= paper["year"] <= 2026
        assert paper["arxiv"] in doc
    for arxiv in ARXIV:
        assert arxiv in doc
    for advice in payload["advice"]:
        assert advice["real_train"] is False
        assert advice["id"] in doc
        assert "0.38671235821914546" in advice["adopt_if"]
    assert "даже без меток" in doc
    assert "0.38671235821914546" in doc
    assert "real_noise_psd" in doc
    assert "featurize_noisy" in doc
    p = next(a for a in payload["advice"] if a["id"] == "P")
    assert p["numbers"]["phi_t_synth"] == 0.170
    assert p["numbers"]["phi_x_synth"] == 0.345
    assert p["numbers"]["forbidden_phi_t"] == 0.357
    assert p["numbers"]["forbidden_phi_x"] == 0.736
    assert p["numbers"]["forbidden_sigma_c"] == 0.283
    assert p["numbers"]["load_real_noise_psd"] is False
    m = next(a for a in payload["advice"] if a["id"] == "M")
    assert m["numbers"]["attach"] == "e1"
    assert m["numbers"]["p"] == 0.5
    assert m["numbers"]["beta"] == 0.6
    assert m["numbers"]["off_at_eval"] is True
    r = next(a for a in payload["advice"] if a["id"] == "R")
    assert r["numbers"]["k"] == 3
    assert r["numbers"]["L_ours"] == [1, 2]


def test_module_does_not_import_torch():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module.split(".")[0])
    assert "torch" not in mods
