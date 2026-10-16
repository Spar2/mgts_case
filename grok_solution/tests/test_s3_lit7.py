"""LIT7: three synth-only pipelines, literature only, no training."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "S3_LIT7.md"
JS = ROOT / "artifacts" / "eval" / "s3_lit7.json"
R7 = ROOT / "artifacts" / "eval" / "r7_ens_u2ft_tcn_lgbm.json"
V10 = ROOT / "artifacts" / "eval" / "v10_ens_u2ft_tcn_lgbm.json"
V11 = ROOT / "artifacts" / "eval" / "v11_ens_u2ft_tcn_lgbm.json"

ARXIV = (
    "2306.12041",
    "2307.07534",
    "2408.03143",
    "2503.02424",
    "2311.12079",
    "2403.01427",
    "2303.08730",
)


def test_papers_advice_and_ban():
    doc = DOC.read_text(encoding="utf-8")
    payload = json.loads(JS.read_text(encoding="utf-8"))
    r7 = json.loads(R7.read_text(encoding="utf-8"))
    v10 = json.loads(V10.read_text(encoding="utf-8"))
    v11 = json.loads(V11.read_text(encoding="utf-8"))
    published = r7["val"]["ens_u2ft_tcn_lgbm@far_1e-3"]["iou"]
    assert published == 0.38671235821914546
    assert r7["test"]["ens_u2ft_tcn_lgbm@far_1e-3"]["iou"] == 0.4052762065230473
    assert v10["test"]["ens_u2ft_tcn_lgbm@far_1e-3"]["iou"] == 0.4038977883892501
    assert v11["test"]["ens_u2ft_tcn_lgbm@far_1e-3"]["iou"] == 0.40390028295846503

    assert payload["name"] == "LIT7"
    assert payload["model_trained"] is False
    assert payload["beats_winner"] is False
    assert payload["real_csv_role"] == "forbidden_even_unlabeled"
    assert payload["n_papers_primary"] == 6
    assert payload["winner"]["val_iou_far_1e-3"] == published
    assert payload["winner"]["v10_test_iou"] == 0.4038977883892501
    assert payload["winner"]["v11_test_iou"] == 0.40390028295846503
    assert payload["nested_bar"] == 0.3897
    assert payload["flow_rejected"]["nested_delta"] == -0.0113
    assert {a["id"] for a in payload["advice"]} == {"M", "P", "K"}
    assert {e["id"] for e in payload["excluded"]} == {"diffusionad", "crf"}
    for paper in payload["papers"]:
        assert 2023 <= paper["year"] <= 2026
        assert paper["arxiv"] in doc
        assert paper["doi"] in doc
    for arxiv in ARXIV:
        assert arxiv in doc
    for advice in payload["advice"]:
        assert advice["real_train"] is False
        assert advice["id"] in doc
        assert "0.38671235821914546" in advice["adopt_if"]
        assert "0.3897" in advice["adopt_if"]
        assert "0.4038977883892501" in advice["adopt_if"]
        assert "0.40390028295846503" in advice["adopt_if"]
        assert advice["cpu"]
    assert "даже без меток" in doc
    assert "BEATS_WINNER" in doc
    assert "real_noise_psd" in doc
    assert "DINO2-R" in doc
    assert "wide_resnet50_2" in doc

    mae = next(a for a in payload["advice"] if a["id"] == "M")
    assert mae["numbers"]["mask_ratio"] == 0.75
    assert mae["numbers"]["brats_mse_auroc"] == 0.614
    assert mae["numbers"]["brats_l1_auroc"] == 0.834
    assert mae["numbers"]["load_vae"] is False
    assert mae["numbers"]["score"] == "abs_residual_channel0_plus_sign"

    mem = next(a for a in payload["advice"] if a["id"] == "P")
    assert mem["numbers"]["knn_only"] is False
    assert mem["numbers"]["load_dino"] is False
    assert mem["numbers"]["coreset_max"] == 8192
    assert mem["numbers"]["inp_prototypes"] == 6
    assert mem["numbers"]["ssn_mvtec_auroc"] == 98.4

    distill = next(a for a in payload["advice"] if a["id"] == "K")
    assert distill["numbers"]["finetune_teacher"] is False
    assert distill["numbers"]["spatial_z_score"] is False
    assert distill["numbers"]["freekd_miou"] == 74.40
    assert distill["numbers"]["freekd_teacher_miou"] == 78.34
    assert distill["numbers"]["hard_ce_weight"] == 0.1


def test_module_does_not_import_torch():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module.split(".")[0])
    assert "torch" not in mods
