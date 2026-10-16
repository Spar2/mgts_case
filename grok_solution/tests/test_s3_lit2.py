"""LIT4: советы в заметке согласованы с уже написанным кодом, без обучения."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "S3_LIT2.md"
JS = ROOT / "artifacts" / "eval" / "s3_lit4.json"
FBD = ROOT / "dts" / "fbd.py"
SEMI = ROOT / "dts" / "semi_contrastive.py"
AXIAL = ROOT / "dts" / "axial.py"

ARXIV = ("2505.21160", "2502.17160", "2505.08082", "2506.09785", "2609.08788")


def test_papers_and_advice_present():
    doc = DOC.read_text(encoding="utf-8")
    payload = json.loads(JS.read_text(encoding="utf-8"))
    assert payload["model_trained"] is False
    assert payload["beats_winner"] is False
    assert payload["real_csv_role"] == "evaluation_only"
    assert payload["n_papers_primary"] == 6
    assert {a["id"] for a in payload["advice"]} == {"A", "E", "X"}
    for paper in payload["papers"]:
        assert 2024 <= paper["year"] <= 2026
        if paper["arxiv"]:
            assert paper["arxiv"] in doc
    for arxiv in ARXIV:
        assert arxiv in doc
    for advice in payload["advice"]:
        assert advice["real_train"] is False
        assert advice["id"] in doc
    assert "только оценка" in doc
    assert payload["advice"][0]["numbers"]["rho_min"] == 0.6


def test_advice_matches_current_sources():
    payload = json.loads(JS.read_text(encoding="utf-8"))
    fbd = FBD.read_text(encoding="utf-8")
    semi = SEMI.read_text(encoding="utf-8")
    axial = AXIAL.read_text(encoding="utf-8")
    code = payload["current_code"]
    assert "_EPS_SCHEDULE = (1e-6, 1e-4, 1e-2, 1e-1)" in fbd
    assert code["fbd_eps"] == [1e-6, 1e-4, 1e-2, 1e-1]
    assert f"PSEUDO_THRESHOLD = {code['pseudo_threshold']}" in semi
    assert "temperature: float = 0.2" in semi
    assert code["ntxent_temperature"] == 0.2
    assert code["pseudo_threshold"] == 0.9
    assert "heads: int = 4" in axial
    assert "max_len: int = 96" in axial
    assert code["axial_heads"] == 4
    assert code["axial_max_len"] == 96
    e = next(a for a in payload["advice"] if a["id"] == "E")
    assert e["numbers"]["temperature"] == code["ntxent_temperature"]
    assert e["numbers"]["pseudo_threshold"] == code["pseudo_threshold"]
    x = next(a for a in payload["advice"] if a["id"] == "X")
    assert x["numbers"]["heads"] == code["axial_heads"]
    assert x["numbers"]["attach"] == "e3"
