"""LIT3: заметка и JSON согласованы с локальными артефактами, без обучения."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "S3_LIT.md"
JS = ROOT / "artifacts" / "eval" / "s3_lit3.json"
R7 = ROOT / "artifacts" / "eval" / "r7_ens_u2ft_tcn_lgbm.json"
PSD = ROOT / "artifacts" / "real_noise_psd.npz"

PRIMARY = ("10.3390/fire8010023", "10.1364/JLT.44.007744", "10.1117/12.3055611")
AGENTS = ("WEAK", "AXIAL", "DOM", "TTA")


def _load():
    doc = DOC.read_text(encoding="utf-8")
    payload = json.loads(JS.read_text(encoding="utf-8"))
    return doc, payload


def test_primary_papers_in_doc_and_json():
    doc, payload = _load()
    papers = [p for p in payload["papers"] if p["primary"]]
    assert len(papers) == 3
    assert payload["n_papers_primary"] == 3
    years = sorted(p["year"] for p in papers)
    assert years[0] >= 2024 and years[-1] <= 2026
    dois = [p["doi"] for p in papers]
    assert dois == list(PRIMARY)
    for doi in dois:
        assert doi in doc
    assert "только оценка" in doc


def test_ideas_cover_agents_and_stay_off_real_labels():
    doc, payload = _load()
    ideas = payload["ideas"]
    assert payload["real_csv_role"] == "evaluation_only"
    assert payload["beats_winner"] is False
    assert payload["model_trained"] is False
    primary = [i for i in ideas if i["priority"] == "primary"]
    assert payload["n_ideas_primary"] == len(primary) == 4
    assert {i["agent"] for i in primary} == set(AGENTS)
    known = {p["id"] for p in payload["papers"]} | {p["id"] for p in payload["papers_supporting"]}
    for idea in ideas:
        assert idea["real_train"] is False
        assert idea["minutes_max"] <= 60
        assert idea["id"] in doc
        for pid in idea["paper_ids"]:
            assert pid in known


def test_numbers_match_winner_and_noise_psd():
    _, payload = _load()
    r7 = json.loads(R7.read_text(encoding="utf-8"))
    w = payload["winner"]
    assert w["val_iou_far_1e-3"] == r7["val"]["ens_u2ft_tcn_lgbm@far_1e-3"]["iou"]
    assert w["test_iou_far_1e-3"] == r7["test"]["ens_u2ft_tcn_lgbm@far_1e-3"]["iou"]
    weak = r7["test"]["ens_u2ft_tcn_lgbm@far_1e-3"]["by_amp"]["<=2C"]
    assert w["test_weak_le2c_event_recall"] == weak["recall"]
    assert w["test_weak_le2c_n"] == weak["n"]
    z = np.load(PSD)
    noise = payload["noise"]
    assert noise["pooled_phi_t"] == float(z["pooled_phi_t"])
    assert noise["pooled_phi_x"] == float(z["pooled_phi_x"])
    assert noise["pooled_phi_t_allvalid"] == float(z["pooled_phi_t_allvalid"])
    assert noise["pooled_phi_x_allvalid"] == float(z["pooled_phi_x_allvalid"])
    assert abs(noise["resid_std_file1"] - float(z["resid_std"][0])) < 1e-6
    d1 = next(i for i in payload["ideas"] if i["id"] == "D1")
    assert d1["phi_t"] == noise["pooled_phi_t"]
    assert d1["phi_x"] == noise["pooled_phi_x"]
    assert d1["sigma_c_deg"] == noise["sigma_c_deg"]
