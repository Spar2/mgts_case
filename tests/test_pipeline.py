import numpy as np
import pytest

from fiber.pipeline import PipelineConfig, analyze_field


def test_operational_status_ignores_temporal_innovation():
    import inspect

    from fiber.anomaly.detect import temporal_innovation
    from fiber.denoise.dsp import median_x

    source = inspect.getsource(analyze_field)
    assert "temporal_innovation" not in source
    assert "temporal_z" not in source
    x = np.arange(120, dtype=np.float64)
    times = np.arange(16, dtype=np.float64) * 60.0
    step = np.full((16, 120), 20.0, dtype=np.float32)
    step[8:, :] = 28.0
    valid = np.ones(step.shape, dtype=bool)
    quiet = analyze_field(step, x, times)
    assert quiet["anomaly_status"] == "Normal"
    assert quiet["zones"] == []
    manifest = quiet["manifest"]
    assert manifest["clean"] == "median_x3"
    assert manifest["detector"] == "spatial_z_halo"
    assert manifest["config"]["threshold_z"] == 3.5
    assert manifest["config"]["inner_m"] == 8.0
    assert manifest["config"]["outer_m"] == 40.0
    assert manifest["config"]["bin_m"] == 2.0
    assert manifest["config"]["floor_c"] == 0.30
    assert manifest["config"]["min_width_m"] == 3.0
    assert manifest["config"]["min_profiles"] == 6
    assert manifest["config"]["min_cells"] == 24
    assert manifest["config"]["max_gap_s"] == 300.0
    clean = median_x(step, valid)
    historical = temporal_innovation(clean, valid, times)
    assert np.median(np.abs(historical[10])) > 3.5
    hot = np.full((12, 100), 22.0, dtype=np.float32)
    hot[:, 45:51] += 6.0
    found = analyze_field(hot, np.arange(100), np.arange(12) * 60.0)
    assert found["anomaly_status"] == "Anomaly"
    assert found["manifest"]["detector"] == "spatial_z_halo"


def test_zone_fields_keep_spatial_status():
    import math

    from fiber.anomaly.detect import temporal_innovation
    from fiber.anomaly.evidence import annotate_zone_fields
    from fiber.denoise.dsp import median_x

    x = np.arange(120, dtype=np.float64)
    times = np.arange(16, dtype=np.float64) * 60.0
    step = np.full((16, 120), 20.0, dtype=np.float32)
    step[8:, :] = 28.0
    valid = np.ones(step.shape, dtype=bool)
    quiet = analyze_field(step, x, times)
    assert quiet["anomaly_status"] == "Normal"
    assert quiet["zones"] == []
    manifest = quiet["manifest"]
    assert manifest["clean"] == "median_x3"
    assert manifest["detector"] == "spatial_z_halo"
    assert manifest["config"]["threshold_z"] == 3.5
    assert manifest["trained_classifier_operational"] is False
    clean = median_x(step, valid)
    historical = temporal_innovation(clean, valid, times)
    assert not np.all(historical == 0)

    hot = np.full((12, 100), 22.0, dtype=np.float32)
    hot[:, 45:51] += 6.0
    hx = np.arange(100, dtype=np.float64)
    ht = np.arange(12, dtype=np.float64) * 60.0
    found = analyze_field(hot, hx, ht)
    assert found["anomaly_status"] == "Anomaly"
    assert found["manifest"]["clean"] == "median_x3"
    assert found["manifest"]["detector"] == "spatial_z_halo"
    assert found["manifest"]["config"]["threshold_z"] == 3.5
    assert found["manifest"]["trained_classifier_operational"] is False
    assert found["zones"]
    alarm = (found["z"] >= 3.5) & found["score_available"]
    assert int(alarm.sum()) == sum(zone["support"] for zone in found["zones"])
    for zone in found["zones"]:
        assert zone["expert_confirmed"] is False
        assert math.isfinite(zone["contrast"]) and zone["contrast"] > 0
        assert math.isfinite(zone["width_m"]) and zone["width_m"] > 0
        assert math.isfinite(zone["observed_duration_s"]) and zone["observed_duration_s"] > 0
        assert zone["support"] == zone["n_cells"] and zone["support"] > 0
        assert zone["periodicity_CV"] is None
        assert zone["evidence"]["spatial"]["view"] == "spatial"
        assert zone["evidence"]["temporal"]["view"] == "temporal"
        cols = np.flatnonzero((hx >= zone["start_m"] - 5e-4) & (hx <= zone["end_m"] + 5e-4))
        rows = np.flatnonzero((ht >= zone["t_start_s"] - 1e-3) & (ht <= zone["t_end_s"] + 1e-3))
        box = alarm[np.ix_(rows, cols)]
        assert zone["support"] == int(box.sum())

    bare = {
        "polarity": "hot",
        "start_m": 4.0,
        "end_m": 6.0,
        "width_m": 3.0,
        "t_start_s": 0.0,
        "t_end_s": 60.0,
        "median_contrast_c": 2.5,
        "n_cells": 8,
        "observed_duration_s": 120.0,
    }
    z = np.zeros((4, 10), dtype=np.float32)
    z[:, 4:7] = 5.0
    kept = annotate_zone_fields(
        [bare],
        temporal=np.zeros_like(z),
        temporal_ok=np.zeros(z.shape, dtype=bool),
        z=z,
        spatial_ok=np.ones(z.shape, dtype=bool),
        x_m=np.arange(10, dtype=np.float64),
        time_s=np.arange(4, dtype=np.float64) * 60.0,
        thresh=3.5,
    )
    assert len(kept) == 1
    assert kept[0]["evidence"]["temporal"]["view"] == "temporal"
    assert kept[0]["evidence"]["temporal"]["available"] is False
    assert kept[0]["evidence"]["temporal"]["score_state"] == "unavailable"
    assert kept[0]["periodicity_CV"] is None


def test_pipeline_reports_hypothesis_and_never_calls_missing_data_normal():
    x=np.arange(100)
    times=np.arange(12)*60.
    field=np.full((12,100),22.)
    field[:,45:51]+=6
    result=analyze_field(field,x,times)
    assert result['anomaly_status']=='Anomaly'
    assert result['zones']
    assert all(z['expert_confirmed'] is False and z['shape_hypothesis']=='Local_Heating' for z in result['zones'])
    missing=analyze_field(np.full_like(field,np.nan),x,times)
    assert missing['anomaly_status']=='Unknown'
    assert missing['score_coverage'] is None


def test_pipeline_respects_explicit_quality_mask():
    field=np.full((12,100),22.)
    field[:,45:51]=28
    valid=np.ones_like(field,bool)
    valid[:,45:51]=False
    result=analyze_field(field,np.arange(100),np.arange(12)*60,valid=valid)
    assert not result['zones']
    assert not result['score_available'][:,45:51].any()


@pytest.mark.parametrize('args',[{'threshold_z':float('nan')},{'max_gap_s':0},{'inner_m':40,'outer_m':8}])
def test_invalid_configs_are_rejected(args):
    with pytest.raises(ValueError): PipelineConfig(**args)


def test_api_rejects_bad_grid_and_exposes_unknown_status():
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from fiber.api import app
    client=TestClient(app)
    response=client.post('/detect/field',json={'raw':[[None]*100]*3,'distances':list(range(100)),'time_s':[0,60,120]})
    assert response.status_code==200
    assert response.json()['anomaly_status']=='Unknown'
    assert response.json()['manifest']['trained_classifier_operational'] is False
    bad=client.post('/detect/field',json={'raw':[[22]*3],'distances':[0,0,2],'time_s':[0]})
    assert bad.status_code==422
    hot=np.full((12,100),22.)
    hot[:,45:51]=28
    found=client.post('/detect/field',json={'raw':hot.tolist(),'distances':list(range(100)),'time_s':(np.arange(12)*60).tolist()})
    assert found.status_code==200
    assert found.json()['anomaly_status']=='Anomaly'
    assert found.json()['zones'][0]['shape_hypothesis']=='Local_Heating'
    assert 'clean' not in found.json()
    assert client.post('/denoise/profile',json={'raw':[22,23],'distances':[1,1]}).status_code==422
    oversized=client.post('/detect/field',json={'raw':[[22,22]],'distances':[0,1e9],'time_s':[0]})
    assert oversized.status_code==413
