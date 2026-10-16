import numpy as np
import pytest

from fiber.classify.physical import FEATURE_NAMES, selection_under_budget, shape_features
from fiber.classify.physical import load_physical_model, classify_physical_shape
from pathlib import Path
import json


def _research_payload():
    from fiber.classify.physical import SCHEMA
    from fiber.classify.rules import LABELS
    return {'schema':SCHEMA,'features':list(FEATURE_NAMES),'labels':list(LABELS),
            'mu':[0.]*len(FEATURE_NAMES),'sig':[1.]*len(FEATURE_NAMES),
            'weight':np.zeros((len(FEATURE_NAMES),len(LABELS))).tolist(),
            'bias':[0.]*len(LABELS),'threshold':.5,'serve':'research_offline','expert_confirmed':False}


def test_research_model_keeps_provenance_and_feature_values(tmp_path):
    path = tmp_path/'model.json'
    path.write_text(json.dumps(_research_payload()))
    model = load_physical_model(path)
    result = classify_physical_shape(model,*_pulse_case())
    assert result['expert_confirmed'] is False
    assert result['serve'] == 'research_offline'
    assert result['features']['positive_episodes'] == 6
    assert result['score_semantics'] == 'uncalibrated_softmax'


def test_old_feature_schema_cannot_be_loaded_as_new_weights(tmp_path):
    blob = _research_payload()
    blob['features'] = ['support','peak','n_half','spike','runs']
    target = tmp_path/'wrong_schema.json'
    target.write_text(json.dumps(blob))
    with pytest.raises(ValueError,match='schema'):
        load_physical_model(target)


def _pulse_case(dt=60., dx=1., periodic=True):
    times = np.arange(0, 3600, dt)
    x = np.arange(0, 80, dx)
    values = np.full((len(times), len(x)), 22.)
    core = (x >= 38) & (x < 44)
    if periodic:
        rows = (times >= 120) & (((times - 120) % 600) < 120)
    else:
        rows = (times >= 120) & (times < 840)
    values[np.ix_(rows, core)] += 4
    return values, np.ones_like(values, bool), x, times


def test_repeated_pulses_and_one_block_are_distinguishable():
    pulse = dict(zip(FEATURE_NAMES, shape_features(*_pulse_case()), strict=True))
    block = dict(zip(FEATURE_NAMES, shape_features(*_pulse_case(periodic=False)), strict=True))
    assert pulse['positive_episodes'] == 6
    assert block['positive_episodes'] == 1
    assert pulse['period_s'] == 600
    assert block['period_s'] == 0
    assert pulse['positive_duty'] == pytest.approx(block['positive_duty'])


@pytest.mark.parametrize('dt,dx', [(10,.1), (30,.25), (120,1.)])
def test_sampling_changes_do_not_change_physical_pulse_features(dt,dx):
    base = shape_features(*_pulse_case())
    sampled = shape_features(*_pulse_case(dt,dx))
    np.testing.assert_allclose(sampled, base, atol=1e-8)


def test_high_recall_cannot_override_the_false_alarm_budget():
    result = selection_under_budget({'event_recall':1.,'false_alarm_rate':1.}, {'event_recall':.99,'false_alarm_rate':0.}, max_false_alarm_rate=.05)
    assert result['selected'] == 'baseline'
    assert result['rejected']['challenger'] == 'false_alarm_budget_exceeded'


def test_no_eligible_model_means_no_selection():
    result = selection_under_budget({'event_recall':1.,'false_alarm_rate':.2}, None, max_false_alarm_rate=.05)
    assert result['selected'] is None


def test_missing_history_does_not_establish_period():
    field, valid, x, times = _pulse_case()
    valid[20] = False
    result = dict(zip(FEATURE_NAMES, shape_features(field, valid, x, times), strict=True))
    assert result['period_s'] == 0
