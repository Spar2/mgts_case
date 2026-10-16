"""Train and validate physical scene features on disjoint synthetic seeds.

This is OFFLINE SCENE classification, not sliding-window event detection,
not expert cable accuracy, and not a replacement for the operational DSP.
The validation set selects a confidence gate under an explicit background
budget. The test set is evaluated once after selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fiber.classify.physical import FEATURE_NAMES, SCHEMA, shape_features
from fiber.classify.plant import SoftmaxSceneModel
from fiber.classify.rules import LABELS

TRAIN_SEEDS = range(64)
VALIDATION_SEEDS = range(10000,10032)
TEST_SEEDS = range(20000,20064)
THRESHOLDS = (0., .5, .6, .7, .8, .9, .95, .99)


def scene(seed, label, *, stress=False):
    rng = np.random.default_rng(seed*7 + LABELS.index(label))
    geometries = ((30.,.25),(60.,1.),(120.,.25)) if not stress else ((10.,.1),(30.,.1),(120.,1.))
    cadence, spacing = geometries[seed % len(geometries)]
    times = np.arange(0,3600,cadence)
    x = np.arange(0,80,spacing)
    field = np.full((len(times),len(x)),22.)
    slope = rng.uniform(-.012,.012)
    field += slope*x[None,:] + .5*np.sin(times[:,None]/700)
    field += rng.normal(0,rng.uniform(.12,.35),field.shape)
    valid = np.ones(field.shape,bool)
    center = rng.uniform(20,60)
    width = rng.uniform(3,10)
    core = np.abs(x-center) < width/2
    amp = rng.uniform(2,7)
    if label in ('Local_Heating','Local_Cooling'):
        active = (times >= 240) & (times < rng.uniform(1600,2600))
        if seed % 2: active[:] = True
        field[np.ix_(active,core)] += amp * (1 if label == 'Local_Heating' else -1)
    elif label == 'Periodic_Thermal':
        period = float(rng.choice([480,600,720]))
        pulse_width = float(rng.choice([120,240]))
        active = (times >= 120) & (((times-120) % period) < pulse_width)
        field[np.ix_(active,core)] += amp
    elif label == 'Broad_Shift':
        field[:,x<center] += amp
    elif label == 'Sensor_Fault':
        field[int(rng.integers(len(times))), int(np.argmin(np.abs(x-center)))] += rng.uniform(14,20)
    return field, valid, x, times


def dataset(seeds, *, stress=False):
    features=[]
    targets=[]
    geometry=[]
    for seed in seeds:
        for index,label in enumerate(LABELS):
            field,valid,x,times=scene(seed,label,stress=stress)
            features.append(shape_features(field,valid,x,times))
            targets.append(index)
            geometry.append({'seed':seed,'label':label,'cadence_s':float(times[1]-times[0]),'spacing_m':float(x[1]-x[0])})
    return np.stack(features), np.array(targets), geometry


def probabilities(model,features):
    logits = (features-model.mu)/model.sig @ model.weight + model.bias
    logits -= logits.max(axis=1,keepdims=True)
    prob=np.exp(logits)
    return prob/prob.sum(axis=1,keepdims=True)


def predict(prob,threshold):
    pred=prob.argmax(axis=1)
    pred[prob.max(axis=1)<threshold]=0
    return pred


def score(target,pred):
    fg=target!=0
    bg=target==0
    matrix=np.zeros((len(LABELS),len(LABELS)),int)
    np.add.at(matrix,(target,pred),1)
    return {'scene_accuracy':float(np.mean(target==pred)),
            'foreground_class_recall':float(np.mean(target[fg]==pred[fg])),
            'background_false_positive_fraction':float(np.mean(pred[bg]!=0)),
            'n_background':int(bg.sum()),'n_false_background':int(np.count_nonzero(pred[bg]!=0)),
            'per_class_recall':{name:float(np.mean(pred[target==i]==i)) for i,name in enumerate(LABELS)},
            'confusion_matrix':matrix.tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/hardening/physical')
    parser.add_argument('--max-false-alarm-rate',type=float,default=.05)
    args=parser.parse_args()
    if not 0<=args.max_false_alarm_rate<=1: parser.error('budget must lie in [0,1]')
    args.out.mkdir(parents=True,exist_ok=True)
    train,y_train,_=dataset(TRAIN_SEEDS)
    validation,y_val,val_geometry=dataset(VALIDATION_SEEDS,stress=True)
    model=SoftmaxSceneModel()
    model.fit(train,y_train,steps=1000)
    val_prob=probabilities(model,validation)
    candidates=[{'threshold':t,**score(y_val,predict(val_prob,t))} for t in THRESHOLDS]
    eligible=[c for c in candidates if c['background_false_positive_fraction']<=args.max_false_alarm_rate]
    chosen=max(eligible,key=lambda c:(c['foreground_class_recall'],c['scene_accuracy'],-c['threshold'])) if eligible else None
    # The test data is not even generated until the candidate is fixed.
    test,y_test,test_geometry=dataset(TEST_SEEDS,stress=True)
    test_score=score(y_test,predict(probabilities(model,test),chosen['threshold'])) if chosen else None
    payload={'schema':SCHEMA,'features':list(FEATURE_NAMES),'labels':list(LABELS),
             'mu':model.mu.tolist(),'sig':model.sig.tolist(),'weight':model.weight.tolist(),'bias':model.bias.tolist(),
             'threshold':chosen['threshold'] if chosen else None,
             'serve':'research_offline','expert_confirmed':False}
    model_bytes=(json.dumps(payload,sort_keys=True,indent=2)+'\n').encode()
    (args.out/'physical_model.json').write_bytes(model_bytes)
    report={'protocol':'synthetic_offline_scene_classification_v1','not_event_detection':True,
            'expert_confirmed':False,'schema':SCHEMA,'features':list(FEATURE_NAMES),'labels':list(LABELS),
            'train_seeds':list(TRAIN_SEEDS),'validation_seeds':list(VALIDATION_SEEDS),'test_seeds':list(TEST_SEEDS),
            'max_false_alarm_rate':args.max_false_alarm_rate,'validation_candidates':candidates,
            'chosen_validation':chosen,'test':test_score,
            'test_budget_pass':bool(test_score and test_score['background_false_positive_fraction']<=args.max_false_alarm_rate),
            'model_sha256':hashlib.sha256(model_bytes).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'feature_code_sha256':hashlib.sha256((ROOT/'fiber/classify/physical.py').read_bytes()).hexdigest(),
            'runtime':{'python':platform.python_version(),'numpy':np.__version__},
            'validation_geometry':val_geometry,'test_geometry':test_geometry}
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('protocol','chosen_validation','test','test_budget_pass','model_sha256')},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
