"""Prepare unlabelled candidate records for blind human review.

Detector proposals and random samples remain hypotheses. No expert label,
negative truth, train/test status or physical cause is invented by this tool.
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fiber.io import Record


def main():
    out=ROOT/'artifacts/hardening/annotation'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    rng=np.random.default_rng(20261006)
    for name in map(str,range(1,7)):
        rec=Record.load(ROOT/'artifacts/records'/name)
        zones=json.loads((ROOT/f'artifacts/anomaly/{name}/zones.json').read_text())
        for index,zone in enumerate(zones):
            a=int(np.searchsorted(rec.time_s,zone['t_start_s']))
            b=int(np.searchsorted(rec.time_s,zone['t_end_s'],side='right'))
            rows.append({'id':f'{name}-zone-{index:04d}','record':name,
                         'source_sha256':rec.source_sha256,'origin':'detector_proposal',
                         'rows_half_open':[a,b],'x_m_inclusive':[zone['start_m'],zone['end_m']],
                         'proposed_shape':zone['shape_hypothesis'],'proposal_peak_z':zone['peak_z'],
                         'expert_label':None,'review_status':'pending','reviewer':None,'confidence':None,
                         'physical_cause':None,'expert_confirmed':False})
        # Random review windows expose detector misses and difficult background.
        # They are never auto-labelled Background.
        live_x=np.flatnonzero(rec.valid.any(axis=0))
        for index in range(6):
            a=int(rng.integers(rec.n_profiles))
            b=int(np.searchsorted(rec.time_s,rec.time_s[a]+1800,side='right'))
            anchor=int(rng.choice(live_x))
            left=max(float(rec.x_m[0]),float(rec.x_m[anchor])-20)
            right=min(float(rec.x_m[-1]),float(rec.x_m[anchor])+20)
            rows.append({'id':f'{name}-random-{index:02d}','record':name,'source_sha256':rec.source_sha256,
                         'origin':'random_review','rows_half_open':[a,b],'x_m_inclusive':[left,right],
                         'proposed_shape':None,'expert_label':None,'review_status':'pending',
                         'reviewer':None,'confidence':None,'physical_cause':None,'expert_confirmed':False})
    # Shuffle and anonymise IDs as well as hiding the suggested class.
    rng.shuffle(rows)
    for index,row in enumerate(rows):
        row['blind_id']=f'review-{index:05d}'
    # Reviewers should first receive the blind copy without proposal hints.
    (out/'proposals.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    blind=[{**{k:v for k,v in row.items() if k not in {'proposed_shape','proposal_peak_z','origin','id','blind_id'}},
            'id':row['blind_id']} for row in rows]
    (out/'blind_review.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in blind),encoding='utf-8')
    manifest={'schema':'blind_review_queue_v1','seed':20261006,'n_items':len(rows),
              'n_detector_proposals':sum(r['origin']=='detector_proposal' for r in rows),
              'n_random_review':sum(r['origin']=='random_review' for r in rows),
              'expert_confirmed':False,'all_expert_labels_null':all(r['expert_label'] is None for r in rows),
              'already_seen_files':True,'independent_unseen_measurement_test':False,
              'source_sha256':{name:Record.load(ROOT/'artifacts/records'/name).source_sha256 for name in map(str,range(1,7))}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__': main()
