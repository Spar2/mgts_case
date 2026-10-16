"""Local-only fault check. Restarts only this project's Redis and ingester.

Run with the platform compose stack already running; requires Docker CLI.
Creates two technical profiles under a fresh replay UUID, never expert votes.
"""

import json
import subprocess
from pathlib import Path

HELPER = r'''
import json, os, sys, uuid, time
from pathlib import Path
import httpx, numpy as np
from workbench.store import Store
mode=sys.argv[1]; statefile=Path('/runtime/qa-recovery.json')
with httpx.Client(timeout=20) as client:
 base='http://stream-backend:8000'
 token=client.get(base+'/api/manifest').json()['mutation_token']
 headers={'X-DTS-Token':token}
 if mode=='init':
  store=Store(os.environ['DTS_RECORDS'],Path('/runtime/cache'))
  r=store.get('5'); meta=store.meta['5']; replay=uuid.uuid4().hex
  sid=client.post(base+'/api/session',headers=headers,json={'mode':'measurement_replay','schema_version':2,'replay_id':replay,'source_sha256':meta['source_sha256']})
  sid.raise_for_status(); bodies=[]
  for row in range(2):
   bodies.append({'session_id':sid.json()['session_id'],'source_row':row,'source_segment':int(r.segment_id[row]),'elapsed_s':float(r.elapsed_s[row]),'native_invalid_reason':r.invalid_reason[row].tolist(),
    'profile':{'dts_id':meta['dts_id'],'channel_id':meta['channel_id'],'timestamp':r.profiles[row]['timestamp'],'points':[[float(x),float(v) if np.isfinite(v) else None] for x,v in zip(r.x_m,r.raw_c[row])]}})
  statefile.write_text(json.dumps({'replay':replay,'sha':meta['source_sha256'],'bodies':bodies}))
  print(json.dumps({'initialized':True}));sys.exit()
 state=json.loads(statefile.read_text())
 def query(sql):
  result=client.post('http://clickhouse:8123',params={'param_replay':state['replay']},content=sql+' FORMAT JSONEachRow');result.raise_for_status();return [json.loads(line) for line in result.text.splitlines()]
 if mode=='down':
  response=client.post(base+'/api/infer',headers=headers,json=state['bodies'][0]);assert response.status_code==503,response.status_code
  print(json.dumps({'redis_outage_status':503}));sys.exit()
 if mode=='enqueue':
  for index,body in enumerate(state['bodies']):
   response=client.post(base+'/api/infer',headers=headers,json=body);response.raise_for_status();assert response.json()['profile_index']==index
  assert not query('SELECT row_idx FROM haklolarp.profiles_v2 FINAL WHERE replay_id={replay:String}')
  print(json.dumps({'accepted_after_retry':2,'persisted_while_ingester_stopped':0}));sys.exit()
 if mode=='verify':
  deadline=time.monotonic()+40
  while time.monotonic()<deadline:
   rows=query('SELECT row_idx FROM haklolarp.profiles_v2 FINAL WHERE replay_id={replay:String} ORDER BY row_idx')
   if len(rows)==2:break
   time.sleep(.5)
  assert [row['row_idx'] for row in rows]==[0,1]
  print(json.dumps({'persisted_after_ingester_restart':2}));sys.exit()
 if mode=='duplicate':
  sid=client.post(base+'/api/session',headers=headers,json={'mode':'measurement_replay','schema_version':2,'replay_id':state['replay'],'source_sha256':state['sha']});sid.raise_for_status()
  body=state['bodies'][0];body['session_id']=sid.json()['session_id']
  response=client.post(base+'/api/infer',headers=headers,json=body);response.raise_for_status()
  time.sleep(2)
  rows=query('SELECT count() AS n FROM haklolarp.profiles_v2 FINAL WHERE replay_id={replay:String}');assert int(rows[0]['n'])==2
  print(json.dumps({'logical_rows_after_duplicate':2}));sys.exit()
'''


def helper(mode):
    result = subprocess.run(
        ['docker', 'exec', '-i', 'haklolarp-web-worker-1', 'python', '-', mode],
        input=HELPER, text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def container(action, name):
    subprocess.run(['docker', action, name], check=True, capture_output=True)


if __name__ == '__main__':
    report = [helper('init')]
    try:
        container('stop', 'haklolarp-web-ingester-1')
        container('stop', 'haklolarp-web-redis-1')
        report.append(helper('down'))
        container('start', 'haklolarp-web-redis-1')
        report.append(helper('enqueue'))
    finally:
        container('start', 'haklolarp-web-redis-1')
        container('start', 'haklolarp-web-ingester-1')
    report.append(helper('verify'))
    report.append(helper('duplicate'))
    Path('.runtime/stream-recovery.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
