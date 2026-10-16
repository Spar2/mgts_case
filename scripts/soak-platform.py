"""Authenticated loopback load and 30-minute resource sampling; no expert votes."""
import argparse,json,time,statistics,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx

parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=int,default=1800);args=parser.parse_args()
credentials=json.loads(Path('.runtime/local-admin.json').read_text(encoding='utf-8'))
clients=[httpx.Client(base_url='http://127.0.0.1:8081',timeout=120,trust_env=False) for _ in range(10)]
for client in clients:client.post('/api/v2/auth/login',json=credentials).raise_for_status()
records=clients[0].get('/api/v1/records').json()
for record in records:
 params=dict(zip(('x0','x1','t0','t1'),[*record['x_bounds'],*record['time_bounds']]),width=128,height=128,format_version=2)
 clients[0].get('/api/v1/records/'+record['id']+'/tiles',params=params).raise_for_status()
results=[];samples=[];errors=[];start=time.monotonic();cycle=0
with ThreadPoolExecutor(max_workers=10) as pool:
 while time.monotonic()-start<args.seconds:
  def read(index):
   record=records[(cycle+index)%len(records)];params=dict(zip(('x0','x1','t0','t1'),[*record['x_bounds'],*record['time_bounds']]),width=128,height=128,format_version=2)
   before=time.perf_counter();r=clients[index].get('/api/v1/records/'+record['id']+'/tiles',params=params);r.raise_for_status();return (time.perf_counter()-before)*1000
  for future in [pool.submit(read,i) for i in range(10)]:
   try:results.append(future.result())
   except Exception as e:errors.append(type(e).__name__)
  stats=subprocess.run(['docker','stats','--no-stream','--format','{{json .}}','haklolarp-web-api','haklolarp-web-stream-worker-1','haklolarp-web-redis-1','haklolarp-web-clickhouse-1'],capture_output=True,text=True)
  jobs=clients[0].get('/api/v2/jobs?limit=100').json()
  samples.append({'elapsed_s':round(time.monotonic()-start,1),'containers':[json.loads(line) for line in stats.stdout.splitlines() if line], 'jobs':[{'id':j['id'],'status':j['status'],'error':j['error']} for j in jobs['items']]})
  report={'duration_s':round(time.monotonic()-start,1),'sessions':10,'requests':len(results),'p95_ms':sorted(results)[int(.95*(len(results)-1))] if results else None,'errors':errors,'samples':samples,'conditions':'loopback Docker Desktop; ten independent authenticated clients; warm 128x128 version2 tiles; sample every 10s; replay jobs active when available; no hardware phone measurement'}
  Path('.runtime/platform-soak.json').write_text(json.dumps(report,indent=2),encoding='utf-8');cycle+=1
  if cycle%6==0:print(json.dumps({k:v for k,v in report.items() if k not in ('samples','conditions')}),flush=True)
  time.sleep(min(10,max(0,args.seconds-(time.monotonic()-start))))
for client in clients:client.close()
