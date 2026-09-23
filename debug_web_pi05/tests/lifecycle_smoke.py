"""Run only in a disposable staging console directory, no GPU or output writes."""
import json,os,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.jobs import Job,registry
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get(path):
    with opener.open('http://127.0.0.1:18901'+path,timeout=5) as r:return json.load(r)
def service(action):subprocess.run([sys.executable,str(ROOT/'service.py'),action],check=True,env={**os.environ,'PI05_CONSOLE_PORT':'18901'})
launcher="""
import json,sys
from pathlib import Path
from app.jobs import Job
root=Path.cwd()
job=Job('train',[],cwd=str(root),title='CPU restart smoke')
manifest=job.dir/'launch.json';manifest.write_text(json.dumps({'argv':[sys.executable,'-c','import time;print("CPU-only",flush=True);time.sleep(5);print("done",flush=True)'],'cwd':str(root),'env':{}}))
job.argv=[sys.executable,str(root/'workers/runner.py'),str(manifest)]
job.start();print(job.id,flush=True)
"""
job_id=subprocess.check_output([sys.executable,'-c',launcher],cwd=ROOT,text=True).strip()
# No original parent survives to write terminal metadata.
service('stop');service('start')
items=get('/api/jobs')['items'];assert any(j['id']==job_id for j in items)
service('stop')
time.sleep(6)
service('start');items=get('/api/jobs')['items'];result=next(j for j in items if j['id']==job_id)
assert result['status']=='succeeded',result
print('Lifecycle: running-job reattach and finished-job recovery passed')
