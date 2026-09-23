from __future__ import annotations
import hashlib,json,math,re,threading,time
from pathlib import Path
from urllib.parse import parse_qs,unquote
from .settings import settings,WEB_DIR,JOBS_DIR,WORKERS_DIR
from .responses import Json,Error,File,Raw,Stream
from .jobs import Job,registry
from .catalog import scan,run_detail,run_log,gpu_status,permitted
from .plans import build_plan

_CREATE_LOCK=threading.RLock()
_CACHE=None
_CACHE_TIME=0
_CACHE_LOCK=threading.RLock()

def clean(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list): return [clean(v) for v in value]
    return value

def reply(value): return Json(clean(value))
def web_build():
    h=hashlib.sha256()
    for p in sorted(WEB_DIR.glob('*')):
        if p.is_file(): h.update(p.read_bytes())
    return h.hexdigest()[:8]

def catalog(refresh=False):
    global _CACHE,_CACHE_TIME
    with _CACHE_LOCK:
        if refresh or _CACHE is None or time.monotonic()-_CACHE_TIME>30:
            _CACHE=scan(); _CACHE_TIME=time.monotonic()
        return _CACHE

def events(job):
    def pack(name,data): return f'event: {name}\ndata: {json.dumps(clean(data),ensure_ascii=False)}\n\n'.encode()
    def generate():
        offset=0; last=None; heartbeat=0
        yield pack('hello',job.to_dict())
        while True:
            chunk=job.log_since(offset); offset=chunk['offset']
            for line in chunk['lines']: yield pack('log',line)
            if job.status!=last:
                last=job.status; yield pack('status',job.to_dict())
            if job.status in ('succeeded','failed','stopped','unknown') and offset>=chunk['total']:
                yield pack('done',job.to_dict()); return
            heartbeat+=1
            if heartbeat%20==0: yield b': heartbeat\n\n'
            time.sleep(.5)
    return Stream(generate(),'text/event-stream; charset=utf-8',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

def dispatch(method,path,query_string,body):
    q=parse_qs(query_string); query=lambda key,default='':(q.get(key) or [default])[0]
    try:
        if path.startswith('/api/weights/'):
            from . import weights
            if path=='/api/weights/analyze' and method=='POST': return reply(weights.launch(body or {}))
            if path=='/api/weights/status' and method=='GET': return reply(weights.status(query('id')))
            if path=='/api/weights/cancel' and method=='POST': return reply(weights.status((body or {}).get('id'),cancel=True))
            if path=='/api/weights/tensor' and method=='POST': return reply(weights.heatmap(body or {}))
            return Error('接口不存在',404)
        if path=='/api/ping': return reply({'ok':True,'web_build':web_build()})
        if path=='/api/config':
            if method=='GET': return reply(settings.all())
            if method=='POST': return reply(settings.update(body or {}))
        if path=='/api/catalog' and method=='GET': return reply(catalog(query('refresh')=='1'))
        if path=='/api/gpus' and method=='GET': return reply(gpu_status())
        if path=='/api/preflight' and method=='POST':
            return reply(build_plan((body or {}).get('kind'),(body or {}).get('params',{})))
        if path=='/api/runlog' and method=='GET': return reply(run_log(query('path')))
        if path=='/api/run' and method=='GET': return reply(run_detail(query('path')))
        if path=='/api/file' and method=='GET':
            file=permitted(query('path'))
            if file.suffix not in ('.mp4','.gif','.webm','.csv','.json','.jsonl','.yaml','.log'): return Error('不支持此文件类型',403)
            return File(file)
        if path=='/api/jobs':
            if method=='GET': return reply({'items':registry.list(limit=200)})
            if method=='POST':
                with _CREATE_LOCK:
                    data=body or {}; plan=build_plan(data.get('kind'),data.get('params',{}))
                    if not plan['ok']: return Error('体检未通过',400,checks=plan['checks'])
                    # A job-owned output directory may never be concurrently claimed.
                    for active in registry.list(limit=10000):
                        if active['status'] in ('pending','running','stopping') and active['params'].get('output')==plan['output']:
                            return Error('这个输出目录已有任务正在使用',409)
                    output=Path(plan['output']); output.mkdir(parents=True,exist_ok=True)
                    params={**plan['params'],'output':plan['output']}
                    job=Job(plan['kind'],[],cwd=plan['cwd'],params=params,title=f"PI05 {'训练' if plan['kind']=='train' else 'LIBERO评测'} · {output.name}")
                    if plan['config']:
                        (output/'console_eval.json').write_text(json.dumps(plan['config'],ensure_ascii=False,indent=2))
                    (output/f'console_plan_{job.id}.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
                    manifest=job.dir/'launch.json'; manifest.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
                    job.argv=[settings.get('python'),str(WORKERS_DIR/'runner.py'),str(manifest)]
                    job._write_meta(); registry.create(job); job.start()
                    return reply({'job':job.to_dict()})
        match=re.fullmatch(r'/api/jobs/([\w-]+)(?:/(events|stop|delete|log))?',path)
        if match:
            job=registry.get(match[1]); action=match[2]
            if not job: return Error('任务不存在',404)
            if action=='events' and method=='GET': return events(job)
            if action=='stop' and method=='POST': return reply(job.stop(block=False))
            if action=='delete' and method=='POST': registry.delete(job.id); return reply({'ok':True})
            if action=='log' and method=='GET': return reply(job.log_snapshot(2000))
            if action is None and method=='GET':
                detail=run_detail(job.params['output']) if job.params.get('output') else {}
                return reply({'job':job.to_dict(),'run':detail})
            return Error('方法不支持',405)
        if path.startswith('/api/'): return Error('接口不存在',404)
        if method!='GET': return Error('方法不支持',405)
        target=(WEB_DIR/unquote(path.lstrip('/') or 'index.html')).resolve()
        if WEB_DIR.resolve() not in target.parents: return Error('禁止访问',403)
        if target.name=='index.html':
            return Raw(target.read_bytes().replace(b'__BUILD__',web_build().encode()),'text/html; charset=utf-8',headers={'Cache-Control':'no-store'})
        if target.is_file():
            response=File(target); response.headers={'Cache-Control':'no-store'}; return response
        return Error('文件不存在',404)
    except (ValueError,TypeError) as exc: return Error(str(exc),400)
    except FileNotFoundError as exc: return Error(str(exc),404)

def runtime_paths(): return {'jobs_dir':str(JOBS_DIR),'repo_root':settings.get('repo_root')}
