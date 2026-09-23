"""Bounded artifact discovery and read-only metrics; no torch import."""
from __future__ import annotations
import json, math, os, struct, subprocess, time
from collections import deque
from pathlib import Path
from .settings import settings, JOBS_DIR


def read_json(path, default=None):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return default


def walk(root, depth=6):
    root=Path(root)
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('videos','data','tokenizer','tensorboard','__pycache__','converted')]
        if len(Path(current).relative_to(root).parts) >= depth: dirs[:]=[]
        yield Path(current), files


def weight_layout(path):
    """Read only safetensors header, not multi-GB weights."""
    with Path(path).open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8: raise ValueError('权重头不完整')
        n=struct.unpack('<Q',raw)[0]
        if n > 64*1024*1024: raise ValueError('权重头异常')
        head=json.loads(f.read(n))
    keys=[k for k in head if k!='__metadata__']
    if not keys: raise ValueError('权重没有 tensor')
    return {'layout':'wrapped' if any(k.startswith(('model.','pi05.')) for k in keys) else 'raw', 'tensors':len(keys)}


def scan():
    checkpoints=[]; datasets=[]; runs=[]
    for root in dict.fromkeys([settings.get('artifacts_root'), settings.get('output_root')]):
        for path, files in walk(root):
            if 'model.safetensors' in files:
                cfg=read_json(path/'config.json',{})
                # LoongForge's native checkpoints have resume_meta but no config.json.
                if cfg.get('type') not in (None,'pi05') or ('pi05' not in str(path).lower() and cfg.get('type')!='pi05'): continue
                meta=read_json(path/'resume_meta.json',{})
                checkpoints.append({'path':str(path),'name':str(path.relative_to(Path(root))),
                    'step':meta.get('completed_steps'), 'resumable':(path/'resume_meta.json').is_file() and any((path/name).is_file() for name in ('training_state.pt','zero_optimizer/metadata.json','dcp/.metadata')),
                    'mtime':(path/'model.safetensors').stat().st_mtime, 'gb':round((path/'model.safetensors').stat().st_size/1024**3,2)})
            if path.name=='meta' and 'info.json' in files:
                info=read_json(path/'info.json',{})
                feat=info.get('features',{})
                datasets.append({'path':str(path.parent),'name':path.parent.name,'episodes':info.get('total_episodes'),
                    'fps':info.get('fps'),'state_dim':(feat.get('observation.state',{}).get('shape') or [None])[0],
                    'action_dim':(feat.get('action',{}).get('shape') or [None])[0],
                    'cameras':[k for k in feat if k.startswith('observation.images.')]})
            if 'metrics.jsonl' in files or 'results.jsonl' in files:
                kind='train' if 'metrics.jsonl' in files else 'eval'
                source=path/('metrics.jsonl' if kind=='train' else 'results.jsonl')
                runs.append({'path':str(path),'name':path.name,'kind':kind,'mtime':source.stat().st_mtime})
    checkpoints.sort(key=lambda x:x['mtime'],reverse=True); runs.sort(key=lambda x:x['mtime'],reverse=True)
    return {'checkpoints':checkpoints,'datasets':datasets,'runs':runs}


def permitted(path):
    p=Path(path).expanduser().resolve()
    roots=[Path(settings.get(k)).resolve() for k in ('output_root','artifacts_root')]+[JOBS_DIR.resolve()]
    if not any(p==r or r in p.parents for r in roots): raise ValueError('路径不在控制台允许的目录内')
    return p


def jsonl_tail(path, count=3000):
    try:
        with Path(path).open('rb') as f:
            f.seek(0,2); size=f.tell(); f.seek(max(0,size-8*1024*1024))
            if f.tell(): f.readline()
            lines=deque(f,maxlen=count)
        return [v for line in lines if (v:=parse_line(line)) is not None]
    except OSError: return []

def parse_line(line):
    try:
        v=json.loads(line)
        return v if isinstance(v,dict) else None
    except (ValueError,UnicodeDecodeError): return None


def run_detail(path):
    root=permitted(path)
    metrics=jsonl_tail(root/'metrics.jsonl')
    episodes=jsonl_tail(root/'results.jsonl',10000)
    # A retry is not another independent evaluation episode.
    dedup={str(e.get('episode_id',f"{e.get('task_id')}/{e.get('episode_idx')}") ):e for e in episodes}
    episodes=list(dedup.values()); tasks={}
    for e in episodes:
        key=str(e.get('task_id',e.get('task','?')))
        row=tasks.setdefault(key,{'task':e.get('task',key),'total':0,'success':0})
        row['total']+=1; row['success']+=int(e.get('success')==1)
    files=[]
    if root.exists():
        for current, _, names in os.walk(root):
            if len(files)>=80: break
            for name in names:
                p=Path(current)/name
                if p.suffix in ('.mp4','.gif','.webm','.csv','.json','.yaml') and p.stat().st_size < 1024**3:
                    files.append({'name':str(p.relative_to(root)),'path':str(p)})
    return {'path':str(root),'metrics':metrics,'latest':metrics[-1] if metrics else {},
        'episodes':episodes[-500:], 'tasks':list(tasks.values()),'completed':len(episodes),
        'successes':sum(e.get('success')==1 for e in episodes),'files':files[:80]}


def gpu_status():
    try:
        out=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True).stdout
        cards=[]
        for line in out.splitlines():
            i,name,used,total,util=[x.strip() for x in line.split(',')]
            cards.append({'index':i,'name':name,'used_mb':float(used),'total_mb':float(total),'util':float(util)})
        result=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,timeout=3).stdout
        processes=[line.strip() for line in result.splitlines() if ('loongforge/embodied/train.py' in line or 'loongforge.embodied.eval.servers.loongforge_server' in line) and 'bash -c' not in line and 'grep' not in line]
        return {'items':cards,'processes':processes}
    except (OSError,ValueError,subprocess.SubprocessError) as exc: return {'items':[],'processes':[],'error':str(exc)}


def run_log(path):
    root=permitted(path)
    candidates=[p for pattern in ('*.log','*.out','logs/*.log') for p in root.glob(pattern) if p.is_file()]
    if not candidates: return {'text':'','path':None}
    chosen=max(candidates,key=lambda p:p.stat().st_mtime)
    permitted(chosen)
    with chosen.open('rb') as f:
        f.seek(0,2); size=f.tell(); f.seek(max(0,size-128*1024))
        if f.tell(): f.readline()
        text=f.read(128*1024).decode('utf-8',errors='replace')
    return {'text':text,'path':str(chosen)}
