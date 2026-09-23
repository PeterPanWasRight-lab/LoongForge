"""CPU-only, bounded sampling of safetensors. Never instantiate a model."""
from __future__ import annotations
import json, math, mmap, struct, threading, time, uuid
from collections import OrderedDict
from pathlib import Path
import numpy as np
from .catalog import permitted

DTYPES={'F32':'<f4','F64':'<f8','F16':'<f2','BF16':'<u2','I64':'<i8','I32':'<i4','I16':'<i2','I8':'i1','U8':'u1','BOOL':'?'}
LABELS={'vision':'视觉塔 · Vision Tower','projector':'视觉投影 · Multimodal Projector','paligemma':'PaliGemma · 语言骨干','expert':'Action Expert','action':'动作输入 / 输出投影','time':'时间步 MLP','other':'其他参数'}

def canonical(key):
    for prefix in ('model.','pi05.'):
        if key.startswith(prefix): return key[len(prefix):]
    return key

def group(key):
    if 'vision_tower.' in key: return 'vision'
    if 'multi_modal_projector.' in key: return 'projector'
    if 'gemma_expert.' in key: return 'expert'
    if 'paligemma.' in key: return 'paligemma'
    if key.startswith(('action_in_proj.','action_out_proj.')): return 'action'
    if key.startswith(('time_mlp','action_time_mlp')): return 'time'
    return 'other'

class Weights:
    def __init__(self,path):
        self.file=None;self.map=None
        p=permitted(path)
        if p.is_dir(): p=p/'model.safetensors'
        p=permitted(p)
        if p.suffix!='.safetensors': raise ValueError('仅支持 safetensors 权重文件或包含 model.safetensors 的目录')
        self.path=str(p);self.file=p.open('rb')
        try:
            size=self.file.seek(0,2);self.file.seek(0)
            raw=self.file.read(8)
            if len(raw)!=8: raise ValueError('权重头不完整')
            n=struct.unpack('<Q',raw)[0]
            if not 0<n<=64*1024**2 or n+8>size: raise ValueError('权重头长度非法')
            header=json.loads(self.file.read(n));self.offset=8+n;self.entries={};self.aliases=0
            for name,entry in sorted(header.items(),key=lambda item:(0 if item[0].startswith('model.') else 2 if item[0].startswith('pi05.') else 1,item[0])):
                if name=='__metadata__':continue
                shape=entry['shape'];offsets=entry['data_offsets'];dtype=entry['dtype']
                if not isinstance(shape,list) or any(type(x)!=int or x<0 for x in shape):raise ValueError('非法张量形状')
                if len(offsets)!=2 or any(type(x)!=int for x in offsets) or not 0<=offsets[0]<=offsets[1]<=size-self.offset:raise ValueError('张量偏移越界')
                count=math.prod(shape)
                if dtype in DTYPES and offsets[1]-offsets[0]!=count*np.dtype(DTYPES[dtype]).itemsize:raise ValueError('张量字节数与形状不一致')
                key=canonical(name)
                item={**entry,'source':name,'key':key,'numel':count,'group':group(key)}
                if key in self.entries:
                    previous=self.entries[key]
                    if previous['shape']!=shape or previous['dtype']!=dtype:raise ValueError(f'别名形状或类型冲突：{key}')
                    self.aliases+=1;continue
                self.entries[key]=item
            self.map=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
        except BaseException:self.close();raise
    def close(self):
        if self.map is not None:self.map.close();self.map=None
        if self.file is not None:self.file.close();self.file=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def take(self,key,indices):
        e=self.entries[key];dtype=e['dtype']
        if dtype not in DTYPES:raise ValueError(f'暂不支持数值类型 {dtype}')
        raw=np.ndarray((e['numel'],),dtype=DTYPES[dtype],buffer=self.map,offset=self.offset+e['data_offsets'][0])
        values=raw[indices]
        if dtype=='BF16':values=(values.astype(np.uint32)<<16).view(np.float32)
        return values.astype(np.float64)
    def info(self):return {'path':self.path,'tensors':len(self.entries),'aliases_removed':self.aliases,'parameters':sum(e['numel'] for e in self.entries.values())}

def sample_indices(n):
    if n<=4096:return np.arange(n,dtype=np.int64)
    # Sixteen distributed contiguous blocks: bounded I/O, identical positions in A/B.
    starts=np.linspace(0,n-256,16,dtype=np.int64)
    return (starts[:,None]+np.arange(256)).ravel()

def stats(values):
    finite=values[np.isfinite(values)]
    if not len(finite):return {'count':len(values),'finite':0,'rms':None,'mean':None,'mean_abs':None,'max_abs':None,'min':None,'max':None,'std':None,'zero_fraction':None}
    return {'count':len(values),'finite':len(finite),'rms':float(np.sqrt(np.mean(finite**2))), 'mean':float(finite.mean()),'mean_abs':float(np.abs(finite).mean()),'max_abs':float(np.abs(finite).max()),'min':float(finite.min()),'max':float(finite.max()),'std':float(finite.std()),'zero_fraction':float(np.mean(finite==0))}

def compare(a,b):
    keep=np.isfinite(a)&np.isfinite(b);a=a[keep];b=b[keep]
    if not len(a):return {'diff_rms':None,'relative':None,'cosine':None,'paired':0}
    aa=float(np.mean(a*a));bb=float(np.mean(b*b));ab=float(np.mean(a*b));dd=float(np.mean((a-b)**2))
    return {'diff_rms':math.sqrt(dd),'relative':math.sqrt(dd/bb) if bb else None,'cosine':max(-1,min(1,ab/math.sqrt(aa*bb))) if aa and bb else None,'paired':len(a),'aa':aa,'bb':bb,'ab':ab,'dd':dd}

def thumbnail(reader,key):
    e=reader.entries[key];shape=e['shape']
    rows=shape[0] if len(shape)>1 else 1
    cols=math.prod(shape[1:]) if len(shape)>1 else e['numel']
    if not rows or not cols:return []
    ri=np.linspace(0,rows-1,min(rows,16),dtype=np.int64)
    ci=np.linspace(0,cols-1,min(cols,16),dtype=np.int64)
    values=reader.take(key,(ri[:,None]*cols+ci).ravel()).reshape(len(ri),len(ci))
    return [[float(x) if np.isfinite(x) else None for x in row] for row in values]

def row_for(a,b,key):
    ea=a.entries.get(key);eb=b.entries.get(key) if b else None;e=ea or eb
    row={'key':key,'group':e['group'],'shape':e['shape'],'numel':e['numel'],'dtype_a':ea['dtype'] if ea else None,'dtype_b':eb['dtype'] if eb else None,'shape_b':eb['shape'] if eb else None,'status':'single' if b is None else 'matched' if ea and eb and ea['shape']==eb['shape'] else 'shape_mismatch' if ea and eb else 'only_a' if ea else 'only_b','exact':e['numel']<=4096}
    if ea:
        av=a.take(key,sample_indices(ea['numel']));row['a']=stats(av)
    if eb:
        bv=b.take(key,sample_indices(eb['numel']));row['b']=stats(bv)
    row['thumbnail']={'a':thumbnail(a,key) if ea else None,'b':thumbnail(b,key) if eb else None}
    if row['status']=='matched':row.update(compare(av,bv))
    return row

def modules(rows):
    result=[]
    for key,label in LABELS.items():
        subset=[r for r in rows if r['group']==key]
        if not subset:continue
        ar=[r for r in subset if r.get('a',{}).get('rms') is not None]
        matched=[r for r in subset if r.get('paired',0)>0]
        total=sum(r['numel'] for r in ar);paired=sum(r['numel'] for r in matched)
        item={'key':key,'label':label,'tensors':len(subset),'parameters_a':total,'matched_parameters':paired,'unmatched':sum(r['status'] not in ('matched','single') for r in subset),'rms':math.sqrt(sum(r['a']['rms']**2*r['numel'] for r in ar)/total) if total else None,'nonfinite_samples':sum(r.get('a',{}).get('count',0)-r.get('a',{}).get('finite',0)+r.get('b',{}).get('count',0)-r.get('b',{}).get('finite',0) for r in subset)}
        if paired:
            moments={k:sum(r[k]*r['numel'] for r in matched)/paired for k in ('aa','bb','ab','dd')}
            item.update(diff_rms=math.sqrt(moments['dd']),relative=math.sqrt(moments['dd']/moments['bb']) if moments['bb'] else None,cosine=max(-1,min(1,moments['ab']/math.sqrt(moments['aa']*moments['bb']))) if moments['aa'] and moments['bb'] else None)
        result.append(item)
    return result

_LOCK=threading.RLock();_RUNS=OrderedDict()
def launch(payload):
    # Open before returning: fd/mmap keeps weights readable if original is unlinked.
    with _LOCK:
        if any(x['status'] in ('running','cancelling') for x in _RUNS.values()):raise ValueError('已有权重分析正在运行，请等待或取消后再分析')
        a=Weights(payload.get('a',''));b=None
        try:
            if payload.get('b'):b=Weights(payload['b'])
        except BaseException:a.close();raise
        ident=uuid.uuid4().hex[:12];event=threading.Event()
        state={'id':ident,'status':'running','progress':0,'total':len(set(a.entries)|set(b.entries if b else {})),'event':event,'started':time.time()};_RUNS[ident]=state
        while len(_RUNS)>4:_RUNS.popitem(last=False)
        def work():
            try:
                rows=[]
                for key in sorted(set(a.entries)|set(b.entries if b else {})):
                    if event.is_set():break
                    try:row=row_for(a,b,key)
                    except ValueError as exc:row={'key':key,'group':group(key),'numel':0,'status':'unsupported','error':str(exc)}
                    rows.append(row)
                    with _LOCK:state['progress']=len(rows)
                with _LOCK:
                    if event.is_set():state['status']='cancelled'
                    else:state.update(status='done',result={'a':a.info(),'b':b.info() if b else None,'rows':rows,'modules':modules(rows),'method':'每个张量最多4096个值：16个等距连续块；模块按参数量加权，均为抽样估计。小张量全量。差值=A−B，相对变化=||A−B||₂/||B||₂；B为零时不定义。'})
            except Exception as exc:
                with _LOCK:state.update(status='failed',error=str(exc))
            finally:
                a.close()
                if b:b.close()
                with _LOCK:state['seconds']=round(time.time()-state['started'],2)
        threading.Thread(target=work,name='weight-analysis',daemon=True).start()
        return {'id':ident}

def status(ident,cancel=False):
    with _LOCK:
        if ident not in _RUNS:raise ValueError('分析已过期或服务已重启，请重新分析')
        state=_RUNS[ident]
        if cancel and state['status']=='running':state['event'].set();state['status']='cancelling'
        return {k:v for k,v in state.items() if k not in ('event','started')}

def heatmap(payload):
    key=payload.get('key','')
    with Weights(payload.get('a','')) as a:
        b=Weights(payload['b']) if payload.get('b') else None
        try:
            if key not in a.entries:raise ValueError('此参数仅存在于 B，不能显示 A 热图')
            e=a.entries[key];shape=e['shape'];rows=shape[0] if len(shape)>1 else 1;cols=math.prod(shape[1:]) if len(shape)>1 else e['numel']
            if not rows or not cols:raise ValueError('空张量无热图')
            def interval(name,size):
                raw=payload.get(name)
                if not raw:return np.linspace(0,size-1,min(size,64),dtype=np.int64)
                if not isinstance(raw,list) or len(raw)!=2 or any(type(x)!=int for x in raw) or not 0<=raw[0]<raw[1]<=size:raise ValueError(f'{name} 必须为合法的 [起始,结束) 区间')
                return np.linspace(raw[0],raw[1]-1,min(raw[1]-raw[0],64),dtype=np.int64)
            ri=interval('rows',rows);ci=interval('cols',cols);indices=(ri[:,None]*cols+ci).ravel()
            av=a.take(key,indices);bv=None
            if b and key in b.entries and b.entries[key]['shape']==shape:bv=b.take(key,indices)
            def grid(v):return [[float(x) if np.isfinite(x) else None for x in row] for row in v.reshape(len(ri),len(ci))]
            sample=a.take(key,sample_indices(e['numel']));finite=sample[np.isfinite(sample)]
            hist,edges=np.histogram(finite,bins=40) if len(finite) else (np.zeros(40),np.arange(41))
            return {'key':key,'shape':shape,'rows':ri.tolist(),'cols':ci.tolist(),'matrix_shape':[rows,cols],'a':grid(av),'b':grid(bv) if bv is not None else None,'delta':grid(av-bv) if bv is not None else None,'stats_a':stats(sample),'histogram':{'counts':hist.tolist(),'edges':edges.tolist()},'note':'网格为原始坐标的等距点采样（非区块均值）；高阶张量将第1维作行、其余维展平作列。直方图使用与模块统计相同的抽样。'}
        finally:
            if b:b.close()
