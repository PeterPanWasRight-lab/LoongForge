import json, math, sys
from pathlib import Path

def inspect(root):
    p=Path(root); out={'path':str(p),'meta':{},'files':[],'sections':{},'moments':[],'optimizer_steps':[],'notes':[]}
    for name in ('resume_meta.json','zero_optimizer/metadata.json'):
        f=p/name
        if f.is_file() and not f.is_symlink() and f.stat().st_size<1024*1024:
            try: out['meta'][name]=json.loads(f.read_text())
            except Exception: out['notes'].append(name+' 无法解析')
    names=['training_state.pt','scheduler.pt','dcp/.metadata']
    for folder in ('rng','dataloader','zero_optimizer'):
        names.extend(str(f.relative_to(p)) for f in sorted((p/folder).glob('*.pt'))[:128])
    for name in names:
        f=p/name
        if f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(p.resolve()): out['files'].append({'name':name,'bytes':f.stat().st_size})
    if not out['files']: out['notes'].append('此目录未发现训练状态文件；权重导出不等于完整续训 checkpoint。')
    torch=None
    def brief(x,depth=0):
        if torch is not None and torch.is_tensor(x):
            if x.numel()==1:return brief(x.item())
            return {'shape':list(x.shape),'dtype':str(x.dtype),'numel':x.numel()}
        if x is None or isinstance(x,(str,bool,int)):return x
        if isinstance(x,float):return x if math.isfinite(x) else str(x)
        if depth>5:return '…'
        if isinstance(x,dict):return {str(k):brief(v,depth+1) for k,v in list(x.items())[:80]}
        if isinstance(x,(tuple,list)):return [brief(v,depth+1) for v in x[:32]]
        return type(x).__name__
    def moments(x,prefix=''):
        if isinstance(x,dict):
            for k,v in x.items():
                key=str(k); path=prefix+'/'+key
                if key=='step' and (isinstance(v,(int,float)) or (torch.is_tensor(v) and v.numel()==1)):
                    out['optimizer_steps'].append({'key':path,'step':brief(v)})
                if torch.is_tensor(v) and key in ('exp_avg','exp_avg_sq','master_param','master_params','fp32_param'):
                    flat=v.detach().reshape(-1); n=flat.numel()
                    sample=flat[torch.linspace(0,n-1,min(n,1024)).long()].float() if n else flat.float()
                    good=sample[torch.isfinite(sample)]; vals=good.double()
                    out['moments'].append({'key':path,'kind':key,'shape':list(v.shape),'dtype':str(v.dtype),'numel':n,'rms':float(vals.square().mean().sqrt()) if vals.numel() else None,'max_abs':float(vals.abs().max()) if vals.numel() else None,'zero_fraction':float((good==0).float().mean()) if good.numel() else None,'nonfinite':int(sample.numel()-good.numel())})
                elif isinstance(v,(dict,list,tuple)):moments(v,path)
        elif isinstance(x,(list,tuple)):
            for i,v in enumerate(x):moments(v,prefix+'/'+str(i))
    for entry in out['files']:
        name=entry['name']; f=p/name
        if name=='dcp/.metadata':
            out['notes'].append('DCP 分片存在；本页不反序列化 DCP metadata，仅确认文件存在，不推断 Adam 或 Master Parameters 已保存。');continue
        if entry['bytes']>512*1024**2:
            out['sections'][name]={'status':'文件存在，超过 512 MiB 读取上限；未加载数值'};continue
        try:
            if torch is None:
                import torch as torch_module
                torch=torch_module;torch.set_num_threads(1)
            state=torch.load(f,map_location='cpu',weights_only=True,mmap=True)
            moments(state,name)
            if isinstance(state,dict) and 'optimizer' in state:
                opt=state['optimizer']; state={**state,'optimizer':{'param_groups':opt.get('param_groups',[]) if isinstance(opt,dict) else brief(opt),'state_entries':len(opt.get('state',{})) if isinstance(opt,dict) else None}}
            out['sections'][name]=brief(state)
        except Exception as e:out['sections'][name]={'status':'文件存在，受限读取未成功（未使用不安全 pickle 回退）','reason':str(e)[:400]}
    return out
if __name__=='__main__':print(json.dumps(inspect(sys.argv[1]),ensure_ascii=False,allow_nan=False))
