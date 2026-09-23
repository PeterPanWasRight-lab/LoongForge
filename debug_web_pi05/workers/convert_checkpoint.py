"""Adapt the native PI05Policy model/pi05 aliases; never modify input weights."""
import os,sys
from pathlib import Path
from safetensors.torch import load_file, save_file

source=Path(sys.argv[1]); target=Path(sys.argv[2]); target.parent.mkdir(parents=True,exist_ok=True)
raw=load_file(str(source),device='cpu')
out={}
for key,value in raw.items():
    if key.startswith('pi05.'): continue
    name=key[6:] if key.startswith('model.') else key
    if name in out: raise ValueError(f'duplicate converted key: {name}')
    out[name]=value
if not out: raise ValueError('conversion produced no tensors')
tmp=target.with_name(f'{target.name}.{os.getpid()}.tmp')
save_file(out,str(tmp)); tmp.replace(target)
print(f'[checkpoint] {len(raw)} → {len(out)} tensors; converted copy: {target}',flush=True)
