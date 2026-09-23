"""Read-only checkpoint state inventory; tensor loading is isolated and restricted."""
import json, subprocess, sys, threading
_LOCK=threading.Lock()
from .catalog import permitted

def inspect(path):
    p=permitted(path)
    if p.is_file(): p=p.parent
    if not _LOCK.acquire(blocking=False): raise ValueError('已有状态检查正在进行，请稍后重试')
    try:
        result=subprocess.run([sys.executable,'-m','app.state_reader',str(p)],capture_output=True,text=True,timeout=25)
    finally: _LOCK.release()
    if result.returncode: raise ValueError('状态读取失败：'+result.stderr[-800:])
    return json.loads(result.stdout)
