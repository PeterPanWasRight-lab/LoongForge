#!/usr/bin/env python
"""Durable child launcher: stdout log and exit event survive console restarts."""
from __future__ import annotations
import json, os, signal, subprocess, sys, time
from pathlib import Path


def main():
    manifest=json.loads(Path(sys.argv[1]).read_text())
    env=dict(os.environ)
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','CONFIG'): env.pop(key,None)
    env.update(manifest['env']); env['PYTHONUNBUFFERED']='1'
    child=None; interrupted=False
    def stop(signum, frame):
        nonlocal interrupted
        interrupted=True
        # Job.stop signals our entire process group, including the child.
        # Forwarding again can interrupt the evaluator's server cleanup twice.
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
    code=1
    try:
        child=subprocess.Popen(manifest['argv'],cwd=manifest['cwd'],env=env,stdin=subprocess.DEVNULL)
        code=child.wait()
    except Exception as exc: print(f'[runner] {type(exc).__name__}: {exc}',flush=True)
    finally:
        if interrupted and code==0: code=130
        print(json.dumps({'t':'exit','code':code}),flush=True)
        target=Path(sys.argv[1]).with_name('exit.json')
        target.write_text(json.dumps({'code':code,'ended_at':time.time()}))
    return code

if __name__=='__main__': raise SystemExit(main())
