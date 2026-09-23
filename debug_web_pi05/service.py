#!/usr/bin/env python3
"""Start/stop only this console. GPU jobs survive web service restarts."""
import fcntl,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime'; RUNTIME.mkdir(exist_ok=True)
PIDFILE=RUNTIME/'service.json'
def identity(pid):
    try:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        start=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        return start if str(ROOT/'server.py').encode() in command else None
    except (OSError,IndexError): return None
def main():
    action=sys.argv[1] if len(sys.argv)>1 else 'start'
    if action not in ('start','stop','status'): raise SystemExit('用法: service.py start|stop|status')
    with (RUNTIME/'service.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old=json.loads(PIDFILE.read_text()) if PIDFILE.exists() else {}
        alive=bool(old.get('pid') and identity(old['pid'])==old.get('identity'))
        if action=='status': print('运行中' if alive else '已停止'); return
        if action=='stop':
            if alive:
                os.kill(old['pid'],signal.SIGTERM)
                for _ in range(100):
                    if identity(old['pid'])!=old['identity']: break
                    time.sleep(.1)
                else: raise SystemExit('服务未退出；没有强杀或影响训练任务')
            PIDFILE.unlink(missing_ok=True); print('网页服务已停止，训练/评测继续运行'); return
        if alive: print(f"已运行，PID {old['pid']}"); return
        port=int(os.environ.get('PI05_CONSOLE_PORT','8901'))
        with (RUNTIME/'server.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'server.py'),'--port',str(port)],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(100):
            if proc.poll() is not None: raise SystemExit('服务启动失败，请查看 runtime/server.log')
            try:
                with opener.open(f'http://127.0.0.1:{port}/api/ping',timeout=.5) as response:
                    assert json.load(response)['ok']
                # A successful bind precedes this server's response.
                time.sleep(.2)
                if proc.poll() is not None: raise SystemExit('端口已被占用，请查看 runtime/server.log')
                token=identity(proc.pid)
                if not token: raise SystemExit('无法确认服务进程')
                PIDFILE.write_text(json.dumps({'pid':proc.pid,'identity':token,'port':port}))
                print(f'PI05 控制台已启动：http://172.16.107.56:{port}/'); return
            except (OSError,ValueError,AssertionError): time.sleep(.1)
        os.kill(proc.pid,signal.SIGTERM)
        raise SystemExit('服务健康检查超时，请查看 runtime/server.log')
if __name__=='__main__': main()
