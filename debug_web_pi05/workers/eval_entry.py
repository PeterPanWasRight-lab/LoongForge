#!/usr/bin/env python
"""Native LIBERO orchestration with the machine's validated PI05 state mapping.

PI05 embeds q99-normalized state in its text prompt. LIBERO state is
position(3) + robosuite quaternion-to-axis-angle(3) + finger positions(2).
This process-local adapter follows the existing launch_libero_eval_stateful.py.
"""
from __future__ import annotations
import argparse, errno, hashlib, json, os, runpy, shutil, signal, struct, subprocess, sys, tempfile, threading, time
from urllib.request import build_opener, ProxyHandler
from pathlib import Path


def encode_libero_state(canonical):
    import numpy as np
    raw=canonical.get('state_raw') or {}
    pos=np.asarray(raw.get('eef_pos'),dtype=np.float32).reshape(-1)
    grip=np.asarray(raw.get('gripper_qpos'),dtype=np.float32).reshape(-1)
    quat=np.asarray(raw.get('eef_quat'),dtype=np.float64).reshape(-1)
    if pos.size!=3 or grip.size<2 or quat.size!=4 or not all(np.isfinite(v).all() for v in (pos,grip,quat)):
        raise ValueError('LIBERO 缺少有效8维状态，拒绝回退为无状态提示')
    w=float(np.clip(quat[3],-1,1)); den=float(np.sqrt(max(0,1-w*w)))
    aa=np.zeros(3,dtype=np.float32) if den<1e-12 else (quat[:3]*2*np.arccos(w)/den).astype(np.float32)
    return np.concatenate([pos,aa,grip[:2]]).astype(np.float32)


def pin_checkpoint(source, directory):
    """Keep immutable weight files alive across external checkpoint pruning."""
    pinned=Path(directory)/'model.safetensors'
    try:
        os.link(source,pinned)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV,errno.EPERM,errno.EOPNOTSUPP): raise
        shutil.copyfile(source,pinned)
    print(f'[checkpoint] pinned for this evaluation: {source}',flush=True)
    return pinned


def prepare_checkpoint(config, python, pin_dir):
    original=Path(config['server']['ckpt_path'])/'model.safetensors'
    source=pin_checkpoint(original,pin_dir)
    with source.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        if n>64*1024*1024: raise ValueError('invalid safetensors header')
        header=json.loads(f.read(n))
    config['server']['ckpt_path']=str(source.parent)
    if not any(k.startswith(('model.','pi05.')) for k in header): return
    stat=source.stat()
    key=hashlib.sha256(f'{original.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:20]
    runtime=Path(os.environ.get('PI05_CONSOLE_RUNTIME',Path(__file__).resolve().parents[1]/'runtime'))
    target=runtime/'converted'/key/'model.safetensors'
    if not target.is_file():
        subprocess.run([python,str(Path(__file__).with_name('convert_checkpoint.py')),str(source),str(target)],check=True)
    config['server']['ckpt_path']=str(target.parent)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--train-python',required=True)
    args=parser.parse_args(); path=Path(args.config); config=json.loads(path.read_text())
    print('[startup] 正在准备 checkpoint，随后构建 PI05、加载权重并预热 GPU；此过程可能耗时数分钟。',flush=True)
    def stop(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
    runtime=Path(os.environ.get('PI05_CONSOLE_RUNTIME',Path(__file__).resolve().parents[1]/'runtime'))
    pins=runtime/'pinned'; pins.mkdir(parents=True,exist_ok=True)
    # Release the temporary hardlink on normal exit, errors, or cancellation.
    with tempfile.TemporaryDirectory(prefix='eval-',dir=pins) as pin_dir:
        prepare_checkpoint(config,args.train_python,pin_dir)
        resolved=path.with_name('console_eval_resolved.json'); resolved.write_text(json.dumps(config,ensure_ascii=False,indent=2))
        from loongforge.embodied.eval.payload_builders.pi05 import Pi05PayloadBuilder
        previous=Pi05PayloadBuilder._encode_state
        def encode(self,canonical):
            return encode_libero_state(canonical) if self.state_encoding=='lib_eef_axis_angle' else previous(self,canonical)
        Pi05PayloadBuilder._encode_state=encode
        print('[PI05] q99 state prompt enabled: position(3)+axis-angle(3)+finger positions(2)',flush=True)
        # The native orchestrator owns its policy-server group and cleans it in finally.
        sys.argv=['run','--config',str(resolved)]
        finished=threading.Event()
        def startup_progress():
            started=time.monotonic()
            opener=build_opener(ProxyHandler({}))
            url=f"http://127.0.0.1:{config['server']['health_port']}/healthz"
            while not finished.wait(30):
                try:
                    with opener.open(url,timeout=2) as response:
                        ready=json.load(response).get('ready')
                    if ready:
                        print('[startup] 策略服务就绪，开始 LIBERO 评测。',flush=True)
                        return
                except Exception:
                    pass
                print(f'[startup] 等待策略服务就绪：{int(time.monotonic()-started)} 秒。模型构建、权重加载及预热可能耗时数分钟；其他模型的可选依赖警告不表示 PI05 失败。',flush=True)
        threading.Thread(target=startup_progress,daemon=True).start()
        try:
            runpy.run_module('loongforge.embodied.eval.orchestrator.run',run_name='__main__')
        finally:
            finished.set()

if __name__=='__main__': main()
