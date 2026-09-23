"""PI05 console settings, independent of the PI0 console."""
from __future__ import annotations
import copy
import json
import os
import threading
from pathlib import Path

DEBUG_WEB_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEBUG_WEB_DIR.parent
WEB_DIR = DEBUG_WEB_DIR / 'web'
WORKERS_DIR = DEBUG_WEB_DIR / 'workers'
RUNTIME_DIR = Path(os.environ.get('PI05_CONSOLE_RUNTIME', str(DEBUG_WEB_DIR / 'runtime')))
JOBS_DIR = RUNTIME_DIR / 'jobs'
JOBS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULTS = {
    'repo_root': '/home/gm/data/LoongForge/LoongForge',
    'python': '/home/gm/data/LoongForge/env/bin/python',
    'benchmark_python': '/home/gm/.conda/envs/wh/bin/python',
    'benchmark_packages': '/home/gm/data/LoongForge/bench_env/site-packages',
    'artifacts_root': '/home/gm/data/LoongForge/artifacts',
    'output_root': '/home/gm/data/LoongForge/outputs',
    'tokenizer': '/home/gm/data/LoongForge/artifacts/paligemma-3b-pt-224',
    'libero_config': '/home/gm/.libero',
    'default_gpus': '3', 'max_log_lines': 20000,
}

class Settings:
    def __init__(self):
        self.path = DEBUG_WEB_DIR / 'config.json'
        self._lock = threading.RLock()
        self._data = dict(DEFAULTS)
        if self.path.is_file(): self._data.update(json.loads(self.path.read_text()))
    def all(self):
        with self._lock: return copy.deepcopy(self._data)
    def get(self, key, default=None):
        with self._lock: return copy.deepcopy(self._data.get(key, default))
    def update(self, patch):
        if not isinstance(patch, dict) or set(patch) - set(DEFAULTS): raise ValueError('未知配置项')
        with self._lock:
            candidate = {**self._data, **patch}
            for key in DEFAULTS:
                if key == 'max_log_lines':
                    if type(candidate[key]) is not int or not 100 <= candidate[key] <= 100000: raise ValueError('日志缓存应在100～100000之间')
                elif not isinstance(candidate[key], str) or not candidate[key].strip(): raise ValueError(f'{key} 不能为空')
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(candidate, ensure_ascii=False, indent=2))
            tmp.replace(self.path); self._data = candidate
            return self.all()
settings = Settings()
def python_executable(): return settings.get('python')
def subprocess_env(extra=None):
    env = dict(os.environ)
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','CONFIG'): env.pop(key, None)
    env.update({'PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','NO_ALBUMENTATIONS_UPDATE':'1'})
    env.update({str(k):str(v) for k,v in (extra or {}).items()})
    return env
