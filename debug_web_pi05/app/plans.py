"""PI05-native validated command plans. Previews never create output files."""
from __future__ import annotations
import json, math, os, re, shlex, socket
from pathlib import Path
from .settings import settings, WORKERS_DIR
from .catalog import read_json, weight_layout, permitted

STRATEGIES={'ddp':'run_pi05_ddp_finetune.sh','zero1':'run_pi05_ddp_zero1_finetune.sh','fsdp':'run_pi05_fsdp_finetune.sh'}

def integer(params,key,default,low,high):
    raw=params.get(key,default)
    if isinstance(raw,bool): raise ValueError(f'{key} 必须是整数')
    try: value=int(str(raw))
    except ValueError: raise ValueError(f'{key} 必须是整数') from None
    if not low<=value<=high: raise ValueError(f'{key} 范围为 {low}～{high}')
    return value

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); return s.getsockname()[1]

def build_plan(kind, p):
    if kind not in ('train','eval'): raise ValueError('只支持 PI05 训练或 LIBERO 评测')
    if not isinstance(p,dict): raise ValueError('参数必须是对象')
    checks=[]
    def check(name, ok, detail, warning=False):
        checks.append({'name':name,'level':'ok' if ok else 'warn' if warning else 'error','detail':detail})
    repo=Path(settings.get('repo_root'))
    python=Path(settings.get('python'))
    checkpoint=Path(str(p.get('checkpoint') or ''))
    dataset=Path(str(p.get('dataset') or ''))
    tokenizer=Path(str(p.get('tokenizer') or settings.get('tokenizer')))
    gpus=str(p.get('gpus',settings.get('default_gpus'))).strip()
    if not re.fullmatch(r'\d+(,\d+)*',gpus) or len(set(gpus.split(',')))!=len(gpus.split(',')):
        raise ValueError('GPU 请填不重复的卡号，例如 3 或 0,1')
    if kind=='eval' and ',' in gpus: raise ValueError('单次 LIBERO 评测请选择一张 GPU')
    check('LoongForge 入口',(repo/'loongforge/embodied/train.py').is_file(),str(repo))
    check('训练/策略解释器',python.is_file() and os.access(python,os.X_OK),str(python))
    check('Tokenizer',(tokenizer/'tokenizer.json').is_file() or (tokenizer/'tokenizer.model').is_file(),str(tokenizer))
    info=read_json(dataset/'meta/info.json',{}) or {}
    features=info.get('features',{})
    action_dim=(features.get('action',{}).get('shape') or [0])[0]
    state_dim=(features.get('observation.state',{}).get('shape') or [0])[0]
    cameras=[k for k in features if k.startswith('observation.images.')]
    check('LeRobot 数据集',bool(info) and info.get('codebase_version')=='v3.0',f'{dataset} · 格式 {info.get("codebase_version","未知")}')
    check('动作与状态维度',isinstance(action_dim,int) and isinstance(state_dim,int) and 0<action_dim<=32 and 0<state_dim<=32,
          f'action={action_dim} / state={state_dim}，最大填充维度32')
    check('相机',len(cameras)==2,f'{len(cameras)} 路：'+', '.join(cameras)+'；当前配方使用双相机')
    stats=read_json(dataset/'meta/stats.json',{}) or {}
    stats_valid=True
    for feature,width in [('action',action_dim),('observation.state',state_dim)]:
        item=stats.get(feature,{})
        low=item.get('q01',[]); high=item.get('q99',[])
        try:
            valid=isinstance(low,list) and isinstance(high,list) and len(low)==len(high)==width and width>0
            valid=valid and all(math.isfinite(float(a)) and math.isfinite(float(b)) and float(b)>=float(a) for a,b in zip(low,high))
        except (ValueError,TypeError): valid=False
        stats_valid &= valid
    check('PI05 q99 统计',stats_valid,'动作反归一化和 state 离散化均使用数据集 q01/q99')
    model_cfg=read_json(checkpoint/'config.json',{}) or {}
    check('PI05 模型来源',model_cfg.get('type','pi05')=='pi05' and ('pi05' in str(checkpoint).lower() or model_cfg.get('type')=='pi05'), str(checkpoint))
    layout={}
    try:
        layout=weight_layout(checkpoint/'model.safetensors'); check('权重文件',True,f"{layout['tensors']} tensors · {layout['layout']}（仅检查文件头）")
    except (OSError,ValueError) as exc: check('权重文件',False,str(exc))
    if kind=='eval' and layout.get('layout')=='wrapped':
        check('评测权重适配',False,'原生训练权重存在包装前缀：启动时转换到独立缓存，保留原权重',True)
    if kind=='train' and layout.get('layout')=='wrapped' and not p.get('resume',False):
        check('训练权重加载',False,'包装 checkpoint 建议通过原输出目录续训；新微调请选原始 PI05 权重',True)
    resume=p.get('resume',False)
    if not isinstance(resume,bool): raise ValueError('resume 必须是布尔值')
    output=Path(str(p.get('output') or '')).expanduser()
    if not p.get('output') or not output.is_absolute(): raise ValueError('请填写输出目录的绝对路径')
    root=Path(settings.get('output_root')).resolve(); output=output.resolve()
    if root not in output.parents: raise ValueError('输出目录必须位于配置的 outputs 根目录下')
    if kind=='train' and resume:
        candidates=[s for s in (output/'checkpoints').glob('steps_*') if s.is_dir() and s.name[6:].isdigit()]
        latest=max(candidates,key=lambda s:int(s.name[6:])) if candidates else None
        meta=read_json(latest/'resume_meta.json',{}) if latest else {}
        valid_meta=isinstance(meta,dict) and type(meta.get('completed_steps')) is int
        has_state=bool(latest and any((latest/name).is_file() for name in ('training_state.pt','zero_optimizer/metadata.json','dcp/.metadata')))
        check('续训状态',has_state and valid_meta,f'{latest or output}: 最新 steps_N 必须含训练状态与 resume_meta.json，不回退到旧 checkpoint')
        if valid_meta:
            completed=meta['completed_steps']
            check('续训目标步数',integer(p,'steps',1000,1,10000000)>completed,f'已完成 {completed} 步；目标必须更大')
    else:
        check('独立输出目录',not output.exists() or (output.is_dir() and not any(output.iterdir())),str(output)+'（不覆盖已有结果）')
    env={'MODEL_NAME':'pi05','CUDA_VISIBLE_DEVICES':gpus,'PYTHONPATH':str(repo),'PATH':str(python.parent)+os.pathsep+os.environ.get('PATH',''),
         'LOONGFORGE_PATH':str(repo),'TOKENIZER_PATH':str(tokenizer),'NO_PROXY':'127.0.0.1,localhost','no_proxy':'127.0.0.1,localhost'}
    horizon=integer(p,'horizon',50,1,100)
    seed=integer(p,'seed',42,0,2147483647)
    eval_config=None
    if kind=='train':
        strategy=p.get('strategy','ddp')
        if strategy not in STRATEGIES: raise ValueError('未知训练策略')
        script=repo/'examples/embodied/pi05'/STRATEGIES[strategy]
        check('训练配方',script.is_file(),f'原生 {strategy.upper()} 脚本')
        steps=integer(p,'steps',1000,1,10000000); batch=integer(p,'batch',4,1,1024)
        accum=integer(p,'accum',1,1,1024); workers=integer(p,'workers',4,0,128)
        save=integer(p,'save_interval',100,0,10000000); warmup=integer(p,'warmup',10,0,10000000)
        if warmup>=steps: raise ValueError('warmup 必须小于总训练步数')
        lr=float(p.get('lr',2.5e-5))
        if not math.isfinite(lr) or not 0<lr<=1: raise ValueError('学习率须在0～1之间')
        for key in ('compile','freeze_vision','expert_only','gradient_checkpointing'):
            if key in p and not isinstance(p[key],bool): raise ValueError(f'{key} 必须为布尔值')
        env.update({'GPUS_PER_NODE':str(len(gpus.split(','))),'NNODES':'1','NODE_RANK':'0','MASTER_ADDR':'127.0.0.1','MASTER_PORT':str(free_port()),
            'CHECKPOINT_PATH':str(checkpoint),'DATA_PATH':str(dataset),'OUTPUT_DIR':str(output),'TENSORBOARD_DIR':str(output/'tensorboard'),
            'TRAIN_ITERS':str(steps),'PER_DEVICE_BATCH_SIZE':str(batch),'GRADIENT_ACCUMULATION_STEPS':str(accum),
            'NUM_WORKERS':str(workers),'SAVE_INTERVAL':str(save),'SEED':str(seed)})
        argv=['bash',str(script),'--lr-base',str(lr),'--lr-warmup-iters',str(warmup),'--min-lr',str(min(lr/10,1e-6)),
              f'model.action_dim={action_dim}',f'model.state_dim={state_dim}',f'model.action_horizon={horizon}',
              f'model.compile_model={str(p.get("compile",False)).lower()}',
              f'model.freeze_vision_encoder={str(p.get("freeze_vision",False)).lower()}',
              f'model.train_expert_only={str(p.get("expert_only",False)).lower()}',
              f'model.gradient_checkpointing={str(p.get("gradient_checkpointing",True)).lower()}']
        if resume: argv.append('--resume')
        check('有效 batch',True,f'{batch} × {len(gpus.split(","))} 卡 × {accum} 累积 = {batch*len(gpus.split(","))*accum}')
    else:
        if 'save_video' in p and not isinstance(p['save_video'],bool): raise ValueError('save_video 必须为布尔值')
        bench=Path(settings.get('benchmark_python'))
        check('LIBERO 解释器',bench.is_file() and os.access(bench,os.X_OK),str(bench))
        check('LIBERO 环境配置',(Path(settings.get('libero_config'))/'config.yaml').is_file(),settings.get('libero_config'))
        check('LIBERO 数据语义',action_dim==7 and state_dim==8 and 'libero' in dataset.name.lower(),
              '当前支持 LIBERO：7维末端动作、8维状态（位置+轴角+双指位置）')
        episodes=integer(p,'episodes',1,1,50); tasks=integer(p,'tasks',1,1,10)
        max_steps=integer(p,'max_steps',520,1,2000); chunk=integer(p,'execute_steps',10,1,horizon)
        port=free_port(); health=free_port()
        while health==port: health=free_port()
        eval_config={'benchmark':{'name':'libero','suite':'libero_10','max_tasks':tasks,'episodes_per_task':episodes,'max_steps':max_steps,
                'num_steps_wait':10,'continuous_gripper':True,'prefer_controller_ee':False},
            'model':{'backend':'loongforge','model_type':'pi05','name':'pi05-console','action_dim':7,'state_dim':8,
                'action_horizon':horizon,'max_action_dim':32,'max_state_dim':32,'compile_model':False,'state_encoding':'lib_eef_axis_angle'},
            'server':{'host':'127.0.0.1','port':port,'health_port':health,'python':str(python),'log':str(output/'policy_server.log'),
                'start_timeout_sec':1800,'ckpt_path':str(checkpoint),'dataset_statistics_path':str(dataset/'meta/stats.json'),
                'tokenizer_path':str(tokenizer),'use_bf16':False,'chunk_execute_steps':chunk},
            'env':{'eval_root':str(repo/'loongforge/embodied/eval'),'loongforge_root':str(repo),'libero_config_path':settings.get('libero_config'),
                'mujoco_gl':'egl','pyopengl_platform':'egl'},
            'run':{'output_dir':str(output),'timestamped_output':False,'seed':seed,'save_trace':False,'save_replay':bool(p.get('save_video',True))},
            'timeouts':{'policy_call_ms':600000,'per_step_sec':600,'per_episode_sec':1800}}
        env.update({'PYTHONPATH':str(repo)+os.pathsep+settings.get('benchmark_packages'),'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl','LIBERO_CONFIG_PATH':settings.get('libero_config')})
        argv=[str(bench),str(WORKERS_DIR/'eval_entry.py'),'--config',str(output/'console_eval.json'),'--train-python',str(python)]
        check('PI05 推理口径',True,f'q99 反归一化 / state 注入语言提示 / FP32 / 执行 {chunk} 帧后重规划')
        check('评测规模',True,f'{tasks}任务 × {episodes}次 = {tasks*episodes} episodes；当前成功率只统计已完成样本')
    return {'ok':not any(c['level']=='error' for c in checks),'checks':checks,'kind':kind,'argv':argv,'env':env,
        'cwd':str(repo),'output':str(output),'config':eval_config,'params':p,'cmdline':shlex.join(argv),
        'notes':'静态检查不验证显存是否足够或所有第三方库；启动后查看实时日志。'}
