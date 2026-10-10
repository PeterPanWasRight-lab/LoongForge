# 复现 Pipeline I：全量微调 → LIBERO-10 90.0%

> 目标模型：`pi05_s3ext10k_long/S3/ckpt_16000` ｜ 服务器：A6000Server（gm@172.16.107.56）
> 环境：`/home/gm/data/LoongForge`（repo HEAD `5364d05`，python=`$ROOT/env/bin/python`）
> 详细超参见《pi05_最优实验配置与复现手册_2026-09-27.md》；本文只给可执行命令序列。

## ⚠️ 先读：与原始实验的一个不可复现点

原始实验从 `steps_6000`（含 `training_state.pt`）做**真 resume**（Adam 动量/scheduler/RNG 全恢复）。
该文件已于 09-26 被共享账号第三方删除 ⇒ **位级复现不可能**。两条替代路径：

| 路径 | 说明 | 预期 |
|---|---|---|
| **A. 从头完整复现** | 重跑混训线（pi05_base → 6000 步）再专攻续训 | 最忠实，共 ~2.5 天 |
| **B. 裸键热启动** | 从现存 `ckpt_6000` 裸键起新训（optimizer 从零） | 快，但非位级一致；参考 LoRA 组同款启动方式（去掉 LoRA shim 即可），预期成功率接近 90% 但不保证逐位相同 |

以下命令序列按 **路径 A（完整版）** 写；路径 B = 跳过阶段 1、阶段 2 改用热启动变体（文中标注）。

---

## 阶段 0：环境自检

```bash
ROOT=/home/gm/data/LoongForge
# 数据集与底座
ls $ROOT/artifacts/libero_10/meta/stats.json        # libero_10: 379 ep / 101469 帧
ls $ROOT/artifacts/libero_full/meta/stats.json      # libero_full: 1693 ep / 273465 帧
ls $ROOT/artifacts/pi05_base/model.safetensors      # 底座 14G
ls $ROOT/artifacts/paligemma-3b-pt-224/             # tokenizer
# GPU：选 free > 40G 的卡；GPU2 有他人常驻 10.7G
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
```

## 阶段 1：混训预微调（pi05_base → 6000 步，0.9 epoch）

```bash
cd $ROOT/outputs
# 参照 run_s3_bigbatch.sh 的环境变量集（repo examples 脚本 + 同款 env）：
#   CHECKPOINT_PATH=$ROOT/artifacts/pi05_base  DATA_PATH=$ROOT/artifacts/libero_full
#   TRAIN_ITERS=6000  PER_DEVICE_BATCH_SIZE=32  SEED=42  SAVE_INTERVAL=2000
#   输出 → outputs/pi05_s3_bigbatch/S3
LINE=bigbatch GPU=<空卡> PORT=29560 bash run_s3_bigbatch.sh   # ~14 h
# 验收：metrics.jsonl 末行 step=6000，0 nan / 0 skipped，loss ≈ 0.133
```

> 阶段 1 产物成功率约 30%（libero_10 口径）。它只是"爬回起点"，单看无意义。

## 阶段 2：专攻续训（6000 → 16000，+3.1 epoch）—— 核心步骤

```bash
cd $ROOT/outputs
# 脚本内已内置：换数据集续训必须跳过 dataloader 位置恢复
#   （否则 torchdata StatefulSampler 快进越界 → StopIteration，崩在 optimizer 恢复成功之后）
LINE=long GPU=<空卡> PORT=29562 setsid nohup bash run_s3ext10k.sh \
    > $ROOT/outputs/s3ext10k_long.out 2>&1 &
# ~23 h；每 2000 步存档（18.7 G model + 15.1 G training_state）
# 监督器 watch_s3ext10k.sh：保最新 2 档 training_state，防磁盘爆
```

**健康判据**（启动后 20 min 内检查）：

| 判据 | 期望值 |
|---|---|
| `learning rate` @ step 6001 | **1.760363e-05**（热重启，逐位可对） |
| action_loss | 0.10–0.16 |
| step_time | ~8.3 s ⇒ ETA 23 h |
| nan / skipped | 0 |

```bash
# 逐档评测（exec=50 口径，学习曲线 30→47→49→65→69）
E=$ROOT/outputs/pi05_s3ext10k_eval
python $E/make_eval_configs.py --line long --steps 8000 10000 12000 16000 \
    --out-dir $E/eval --base-port 15700 --step-port 10
# 每个 yaml 的 dataset_statistics_path 必须是 libero_10（🚨 见防呆说明）
```

## 阶段 3：执行长度调优（50 → 15，纯推理，0 训练成本）

```bash
R=$ROOT/outputs/pi05_exec_sweep
# 1) manifest 加一行（格式：组名 TAB horizon TAB exec_steps TAB 端口 TAB ckpt目录）
printf "my_exec15\t50\t15\t15900\t$ROOT/outputs/pi05_s3ext10k_long/S3/ckpt_16000\n" >> $R/eval/manifest.tsv
# 2) 由 exec15 模板派生 yaml（改 ckpt 路径/组名/端口）
sed -e "s/e50_16000_exec15/my_exec15/g" -e "s/15860/15900/" -e "s/15861/15901/" \
    -e "s#^  ckpt_path: .*#  ckpt_path: $ROOT/outputs/pi05_s3ext10k_long/S3/ckpt_16000#" \
    $R/eval/eval_e50_16000_exec15.yaml > $R/eval/eval_my_exec15.yaml
# 3) 跑（裸键已存在则自动跳过转换）
cd $R && bash run_exec_sweep.sh my_exec15 <GPU>   # ~60-75 min
```

**验收判据（必须过，"跑起来"不算过）**：

```bash
# trace 中 chunk_cache_hit=False 次数 ≈ 真实推理次数：exec=15 ⇒ 520 步 ≈ 35 次
python $R/eval/verify_exec_len.py <run_dir> 50 15
# 成功率期望 ≈ 90.0%；若 15xxx 端口被占必须换（撞端口 = 静默复用他人 server，出假数）
```

## 🚨 三条铁律（违反即产废数据）

1. **stats 绑定训练数据集**：libero_10 训练的 ckpt 配 libero_full 的 stats ⇒ 不报错、动作被压 2.55 倍、成功率骤降（实测 49% 被测成 11%）
2. **起评测前查端口**：`ss -ltn | grep :159xx`——编排器对已占用端口**静默复用**对方的 policy server
3. **中途百分比不可外推**：评测按 task-major 排序，前 N 条全是易任务（曾据 70 条子集外推 95.7%，满 100 条落回 89%）
