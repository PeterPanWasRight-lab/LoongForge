# 复现 Pipeline II：LoRA 微调 → LIBERO-10 80.0%

> 目标模型：`pi05_lora_eval/merged_10000`（merged 裸键，80.0%）  
> 对照：全量组 90.0%（同起点/数据/预算/lr，唯一变量 = 更新方式）  
> 前置：阶段 2 的起点 `long/ckpt_6000`（813 裸键）——见 Pipeline I

## 流程总览

```
ckpt_6000(813裸键)
   │  ① 训练 10000 步（LoRA shim 注入 peft，adapter-only 存档）
   ▼
pi05_lora_long/S3/checkpoints/steps_{2000..10000}   ← 每档 0.7 G
   │  ② 纯张量 merge（W += α/r·BA，秒级）
   ▼
pi05_lora_eval/merged_10000/model.safetensors        ← 813 裸键
   │  ③ 评测（exec=15，100 条，固定噪声）
   ▼
80.0%（全量同路径 = 90.0%）
```

## ① 训练（~29 h，GPU2 实测）

```bash
ROOT=/home/gm/data/LoongForge
cd $ROOT/outputs

# ── 前置：LoRA shim 必须在位（PYTHONPATH 注入，零改仓库）──
# outputs/pi05_lora_shim/sitecustomize.py：
#   · patch _build_optimizer → 包 peft.get_peft_model(rank32/α32/dropout0.05,
#     target=q/k/v/o+gate/up/down)   ⚠️ 必须在 optimizer 创建之前
#   · patch _save_checkpoint → adapter-only 保存（不 patch 首个保存间隔即崩）
ls $ROOT/outputs/pi05_lora_shim/sitecustomize.py || echo "缺 shim，先部署"

# ── probe（20 步，必做）：三项验收 ──
MODE=probe GPU=<卡> PORT=29620 bash run_lora.sh
# 验收 1：日志 trainable=59.06M / 4202.46M (1.41%)
# 验收 2：loss 0.08–0.15（与起点收敛值同量级 ⇒ 权重真加载；随机初始化会大一个量级）
# 验收 3：显存峰值 < 卡容量 − 同卡他人占用

# ── 保存路径快验（22 步强制触发存档；probe 本身不触发保存）──
MODE=probe TRAIN_ITERS=22 SAVE_INTERVAL=20 GPU=<卡> PORT=29621 bash run_lora.sh
# 验收：checkpoints/steps_20/ 出现 adapter_model.safetensors(~472M) + adapter_state.pt(~236M)

# ── 正式训练 ──
MODE=full GPU=<卡> PORT=29620 setsid nohup bash run_lora.sh \
    > $ROOT/outputs/lora_full.out 2>&1 &
# 脚本内置关键参数（对齐全量组 lr 轨迹）：
#   --train-iters 10000 --lr-base 1.76e-5 --min-lr 1e-6 --lr-warmup-iters 10
#   --no-save-training-state   bs=32  seed=42
# 健康判据：step2 lr=3.52e-6(=1.76e-5×2/10)  loss 0.135→0.05  0 nan  步时 ~10.2s
```

**已知不可对齐项（写入结论时必须声明）**：全量组恢复了 ckpt_6000 的 Adam 动量，LoRA 参数从零动量开始。

## ② Merge（秒级，纯张量）

```bash
E=$ROOT/outputs/pi05_lora_eval
python $E/lora_merge.py \
    $ROOT/outputs/pi05_s3ext10k_long/S3/ckpt_6000 \      # base 裸键
    $ROOT/outputs/pi05_lora_long/S3/checkpoints/steps_10000 \  # adapter
    $E/merged_10000
# 数学：merged[W] = base[W] + (α/r)·B·A = base[W] + 1.0·B·A
```

**验收判据（三条全过才算成功）**：

| 判据       | 期望                                                                        |
| -------- | ------------------------------------------------------------------------- |
| 去重后待合并模块 | **333**（adapter 键带 `base_model.model.module.{model,pi05}.` 双前缀别名，脚本已自动去重） |
| 变化张量数    | **328**（202 paligemma + 126 gemma_expert，全部是目标 proj 权重）                   |
| 非目标张量    | 485 个（bias/layernorm/action 投影头）**逐位不变**                                  |

## ③ 评测（~60-75 min，与全量同口径）

```bash
R=$ROOT/outputs/pi05_exec_sweep
# manifest + yaml（模板 sed 派生，ckpt 指向 merged 目录）
printf "lora_exec15\t50\t15\t15900\t$ROOT/outputs/pi05_lora_eval/merged_10000\n" >> $R/eval/manifest.tsv
sed -e "s/e50_16000_exec15/lora_exec15/g" -e "s/15860/15900/" -e "s/15861/15901/" \
    -e "s#^  ckpt_path: .*#  ckpt_path: $ROOT/outputs/pi05_lora_eval/merged_10000#" \
    $R/eval/eval_e50_16000_exec15.yaml > $R/eval/eval_lora_exec15.yaml
cd $R && bash run_exec_sweep.sh lora_exec15 <GPU>
# 验收：shim 日志两进程 INSTALLED(fixed_noise=True execute_steps=15)
#       stats = libero_10（与训练数据集绑定）
#       results.jsonl 满 100 条才算数；配对 McNemar vs 全量组（本地 sweep_results/）
```

## 🚨 踩坑清单（全部实测）

| # | 坑                          | 后果                  | 对策                               |
| - | -------------------------- | ------------------- | -------------------------------- |
| 1 | LoRA 注入晚于 optimizer 创建     | optimizer 拿旧参数，训练白跑 | patch `_build_optimizer` 钩子      |
| 2 | 不 patch `_save_checkpoint` | 首个保存间隔崩溃            | shim 内 adapter-only 保存           |
| 3 | adapter 双前缀别名直接 merge      | 0 模块匹配，静默出全 base 权重 | 剥 `module.{model,pi05}.` 前缀 + 去重 |
| 4 | probe 未触发保存就上正式跑           | 2 h 后首存档时崩          | 先 22 步 SAVE_INTERVAL=20 快验       |
| 5 | 起点用非裸键                     | 加载器判 unexpected key | `convert_bare_ckpt.py` 先转        |
| 6 | 磁盘 <10 G 时跑评测              | 裸键 8.7 G 写入失败       | 保持 ≥30 G 余量                      |

## 产物与对照

| 产物            | 路径                                                                                   |
| ------------- | ------------------------------------------------------------------------------------ |
| 5 个 adapter 档 | `pi05_lora_long/S3/checkpoints/steps_{2000..10000}`（各 677 M）                         |
| **80% 模型**    | `pi05_lora_eval/merged_10000/model.safetensors`（813 裸键，8.8 G）                        |
| 评测结果          | `pi05_exec_sweep/20260929_192942_lora_exec15/`（本地 `sweep_results/lora_exec15.jsonl`） |
| 对照 90%        | `pi05_s3ext10k_long/S3/ckpt_16000`（见 Pipeline I）                                     |
