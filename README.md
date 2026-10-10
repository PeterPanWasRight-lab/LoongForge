# LoongForge · pi05 后训练与评测

本仓库已收敛为 **π0.5 (pi05) 后训练 / 微调 / LIBERO 评测** 单一主线（其余 LLM/VLM 组件已于 2026-10 移除，见 git 历史）。

## 目录结构

| 路径 | 内容 |
|---|---|
| `loongforge/embodied/` | pi05 模型、训练器（DDP/FSDP finetune）、评测编排（LIBERO runner / policy server） |
| `configs/` | pi05 训练/评测配置 |
| `examples/embodied/pi05/` | 启动脚本（`run_pi05_ddp_finetune.sh` 等） |
| `debug_web_pi05/` | pi05 调试台（web server，8901 端口） |
| `tests/embodied/` | pi05 相关基线与测试 |
| `outputsDoc/` | 实验报告与复现 pipeline（全量 90% / LoRA 80% / 执行长度消融） |

## 关键结论速览

- 执行长度（re-planning 频率）是首要瓶颈：exec 50→15 = 69%→90%（+21 pt, p=0.0003），单峰曲线
- libero_10 专攻 ≫ libero_full 混训（69% vs 44%，p=1e-4）
- LoRA(rank32) 拿到 80.0% vs 全量 90.0%（1.41% 参数，p=0.0414）
- 评测必须：固定噪声 shim + stats 绑定训练数据集 + 100 条档配对 McNemar

详见 `outputsDoc/` 下的报告与复现 pipeline。
