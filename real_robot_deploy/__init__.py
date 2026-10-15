"""边端 policy 执行端（真机闭环回放）—— 与 real_robot_collect 同级的独立模块。

- ``policy_client``：极简 WebSocket 客户端（协议对齐 LoongForge eval/transport/client.py）
- ``run_policy``：主循环（影子模式 / 驱动模式）

设计原则：**执行端只做"硬件 ↔ 协议"的胶水**，不重复实现 RPC；图像/state/task
一律复用 real_robot_collect 的同一份实现，确保与训练分布严格一致。
"""

from .policy_client import PolicyClient, as_chunk

__all__ = ["PolicyClient", "as_chunk"]
