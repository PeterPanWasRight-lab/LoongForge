"""复位预演 —— 只驱动一次 ``reset_to_position``，不采集、不启相机。

**为什么需要它**：``reset_to_position`` 会把目标位姿 **clip 到 workspace_limits**，
然后用 ``interpolate_move(..., steps=50, delay=0.02)`` 在 **1 秒内**插值过去。
所以只要 ``reset_position`` 或 ``workspace_limits`` 与现场不符，
一条 episode 还没开始，机械臂就可能冲出一大段（2026-10-13 实测过一次：
旧 limits 会让复位位移达 187 mm）。

换工位 / 改 ``reset_position`` / 改 ``workspace_limits`` 之后，
**第一次真正驱动机械臂前先跑这个**，确认：
    ① 复位目标 6 维全在边界内（不会被 clip ⇒ 不会产生额外位移）
    ② 复位前后实测位姿几乎不变（< 5 mm）

用法::

    python -m real_robot_collect.reset_preview --config real_robot_collect/configs/peg_v1.json
    python -m real_robot_collect.reset_preview --config ... --no-reset    # 只做边界检查，不驱动

退出码：0 = 通过；2 = 目标越界（**不要此时启动采集**）；3 = 复位后位移过大。
"""

from __future__ import annotations

import argparse
import time

#: 与 workspace_limits 的键顺序一致（x, y, z, rx, ry, rz）
_LIMIT_KEYS = [
    ("x_min", "x_max"),
    ("y_min", "y_max"),
    ("z_min", "z_max"),
    ("rx_min", "rx_max"),
    ("ry_min", "ry_max"),
    ("rz_min", "rz_max"),
]


def check_limits(cfg) -> list[int]:
    """返回越界的维度下标（空 = 全部通过）。"""
    L = cfg.robot.workspace_limits
    pos = cfg.record.reset_position
    bad: list[int] = []
    print("复位目标 :", [round(float(v), 3) for v in pos])
    for i, (klo, khi) in enumerate(_LIMIT_KEYS):
        lo, hi = float(L[klo]), float(L[khi])
        ok = lo <= float(pos[i]) <= hi
        if not ok:
            bad.append(i)
        flag = "OK" if ok else "OUT OF RANGE -> 会被 clip！"
        print(f"  dim{i}  {float(pos[i]):>10.3f}  in [{lo:>8.1f}, {hi:>8.1f}]   {flag}")
    return bad


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="复位预演（只调用一次 reset_to_position）")
    p.add_argument("--config", required=True, help="采集配置文件")
    p.add_argument("--countdown", type=float, default=5.0, help="执行前倒计时秒数（默认 5）")
    p.add_argument("--no-reset", action="store_true", help="只做边界检查，不连硬件、不驱动")
    args = p.parse_args(argv)

    from .config import load_config
    from .hw import Hardware

    cfg = load_config(args.config)

    print("=" * 66)
    print("① 边界检查（复位目标是否会被 clip）")
    print("=" * 66)
    bad = check_limits(cfg)
    if bad:
        print(f"\n❌ dim {bad} 越界 ⇒ 复位目标会被 workspace_limits 改写，")
        print("   机械臂会在 1 秒内插值移动到边界值。**此时不要启动采集。**")
        print("   修法：把 workspace_limits 放宽到包含 reset_position，")
        print("         或把 reset_position 改成实测位姿（见 --no-reset 打印值）。")
        return 2
    print("\n✅ 6 维全在边界内 ⇒ 复位目标不会被改写")

    rgr = float(cfg.record.reset_gripper)
    print(f"\n夹爪目标 reset_gripper = {rgr}"
          f"  ->  {'闭合 / 保持夹紧' if rgr > 0.5 else '⚠️ 张开（夹持中的工件会掉落）'}")

    if args.no_reset:
        print("\n（--no-reset：只检查，不驱动硬件）")
        # 仍然读一次实测位姿，方便和 reset_position 对照
        hw = Hardware(cfg)
        hw.connect(connect_teleop=False)
        try:
            obs = hw.read_raw_obs()
            cur = [round(float(obs[f"loc_{i}.pos"]), 3) for i in range(6)]
            print("\n当前实测位姿 :", cur)
            print("复位目标     :", [round(float(v), 3) for v in cfg.record.reset_position])
            d = [round(float(a) - float(b), 3)
                 for a, b in zip(cur, cfg.record.reset_position)]
            print("差值(实测-目标):", d)
        finally:
            hw.close()
        return 0

    hw = Hardware(cfg)
    hw.connect(connect_teleop=False)
    try:
        obs = hw.read_raw_obs()
        before = [float(obs[f"loc_{i}.pos"]) for i in range(6)]
        print("\n复位前实测 :", [round(v, 3) for v in before])
        print(f"\n{args.countdown:.0f} 秒后执行复位 —— 请盯住机械臂，随时准备急停 / Ctrl+C")
        for k in range(int(args.countdown), 0, -1):
            print(f"  {k}...", flush=True)
            time.sleep(1.0)

        hw.reset_to_start()
        time.sleep(2.0)

        obs2 = hw.read_raw_obs()
        after = [float(obs2[f"loc_{i}.pos"]) for i in range(6)]
        print("\n复位后实测 :", [round(v, 3) for v in after])
        delta = [a - b for a, b in zip(after, before)]
        print("位移        :", [round(v, 3) for v in delta])
        mx = max(abs(v) for v in delta[:3])
        print(f"\n最大位置位移 = {mx:.1f} mm")
        if mx < 5.0:
            print("✅ 几乎没动 —— 复位点是安全的，可以启动采集")
            return 0
        print("⚠️ 动了 —— 检查 reset_position / workspace_limits 后再采集")
        return 3
    finally:
        hw.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
