"""使能与运动通道探测 —— 独立验证「机械臂到底能不能动」。

背景（2026-10-13 实测）
----------------------
``hs_robot.py:241`` 把 ``self.hsrobot.enable()`` **注释掉了**，只留了一句
``logger.info("Robot enabled successfully")``（假的）。于是 connect 之后
机械臂处于**未使能**状态：

* 读位姿 ``get_cartesian_position()`` 一切正常 —— 完全看不出问题
* 但 ``move_to_cartesian_position()`` / ``send_action()`` **全部无效且不报错**
* 实测症状：复位流程（含"先垂直往上走 20mm"）零位移、推 SpaceMouse 零位移，
  而 ``_cached_target_position`` 却在正常累积 ⇒ 指令发出了，只是没人执行。

本模块的 ``hw.connect()`` 现在会自动补上 ``enable()``（由
``robot.enable_on_connect`` 控制，默认开）。这个脚本用来**独立验证**这一点。

用法::

    # ① 只看：connect + 使能 + 读位姿（机械臂不会动）
    python -m real_robot_collect.enable_probe --config real_robot_collect/configs/peg_v1.json

    # ② 运动测试：沿 +Z 走 5mm 再回原位（⚠️ 机械臂会真的动）
    python -m real_robot_collect.enable_probe --config ... --move-test

    # ③ 对照实验：不使能就做运动测试（应当"不动"，证明因果）
    python -m real_robot_collect.enable_probe --config ... --move-test --no-enable

退出码：0 = 运动通道可用；3 = 使能后仍然不动（去查示教器/急停/伺服）。
"""

from __future__ import annotations

import argparse
import time


def _read_pose(hw) -> list[float]:
    obs = hw.read_raw_obs()
    return [float(obs[f"loc_{i}.pos"]) for i in range(6)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="使能 + 运动通道探测")
    p.add_argument("--config", required=True, help="采集配置文件")
    p.add_argument("--move-test", action="store_true",
                   help="做一次小幅往返运动（⚠️ 机械臂会真动，默认不做）")
    p.add_argument("--axis", type=int, default=2, choices=[0, 1, 2],
                   help="测试轴：0=x 1=y 2=z（默认 2=z）")
    p.add_argument("--delta", type=float, default=5.0, help="运动幅度 mm（默认 5）")
    p.add_argument("--no-enable", action="store_true",
                   help="对照实验：临时关掉自动使能（预期结果是「不动」）")
    p.add_argument("--settle", type=float, default=3.0, help="每步后等待秒数")
    args = p.parse_args(argv)

    from .config import load_config
    from .hw import Hardware

    cfg = load_config(args.config)
    if args.no_enable:
        cfg.robot.enable_on_connect = False

    axis_name = "xyz"[args.axis]

    print("=" * 66)
    print(f"enable_on_connect = {cfg.robot.enable_on_connect}")
    print("=" * 66)

    hw = Hardware(cfg)
    hw.connect(connect_teleop=False)
    try:
        assert hw.robot is not None
        p0 = _read_pose(hw)
        print("\nconnect 后位姿 :", [round(v, 3) for v in p0])

        # 诊断：运动起始点。hw.connect() 已在 enable 之后设过一次，这里再确认一次。
        # ⚠️ 注意 fork 里 `start_cartesian_position` 的返回值被 `not` 反转
        #    （`hsrosi/controller.py:379`），所以**不能只看它的返回值** ——
        #    真正判据是 stdout 里有没有 `[error] 设置运动起始点失败`。
        try:
            hrc = getattr(hw.robot, "HRC", None)
            cur = [float(v) for v in hrc.get_cartesian_position()]
            hrc.start_cartesian_position(cur)
            print("起始点重设     : 已调用（判据看上面 stdout 有无 "
                  "`[error] 设置运动起始点失败`；无 error = 成功）")
        except Exception as exc:
            print(f"起始点重设     : ❌ {type(exc).__name__}: {exc}")

        if not args.move_test:
            print(f"\n（未做运动测试。加 --move-test 会沿 {axis_name} 轴走 "
                  f"±{args.delta:.1f} mm 后回原位）")
            return 0

        target = list(p0)
        target[args.axis] += args.delta
        print(f"\n>>> 运动测试：沿 {axis_name} 轴 +{args.delta:.1f} mm")
        print("    目标位姿 :", [round(v, 3) for v in target])

        hw.robot.HRC.move_to_cartesian_position(target)
        time.sleep(args.settle)
        p1 = _read_pose(hw)
        print("    实测位姿 :", [round(v, 3) for v in p1])

        moved = p1[args.axis] - p0[args.axis]
        print(f"    实际位移 = {moved:+.2f} mm")

        if abs(moved) < abs(args.delta) * 0.5:
            print("\n❌ 机械臂**没有动** —— 运动通道仍然不通。")
            print("   按这个顺序排查（都在现场能看/能问）：")
            print("     1) 示教器是否处于【远程 / 自动】模式（手动模式不接受程序指令）")
            print("     2) 是否处于急停状态、或伺服未上电（看控制柜指示灯）")
            print("     3) 是否有别的程序也在连这台臂（ps 查）")
            print("     4) hsrobot 是否真的支持 enable()（看上面的日志有没有警告）")
            return 3

        print(f"\n✅ 运动通道可用（位移 {moved:+.2f} mm，目标 {args.delta:+.1f} mm）")
        print("   >>> 现在回原位 ...")
        hw.robot.HRC.move_to_cartesian_position(p0)
        time.sleep(args.settle)
        p2 = _read_pose(hw)
        print("   回位后位姿 :", [round(v, 3) for v in p2])
        back = p2[args.axis] - p0[args.axis]
        print(f"   残余偏差 = {back:+.2f} mm")
        return 0
    finally:
        hw.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
