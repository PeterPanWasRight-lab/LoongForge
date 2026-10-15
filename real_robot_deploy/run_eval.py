#!/usr/bin/env python3
"""真机评测脚本 —— 自动复位 + 人工判定（s/q）+ 成功率统计。

与 `run_policy.py --drive` 的区别：**这个脚本是给"跑 N 次统计成功率"用的**，
每轮自动复位、自动进入下一轮，人只负责判定成功/失败。

交互（**单字符，无需回车**）
--------------------------
* `空格` —— 本轮准备就绪，开始跑（程序会先自动复位）
* `s`    —— 标记**成功**，立即停止本轮下发，进入下一轮
* `q`    —— 标记**失败**，立即停止本轮下发，进入下一轮
* `ESC`  —— **中止整个测试**（不是当前轮）

流程（每轮）
------------
1. 先向上抬 `--lift-mm`（默认 40mm，分步走）—— 避免从孔底直接插值复位时刮擦
2. `reset_to_position()` 自动复位到配置的起始位，等停稳
3. 提示"请装好 pin，按空格开始"
4. 跑 policy 最多 `--steps` 步（每 `--exec-steps` 步重推理）
5. 人按 s/q 判定；若步数用尽仍未判定，程序会等待判定

安全件（与 run_policy 同源）
-----------------------------
前置位姿检查 / 单帧限幅 / **等停稳** / **比值看门狗** / infer 超时 / ESC 中止。
任何安全件触发 ⇒ 该轮自动记为失败（并打印原因），不阻塞测试流程。

用法::

    python -m real_robot_deploy.run_eval --config .../pin_in_hole_v1.json \
        --episodes 10 --steps 100 --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from real_robot_collect.config import load_config                             # noqa: E402
from real_robot_collect.features import action_to_robot, state_from_obs       # noqa: E402
from real_robot_collect.hw import Hardware                                    # noqa: E402

from real_robot_deploy.policy_client import PolicyClient, as_chunk            # noqa: E402
from real_robot_deploy.run_policy import _limits_bad, _pose, _wait_settle     # noqa: E402


# --------------------------------------------------------------------------- 键盘
class Keys:
    """单字符非阻塞键盘。stdin 不是 tty（被重定向）时降级为逐行读。"""

    def __init__(self) -> None:
        self.is_tty = sys.stdin.isatty()
        self._old = None

    def __enter__(self) -> "Keys":
        if self.is_tty:
            self._old = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc) -> None:
        if self.is_tty and self._old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)

    def get(self, timeout: float = 0.0):
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        ch = sys.stdin.read(1)
        return ch or None

    def wait(self, wanted: str, timeout: float = 0.25):
        """阻塞直到读到 wanted 里的字符（或 ESC）；返回读到的字符。"""
        while True:
            ch = self.get(timeout)
            if ch is None:
                continue
            if ch in wanted or ch == "\x1b":
                return ch


ESC = "\x1b"


# --------------------------------------------------------------------------- 抬升+复位
def lift_and_reset(args, cfg, hw, label: str = "") -> np.ndarray:
    """先向上抬 `--lift-mm`（分步），再 reset_to_position 回起始位，最后等停稳。

    为什么要先抬：若上一轮把 pin 插进孔里，直接调用 reset_to_position 会**斜向/直线**离开，
    容易刮擦孔壁或把工件带歪；先纯 +z 抬起是安全做法。
    返回复位后的位姿。
    """
    if not args.no_lift and args.lift_mm > 0:
        step = args.lift_mm / max(1, args.lift_steps)
        for _ in range(args.lift_steps):
            hw.robot.send_action(action_to_robot(
                {"delta_x": 0.0, "delta_y": 0.0, "delta_z": float(step)}))
            time.sleep(1.0 / args.fps)
        _wait_settle(hw, args.settle_mm, args.settle_timeout)
        print(f"  {label}[抬升] 已向上 {args.lift_mm:.0f}mm 并停稳")

    print(f"  {label}[复位] 回到起始位 {np.round(cfg.record.reset_position, 1).tolist()} …")
    hw.reset_to_start()
    if not args.no_settle:
        _wait_settle(hw, args.settle_mm, args.settle_timeout)
    p = _pose(hw.read_raw_obs())
    d = np.abs(p - np.array(cfg.record.reset_position, dtype=float))[:3]
    print(f"  {label}[复位] 完成，偏差 {np.round(d, 2).tolist()} mm")
    return p


# --------------------------------------------------------------------------- 单轮
def run_episode(args, cfg, hw, client, kb: Keys, ep_idx: int):
    """跑一轮。返回 (outcome, steps_done, reason)。

    outcome: 'success' | 'fail' | 'abort'
    """
    task = cfg.dataset.task
    obs = hw.read_raw_obs()
    p_prev = _pose(obs)
    n_sent = 0
    reason = ""

    while n_sent < args.steps:
        ch = kb.get(0.0)
        if ch == "s":
            return "success", n_sent, "人工判定成功"
        if ch == "q":
            return "fail", n_sent, "人工判定失败"
        if ch == ESC:
            return "abort", n_sent, "ESC 中止"

        # ---- 推理 ----
        pp = hw.preprocess_images(obs)
        # 可选：把本段的观测落盘（视觉/光照干扰测试用 —— 事后能直接看 policy「看到了什么」）
        if args.dump_images:
            try:
                from PIL import Image as _Image
                _d = Path(args.dump_images)
                _d.mkdir(parents=True, exist_ok=True)
                for _c in cfg.camera_keys:
                    _Image.fromarray(np.asarray(pp[_c])).save(
                        _d / f"ep{ep_idx + 1:02d}_s{n_sent:04d}_{_c}.png")
            except Exception as _e:      # 存图失败不能影响评测
                print(f"    (存图失败: {_e})")
        st = state_from_obs(obs, include_gripper=cfg.record.include_gripper,
                            collect_force_torque=cfg.record.collect_force_torque)
        t0 = time.perf_counter()
        try:
            data = client.predict_action(
                images=[pp[c] for c in cfg.camera_keys],
                instructions=[task],
                state=np.asarray(st, dtype=np.float32)[None, :],
                episode_id=f"eval{ep_idx}",
                episode_step=n_sent,
                return_action_chunk=True,
                disable_action_cache=True,
            )
        except Exception as exc:
            return "fail", n_sent, f"infer 失败: {type(exc).__name__}"
        chunk = as_chunk(data)[:, : cfg.action_dim]
        plan = chunk[: max(1, args.exec_steps)]
        t_infer = (time.perf_counter() - t0) * 1000.0

        # ---- 限幅 ----
        mag = np.linalg.norm(plan, axis=1)
        if (mag > args.max_step_mm).any():
            k = int(np.argmax(mag))
            return "fail", n_sent, f"限幅触发(第{k}步 |Δ|={mag[k]:.2f}mm)"

        # ---- 逐帧下发（每步查键盘）----
        exp = np.zeros(3)
        for i, a in enumerate(plan):
            ch = kb.get(0.0)
            if ch == "s":
                return "success", n_sent, "人工判定成功"
            if ch == "q":
                return "fail", n_sent, "人工判定失败"
            if ch == ESC:
                return "abort", n_sent, "ESC 中止"
            if n_sent >= args.steps:
                break
            t1 = time.perf_counter()
            hw.robot.send_action(action_to_robot(
                {"delta_x": float(a[0]), "delta_y": float(a[1]), "delta_z": float(a[2])}
            ))
            n_sent += 1
            exp += a[:3]
            dt = time.perf_counter() - t1
            if dt < 1.0 / args.fps:
                time.sleep(1.0 / args.fps - dt)

        # ---- 等停稳 + 比值看门狗 ----
        if args.no_settle:
            p_now = _pose(hw.read_raw_obs())
            settled = True
        else:
            p_now, settled = _wait_settle(hw, args.settle_mm, args.settle_timeout)
        actual = p_now[:3] - p_prev[:3]
        sel = np.abs(exp) > 0.5
        ratio = np.full(3, np.nan)
        ratio[sel] = actual[sel] / exp[sel]
        bad = [i for i in range(3) if sel[i] and not (args.ratio_lo <= ratio[i] <= args.ratio_hi)]
        print(f"    ep{ep_idx+1} 步{n_sent:>4}  infer {t_infer:>5.0f}ms  "
              f"预期Δ={np.round(exp, 2)} 实测Δ={np.round(actual, 2)} "
              f"比值={np.round(ratio, 2)} pose={np.round(p_now[:3], 1)}"
              + ("" if settled else "  ⚠️停稳超时"))
        p_prev = p_now
        obs = hw.read_raw_obs()
        if not args.no_dev_check and bad:
            return "fail", n_sent, f"看门狗触发(维{[bad]} 比值{np.round([ratio[i] for i in bad], 2)})"

    # 步数用尽 → 等人工判定
    print(f"    ep{ep_idx+1} 已达步数上限 {args.steps}，请判定：s=成功 / q=失败")
    ch = kb.wait("sq")
    if ch == "s":
        return "success", n_sent, "人工判定成功(步数用尽)"
    if ch == "q":
        return "fail", n_sent, "人工判定失败(步数用尽)"
    return "abort", n_sent, "ESC 中止"


# --------------------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="真机评测：自动复位 + s/q 判定 + 成功率统计")
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=15515)
    ap.add_argument("--episodes", type=int, required=True, help="本次测试几轮")
    ap.add_argument("--steps", type=int, default=150, help="每轮最多下发多少步")
    ap.add_argument("--exec-steps", type=int, default=10, help="每个 chunk 执行几步后重推理")
    ap.add_argument("--max-step-mm", type=float, default=5.0)
    ap.add_argument("--ratio-lo", type=float, default=0.2)
    ap.add_argument("--ratio-hi", type=float, default=2.0)
    ap.add_argument("--settle-mm", type=float, default=0.3)
    ap.add_argument("--settle-timeout", type=float, default=4.0)
    ap.add_argument("--no-settle", action="store_true")
    ap.add_argument("--no-dev-check", action="store_true")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--lift-mm", type=float, default=40.0,
                    help="复位前先向上抬这么多 mm（避免从孔底直线复位刮擦）")
    ap.add_argument("--lift-steps", type=int, default=10)
    ap.add_argument("--no-lift", action="store_true")
    ap.add_argument("--no-final-reset", action="store_true",
                    help="测试结束后不回原点（默认**回**：先抬升再复位）")
    ap.add_argument("--dump-images", default="",
                    help="把每段观测的 front/wrist 图存到该目录（视觉/光照干扰测试用）")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    print("=" * 84)
    print(f"真机评测：{args.episodes} 轮 · 每轮最多 {args.steps} 步 · "
          f"chunk {args.exec_steps} 步 · 限幅 {args.max_step_mm}mm")
    print("交互：空格=开始本轮 · s=成功 · q=失败 · ESC=中止整个测试（单字符，不用回车）")
    print("=" * 84)

    hw = Hardware(cfg)
    client = PolicyClient(args.host, args.port, timeout=120.0)
    results = []          # (outcome, steps, reason)
    try:
        hw.connect(connect_teleop=False)
        md = client.connect()
        print("server:", {k: md[k] for k in ("protocol_version", "action_dim", "model_type") if k in md})

        # 起始位姿态检查（只在开始前做一次）
        p0 = _pose(hw.read_raw_obs())
        bad = _limits_bad(p0, cfg.robot.workspace_limits)
        if bad:
            print("❌ 起始位姿超出 workspace_limits，拒绝启动：", bad)
            return 3

        with Keys() as kb:
            aborted = False
            for ep in range(args.episodes):
                # ---- ①② 抬升 + 自动复位 ----
                lift_and_reset(args, cfg, hw)

                # ---- ③ 等用户就绪 ----
                print(f"\n===== 第 {ep + 1}/{args.episodes} 轮 =====")
                print("请确认 pin 已装好 → 按【空格】开始（ESC 中止整个测试）")
                ch = kb.wait(" ")
                if ch == ESC:
                    aborted = True
                    break

                # ---- ④ 跑 ----
                outcome, nsteps, reason = run_episode(args, cfg, hw, client, kb, ep)
                results.append((outcome, nsteps, reason))
                tag = {"success": "✅ 成功", "fail": "❌ 失败", "abort": "⏹ 中止"}[outcome]
                print(f"  → 第 {ep + 1} 轮：{tag}（{nsteps} 步）· {reason}")
                if outcome == "abort":
                    aborted = True
                    break

            # ---- ⑤ 收尾：先返回原点，再出统计 ----
            if not args.no_final_reset:
                print("\n[收尾] 正在返回原点 …")
                lift_and_reset(args, cfg, hw, label="收尾")
                print("[收尾] ✅ 已回到起始位")

            # ---- ⑥ 统计 ----
            print("\n" + "=" * 84)
            print("评测结果")
            print("=" * 84)
            n_ok = sum(1 for o, _, _ in results if o == "success")
            n_fail = sum(1 for o, _, _ in results if o == "fail")
            print(f"  {'轮':<4}{'结果':<8}{'步数':>6}   原因")
            for i, (o, s, r) in enumerate(results, 1):
                print(f"  {i:<4}{o:<8}{s:>6}   {r}")
            print("-" * 84)
            if results:
                print(f"  成功 {n_ok} / 失败 {n_fail} / 共 {len(results)} 轮"
                      f"  ⇒  成功率 {n_ok / len(results) * 100:.1f}%")
            # 失败模式归类
            fails = [r for o, _, r in results if o == "fail"]
            for key in ("看门狗", "限幅", "infer", "人工判定失败"):
                c = sum(1 for r in fails if key in r)
                if c:
                    print(f"    失败原因含『{key}』: {c} 轮")
            if aborted:
                print("  ⚠️ 测试被 ESC 提前中止")
            print("=" * 84)
        return 0
    except KeyboardInterrupt:
        print("\n■ Ctrl-C：已停止")
        return 130
    finally:
        try:
            client.close()
        except Exception:
            pass
        try:
            hw.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
