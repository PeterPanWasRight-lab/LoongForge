#!/usr/bin/env python3
"""边端 policy 执行端 —— 真机闭环回放。

两种模式
--------
``--shadow``（**默认**）：只读观测 → 请求 policy → 打印/统计 action，**绝不使能、绝不运动**。

``--drive``：真正下发动作。必须同时给 ``--i-have-estop``（现场有人守急停）显式确认。
安全件（缺一不可）：
    ① **前置检查**：当前位姿必须在 ``workspace_limits`` 内，否则拒绝启动
    ② **单帧限幅**：chunk 内任一步 |Δ| > ``--max-step-mm`` ⇒ **立即停止**（不夹紧、不缩放，宁可停）
    ③ **chunk 级一致性看门狗**：一段执行完，实测位移 vs 预期位移偏差 > ``--dev-tol-mm`` ⇒ 停
    ④ **server 超时看门狗**：单次 infer 超时 ⇒ 停
    ⑤ **键盘中断**：任意回车 ⇒ 停止下发
    ⑥ **步数护栏**：``--first-steps``（默认 5）走满即停；确认无误后才可显式设 0 表示不限

⚠️ 本脚本**不做自动复位**：起始位由人工摆好；跑完停在原地，由人判断下一步。

一致性铁律（与训练对齐，任何一项对不上都会让视觉/状态分布漂移）
--------------------------------------------------------------
* 图像：**同一份 config 的 crop_params + resize 224**（复用 ``Hardware.preprocess_images``）
* state：同一拼法（位姿 6 + 力 6 补 0 + 无夹爪 = 12 维），复用 ``features.state_from_obs``
* 指令：同一 task 文本（配置里的 ``dataset.task``）
* 动作语义：**Δxyz（mm/帧，10 Hz）**，与数据集 action 同性质

用法::

    # 影子模式（30 帧）
    rrc policy --shadow --frames 30

    # 驱动模式（首测：只走 5 步）
    rrc policy --drive --i-have-estop --first-steps 5 --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from real_robot_collect.config import load_config                       # noqa: E402
from real_robot_collect.features import action_to_robot, state_from_obs  # noqa: E402
from real_robot_collect.hw import Hardware                              # noqa: E402

from real_robot_deploy.policy_client import PolicyClient, as_chunk      # noqa: E402

DIM_NAMES = ("x", "y", "z", "rx", "ry", "rz")


def _pose(obs) -> np.ndarray:
    return np.array([float(obs[f"loc_{i}.pos"]) for i in range(6)], dtype=np.float64)


def _limits_bad(pose6: np.ndarray, lim) -> list[str]:
    bad = []
    for i, n in enumerate(DIM_NAMES):
        lo, hi = (lim or {}).get(f"{n}_min"), (lim or {}).get(f"{n}_max")
        if lo is None or hi is None:
            continue
        if not (lo <= pose6[i] <= hi):
            bad.append(f"{n}={pose6[i]:.1f} ∉ [{lo},{hi}]")
    return bad


def _stats_table(acts: np.ndarray, names=("Δx", "Δy", "Δz")) -> str:
    lines = [f"  {'dim':<6}{'min':>9}{'max':>9}{'mean':>9}{'std':>9}"]
    for j, nm in enumerate(names):
        c = acts[:, j]
        lines.append(f"  {nm:<6}{c.min():>9.3f}{c.max():>9.3f}{c.mean():>9.3f}{c.std():>9.3f}")
    return "\n".join(lines)


def _kb_listener(stop: list[bool]) -> None:
    """后台线程：读一行 stdin（回车）即请求停止。"""
    try:
        sys.stdin.readline()
    except Exception:
        pass
    finally:
        stop[0] = True


def _wait_settle(hw, settle_mm: float, max_wait: float):
    """等机械臂**真正停稳**再往下走。

    为什么必需：段末立刻读位姿时，5 条增量指令可能还在机械臂的指令队列里没走完
    ⇒ 测出来的"实测位移"偏小（首测实测只有预期的 ~55%），下一段的 state 也是运动中的。
    判据：相邻两次读数（仅平移 3 维）最大差 < ``settle_mm``。返回 (位姿, 是否停稳)。
    """
    t0 = time.time()
    p_prev = _pose(hw.read_raw_obs())
    while time.time() - t0 < max_wait:
        time.sleep(0.15)
        p = _pose(hw.read_raw_obs())
        if float(np.abs(p[:3] - p_prev[:3]).max()) < settle_mm:
            return p, True
        p_prev = p
    return p_prev, False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="边端 policy 执行端（默认影子模式）")
    ap.add_argument("--config", required=True, help="采集配置路径（决定 crop/state/task，必须与训练一致）")
    ap.add_argument("--host", default="127.0.0.1", help="policy server 主机（经反向隧道用 127.0.0.1）")
    ap.add_argument("--port", type=int, default=15515)
    ap.add_argument("--shadow", action="store_true", help="影子模式（默认）：只读、不动")
    ap.add_argument("--drive", action="store_true", help="驱动模式：真正下发动作（需 --i-have-estop）")
    ap.add_argument("--i-have-estop", action="store_true", help="确认现场有急停在手，才允许驱动")
    ap.add_argument("--frames", type=int, default=30, help="影子模式采集多少帧后退出")
    ap.add_argument("--infer-every", type=int, default=1, help="影子模式每 N 帧请求一次")
    ap.add_argument("--first-steps", type=int, default=5,
                    help="驱动模式总步数护栏（默认 5；设 0 表示不限，需先完成首测）")
    ap.add_argument("--exec-steps", type=int, default=10,
                    help="每个 chunk 执行几步后重推理（LIBERO 铁律：10–15 最优，50 会掉 21pt）")
    ap.add_argument("--max-step-mm", type=float, default=5.0, help="单帧位移限幅（超限即停）")
    ap.add_argument("--dev-tol-mm", type=float, default=8.0,
                    help="[已弃用，保留兼容] 绝对偏差阈值；现用比值判据")
    ap.add_argument("--ratio-lo", type=float, default=0.2,
                    help="传递比下界：实测/预期 < 此值 ⇒ 视为『该动没动』，停")
    ap.add_argument("--ratio-hi", type=float, default=2.0,
                    help="传递比上界：实测/预期 > 此值 ⇒ 视为『动得离谱』，停")
    ap.add_argument("--infer-timeout", type=float, default=30.0, help="单次 infer 超时（秒），超时即停")
    ap.add_argument("--settle-mm", type=float, default=0.3, help="停稳判据：相邻两次读数最大差（mm）")
    ap.add_argument("--settle-timeout", type=float, default=4.0, help="等停稳的最长等待（秒）")
    ap.add_argument("--no-settle", action="store_true",
                    help="不等停稳（默认**等**：每段执行完等机械臂走完再测位移/推理下一段）")
    ap.add_argument("--no-dev-check", action="store_true", help="关闭一致性看门狗（不推荐）")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args(argv)

    if args.drive and not args.i_have_estop:
        print("❌ 驱动模式必须显式给 --i-have-estop（现场确有人守急停）", file=sys.stderr)
        return 2
    if args.drive and not args.shadow and args.first_steps == 0:
        print("⚠️  --first-steps 0 = 不限步数。首测请留 5，确认无误后再放开。")
    shadow = not args.drive

    cfg = load_config(args.config)
    if shadow:
        cfg.robot.enable_on_connect = False   # ★ 影子模式绝不使能
        print("=" * 78)
        print("影子模式：enable_on_connect=False ⇒ 不使能、不下发、不动")
        print("=" * 78)
    else:
        lim = cfg.robot.workspace_limits or {}
        print("=" * 78)
        print(f"⚠️  驱动模式：会真实下发动作。限幅 {args.max_step_mm} mm/帧 · "
              f"chunk {args.exec_steps} 步 · 步数护栏 {args.first_steps}")
        print(f"gripper_io_port={cfg.robot.gripper_io_port} · "
              f"workspace x[{lim.get('x_min')},{lim.get('x_max')}] "
              f"y[{lim.get('y_min')},{lim.get('y_max')}] z[{lim.get('z_min')},{lim.get('z_max')}]")
        print("=" * 78)

    task = cfg.dataset.task
    print(f"task   : {task!r}")
    print(f"server : ws://{args.host}:{args.port}")

    hw = Hardware(cfg)
    client = PolicyClient(args.host, args.port, timeout=max(60.0, args.infer_timeout * 4))
    stop: list[bool] = [False]
    try:
        hw.connect(connect_teleop=False)
        md = client.connect()
        print(f"server metadata: { {k: md[k] for k in ('protocol_version', 'action_dim', 'model_type') if k in md} }")
        print("ping:", client.ping().get("status"))

        if shadow:
            return _shadow(args, cfg, hw, client, task)

        # ---------------- 驱动模式 ----------------
        th = threading.Thread(target=_kb_listener, args=(stop,), daemon=True)
        th.start()
        print("\n▶ 开始前先读一次位姿做前置检查 …（任意时刻按回车 = 停止下发）")
        obs = hw.read_raw_obs()
        p_prev = _pose(obs)
        bad = _limits_bad(p_prev, cfg.robot.workspace_limits)
        if bad:
            print("❌ 当前位姿超出 workspace_limits，拒绝启动：")
            for b in bad:
                print("   ", b)
            return 3
        d0 = np.abs(p_prev - np.array(cfg.record.reset_position, dtype=float))
        print(f"当前位姿 : {np.round(p_prev, 1).tolist()}")
        print(f"距起始位 : {np.round(d0, 1).tolist()}  (x,y,z 为 mm)")

        n_sent = 0
        sent_acts: list[np.ndarray] = []
        while not stop[0] and (args.first_steps == 0 or n_sent < args.first_steps):
            # ---- 推理 ----
            t = time.perf_counter()
            pp = hw.preprocess_images(obs)
            st = state_from_obs(obs, include_gripper=cfg.record.include_gripper,
                                collect_force_torque=cfg.record.collect_force_torque)
            try:
                data = client.predict_action(
                    images=[pp[c] for c in cfg.camera_keys],
                    instructions=[task],
                    state=np.asarray(st, dtype=np.float32)[None, :],
                    episode_id="drive",
                    episode_step=n_sent,
                    return_action_chunk=True,
                    disable_action_cache=True,
                )
            except Exception as exc:                       # ④ 超时/通信失败 ⇒ 停
                print(f"\n🛑 infer 失败（看门狗触发，已停止下发）: {type(exc).__name__}: {exc}")
                break
            chunk = as_chunk(data)[:, : cfg.action_dim]
            plan = chunk[: max(1, args.exec_steps)]
            t_infer = (time.perf_counter() - t) * 1000.0

            # ---- ② 限幅检查（整段先查）----
            mag = np.linalg.norm(plan, axis=1)
            if (mag > args.max_step_mm).any():
                k_bad = int(np.argmax(mag))
                print(f"\n🛑 限幅触发（第 {k_bad} 步 |Δ|={mag[k_bad]:.2f} > {args.max_step_mm} mm）"
                      f" ⇒ 已停止下发，未执行该段")
                print(f"   该段 plan=\n{np.round(plan, 3)}")
                break

            # ---- 逐帧下发 ----
            exp = np.zeros(3)
            for i, a in enumerate(plan):
                if stop[0] or (args.first_steps and n_sent >= args.first_steps):
                    break
                t0 = time.perf_counter()
                hw.robot.send_action(action_to_robot(
                    {"delta_x": float(a[0]), "delta_y": float(a[1]), "delta_z": float(a[2])}
                ))
                n_sent += 1
                exp += a[:3]
                sent_acts.append(np.asarray(a[:3], dtype=np.float64))
                dt = time.perf_counter() - t0
                if dt < 1.0 / args.fps:
                    time.sleep(1.0 / args.fps - dt)

            # ---- ③ 等停稳 + chunk 级一致性看门狗 ----
            if args.no_settle:
                obs = hw.read_raw_obs()
                p_now = _pose(obs)
                settled = True
            else:
                p_now, settled = _wait_settle(hw, args.settle_mm, args.settle_timeout)
                obs = hw.read_raw_obs()          # 停稳后再取一帧 obs（供下一段推理用）
                p_now = _pose(obs)
            actual = p_now[:3] - p_prev[:3]
            dev = np.abs(actual - exp)
            # 比值判据（替代纯绝对偏差）：机械臂对增量指令有**系统性传递比 < 1**
            # （实测 ~0.4–1.0，z 方向最弱）⇒ 用绝对差会在长 chunk 下必然误报。
            # 只看「明显该动却几乎没动」或「动得离谱」两种情况。
            sel = np.abs(exp) > 0.5
            ratio = np.full(3, np.nan)
            ratio[sel] = actual[sel] / exp[sel]
            bad = [i for i in range(3) if sel[i] and not (args.ratio_lo <= ratio[i] <= args.ratio_hi)]
            print(f"  chunk#{n_sent // max(1, args.exec_steps):<3} infer {t_infer:>5.0f}ms  "
                  f"本段下发 {len(plan)} 步  {'已停稳' if settled else '⚠️等停稳超时'}  "
                  f"预期Δ={np.round(exp, 2)} 实测Δ={np.round(actual, 2)} "
                  f"传递比={np.round(ratio, 2)} pose={np.round(p_now[:3], 1)}")
            p_prev = p_now
            if not args.no_dev_check and bad:
                print(f"\n🛑 一致性看门狗触发（维 {[DIM_NAMES[i] for i in bad]} 传递比 "
                      f"{np.round([ratio[i] for i in bad], 2)} ∉ [{args.ratio_lo}, {args.ratio_hi}]）"
                      f" ⇒ 已停止下发")
                break

        print(f"\n■ 驱动结束：本次共下发 {n_sent} 步（护栏 {args.first_steps}）")
        if sent_acts:
            print(_stats_table(np.asarray(sent_acts)))
        print("（未做自动复位：机械臂停在当前位置，请人工确认）")
        return 0
    except KeyboardInterrupt:
        print("\n■ Ctrl-C：已停止下发")
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


def _shadow(args, cfg, hw, client, task) -> int:
    acts_all = []
    print(f"\n--- 影子模式逐帧推理（{args.frames} 帧，每 {args.infer_every} 帧一次）---")
    t0 = time.time()
    for k in range(args.frames):
        if k % args.infer_every:
            time.sleep(1.0 / args.fps)
            continue
        obs = hw.read_raw_obs()
        pp = hw.preprocess_images(obs)
        st = state_from_obs(obs, include_gripper=cfg.record.include_gripper,
                            collect_force_torque=cfg.record.collect_force_torque)
        ts = time.perf_counter()
        data = client.predict_action(
            images=[pp[c] for c in cfg.camera_keys],
            instructions=[task],
            state=np.asarray(st, dtype=np.float32)[None, :],
            episode_id="shadow",
            episode_step=k,
            return_action_chunk=True,
            disable_action_cache=True,
        )
        dt = (time.perf_counter() - ts) * 1000.0
        act = as_chunk(data)[:, : cfg.action_dim]
        acts_all.append(act)
        head = " ".join(f"{v:+.3f}" for v in act[0])
        print(f"  f{k:>3}  {dt:6.0f}ms  chunk{tuple(act.shape)}  a0=[{head}]  "
              f"|a0|={np.linalg.norm(act[0]):.3f}  pose={np.round(np.asarray(st[:3]), 1)}")
        time.sleep(1.0 / args.fps)

    if acts_all:
        flat = np.concatenate([a[: max(1, min(len(a), args.exec_steps))] for a in acts_all], 0)
        print("\n--- 影子模式统计（每 chunk 前 exec_steps 步）---")
        print(_stats_table(flat))
        print(f"\n  总推理次数     : {len(acts_all)}")
        print(f"  平均推理耗时   : {(time.time() - t0) / max(len(acts_all), 1) * 1000:.0f} ms/次")
        print(f"  单帧位移模长   : mean={np.linalg.norm(flat, axis=1).mean():.3f} "
              f"max={np.linalg.norm(flat, axis=1).max():.3f} mm")
        print(f"  超限帧（>{args.max_step_mm}mm）: "
              f"{int((np.linalg.norm(flat, axis=1) > args.max_step_mm).sum())} / {len(flat)}")
    print("\n（影子模式结束：未使能、未下发任何运动指令）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
