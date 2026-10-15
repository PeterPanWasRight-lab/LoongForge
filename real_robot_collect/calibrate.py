#!/usr/bin/env python3
"""起始位姿标定工具 —— **默认只读**，绝不使能、绝不运动。

用途（现场评估 + 标定起始点）
----------------------------
1. 读当前位姿，打印它与配置 `record.reset_position` 的**逐维差值**，并对
   `robot.workspace_limits` 做**逐维边界检查**（超限会被 `send_action`/`reset_to_position`
   静默 clip，是撞机的直接原因）。
2. 存下两路相机的**原始帧 + 预处理帧**（人眼确认工位/工件是否就位）。
3. 只有显式加 `--apply` 才会把**当前位姿写成新的 `reset_position`**，并围绕它重算
   `workspace_limits`（平移 ±`--span` mm）；写之前自动备份原配置。

安全性设计
----------
* 连接时强制 `enable_on_connect=False` ⇒ **不会使能机械臂**，位姿只读。
* `--apply` 前会**连读两次位姿**，若仍在移动（差值 > `--stable-mm`）则拒绝标定 ——
  防止把"运动中"的位姿当成起始点。
* 写配置前先备份为 `<config>.bak-<YYYYmmdd-HHMMSS>`。

用法::

    python -m real_robot_collect.calibrate                  # 只读：打印差值 + 存快照
    python -m real_robot_collect.calibrate --apply          # 标定：写成起始点 + 重算边界
    python -m real_robot_collect.calibrate --apply --span 200
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

DIM_NAMES = ["x", "y", "z", "rx", "ry", "rz"]
LIMIT_KEYS = [(f"{n}_min", f"{n}_max") for n in DIM_NAMES]


def read_pose(hw, n: int = 6) -> list[float]:
    obs = hw.read_raw_obs()
    return [round(float(obs[f"loc_{i}.pos"]), 3) for i in range(n)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="起始位姿标定（默认只读）")
    ap.add_argument("--config", required=False, default=None, help="配置路径（rrc 包装器会传）")
    ap.add_argument("--apply", action="store_true", help="把当前位姿写成 reset_position（会先备份配置）")
    ap.add_argument("--span", type=float, default=100.0,
                    help="重算 workspace_limits 时平移维的半宽 mm（默认 100）")
    ap.add_argument("--stable-mm", type=float, default=0.5,
                    help="--apply 前的静止判据：两次读数最大差 (mm/deg)，默认 0.5")
    ap.add_argument("--outdir", default="/tmp/rrc_calib", help="快照输出目录")
    ap.add_argument("--no-snap", action="store_true", help="不存相机快照")
    args = ap.parse_args(argv)

    from .config import load_config
    from .hw import Hardware

    cfg_path = Path(args.config or "/home/gm/Project/Test/lerobot_test/real_robot_collect/configs/peg_v1.json")
    cfg = load_config(cfg_path)
    # ★ 铁律：标定工具绝不使能、绝不运动
    cfg.robot.enable_on_connect = False

    hw = Hardware(cfg)
    try:
        hw.connect(connect_teleop=False)
        pose_a = read_pose(hw)
        time.sleep(1.0)
        pose_b = read_pose(hw)

        print("\n" + "=" * 78)
        print("当前位姿      :", pose_a)
        print("配置 reset    :", list(cfg.record.reset_position))
        print("差值(当前-配置):", [round(pose_a[i] - cfg.record.reset_position[i], 2) for i in range(6)])
        print("-" * 78)
        L = cfg.robot.workspace_limits or {}
        print("%-4s %12s   %18s   %s" % ("维", "当前", "workspace_limits", "判定"))
        bad = 0
        for i, (n, (lo_k, hi_k)) in enumerate(zip(DIM_NAMES, LIMIT_KEYS)):
            lo, hi = L.get(lo_k), L.get(hi_k)
            if lo is None or hi is None:
                print("%-4s %12.2f   %18s   (未配置)" % (n, pose_a[i], "-")); continue
            ok = lo <= pose_a[i] <= hi
            bad += 0 if ok else 1
            print("%-4s %12.2f   [%8.1f, %8.1f]   %s" % (n, pose_a[i], lo, hi, "OK" if ok else "❌ 超限"))
        if bad:
            print(f"\n⚠️  {bad} 维超限：直接开采集会在复位时被 clip 并做大幅移动（撞机风险）")
        print("=" * 78)

        drift = max(abs(pose_a[i] - pose_b[i]) for i in range(6))
        print(f"静止判定：两次读数最大差 = {drift:.3f}（阈值 {args.stable_mm}）"
              f" ⇒ {'静止 ✓' if drift <= args.stable_mm else '⚠️ 仍在移动'}")

        if not args.no_snap:
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            obs = hw.read_raw_obs()
            try:
                from PIL import Image
            except ModuleNotFoundError:
                print("（无 PIL，跳过快照）")
            else:
                for cam in cfg.camera_keys:
                    a = np.asarray(obs[cam])
                    Image.fromarray(a).save(outdir / f"raw_{cam}.png")
                for k, v in hw.preprocess_images(obs).items():
                    Image.fromarray(v).save(outdir / f"pp_{k}.png")
                print(f"快照已存: {outdir}/raw_*.png + pp_*.png")

        if not args.apply:
            print("\n（只读模式：未改动任何配置。确认当前位姿就是起始点后，加 --apply 再跑一次）")
            return 0

        if drift > args.stable_mm:
            print("\n❌ 机械臂仍在移动，拒绝标定（防把运动中的位姿当起点）。停稳后重试，"
                  "或临时加更大的 --stable-mm。")
            return 3

        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = cfg_path.with_suffix(cfg_path.suffix + f".bak-{ts}")
        shutil.copy2(cfg_path, bak)
        raw = json.loads(cfg_path.read_text())
        raw["record"]["reset_position"] = [round(float(v), 3) for v in pose_a]
        lim = {}
        for n in ("x", "y", "z"):
            lim[f"{n}_min"] = round(pose_a[DIM_NAMES.index(n)] - args.span, 1)
            lim[f"{n}_max"] = round(pose_a[DIM_NAMES.index(n)] + args.span, 1)
        for n in ("rx", "ry", "rz"):
            lim[f"{n}_min"], lim[f"{n}_max"] = -180.0, 180.0
        raw["robot"]["workspace_limits"] = lim
        cfg_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
        print("\n✅ 已写入配置")
        print("   reset_position   =", raw["record"]["reset_position"])
        print("   workspace_limits =", lim)
        print("   备份            =", bak)
        print("\n⚠️ 提醒：工件（板卡）要装回夹爪、工位摆好，再用采集窗口。")
        return 0
    finally:
        try:
            hw.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
