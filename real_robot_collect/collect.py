# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""独立采集入口。

用法（在**边缘端**、fork 的仓库根目录下）::

    # 0) 先做环境自检，不碰硬件
    python -m real_robot_collect.collect --selftest --config configs/peg_v1.json

    # 1) 只看配置解析 + 数据集目录能否创建（会真的建目录，随后删掉）
    python -m real_robot_collect.collect --dry-run --config configs/peg_v1.json

    # 2) 正式采集
    python -m real_robot_collect.collect --config configs/peg_v1.json

退出码：0 = 正常结束；2 = 配置/环境错误；3 = 采集过程被 Ctrl+C 中断（数据已保留）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from .config import CollectConfig, dump_config, load_config
from .features import build_features


def _setup_logging(verbose: bool, log_file: Path | None) -> None:
    fmt = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=fmt, handlers=handlers)


def _dataset_root(cfg: CollectConfig) -> Path:
    name = cfg.dataset.repo_id
    if cfg.dataset.timestamp_suffix:
        name = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    return cfg.dataset_root_abs() / name


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------


def cmd_selftest(cfg: CollectConfig) -> int:
    """只 import 驱动 + 打印将生效的配置。不碰硬件、不建目录。"""
    from .hw import selftest_imports

    failures = selftest_imports()
    if failures:
        print("❌ import 自检失败：")
        for f in failures:
            print("   -", f)
        print("\n提示：请在 fork 的仓库根目录、用 fork 自己的 python 运行；")
        print("      若 hs_robot 的 .so 缺失，见 fork 的 CLAUDE.md（需手动放置，不入库）。")
        return 2

    print("✅ 所有 lerobot 驱动 import 通过")
    feats = build_features(cfg.camera_keys, cfg.record.image_size, cfg.state_dim, cfg.action_dim,
                           cfg.record.use_videos)
    print("\n将声明的 features（数据集里只会出现这些键 + task）：")
    for k, v in feats.items():
        print(f"   {k:<34} dtype={v['dtype']:<7} shape={tuple(v['shape'])}")
    print("\n派生维度：state_dim=%d  action_dim=%d" % (cfg.state_dim, cfg.action_dim))
    print("相机（顺序 = 写入 info.json 的顺序）：", cfg.camera_keys)
    print("   ⚠️ LoongForge 读图像时按 key 的**字典序**决定视野槽位，")
    print("      所以训练/执行两侧的相机命名必须完全一致。")
    print("\n数据集将落在：", _dataset_root(cfg))
    print("磁盘可用：")
    os.system(f"df -h {cfg.dataset_root_abs().anchor or '/'} | tail -1")
    return 0


def cmd_dry_run(cfg: CollectConfig) -> int:
    """建一个空数据集再删掉，验证 features / 目录 / 磁盘权限。不需要硬件。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = _dataset_root(cfg)
    feats = build_features(cfg.camera_keys, cfg.record.image_size, cfg.state_dim, cfg.action_dim,
                           cfg.record.use_videos)
    if root.exists():
        print(f"❌ 目标目录已存在，先改名或删除：{root}")
        return 2

    print(f"尝试创建数据集：{root}")
    try:
        ds = LeRobotDataset.create(
            repo_id=root.name,
            fps=int(cfg.record.fps),
            features=feats,
            root=str(root),
            robot_type=cfg.dataset.robot_type,
            use_videos=bool(cfg.record.use_videos),
            image_writer_processes=0,
            image_writer_threads=0,   # dry-run 不写图
        )
    except Exception as exc:
        print(f"❌ 创建失败：{type(exc).__name__}: {exc}")
        return 2

    info = json.loads((root / "meta" / "info.json").read_text())
    print("✅ 创建成功，info.json 关键字段：")
    print("   codebase_version =", info["codebase_version"])
    print("   fps              =", info["fps"])
    print("   robot_type       =", info["robot_type"])
    print("   video_path       =", info["video_path"], "（None ⇒ 图像以 PNG 内嵌 parquet）")

    # 造一帧假数据，验证 validate_frame 的严格校验确实放行我们的键集合
    import numpy as np

    frame = {
        **{f"observation.images.{c}": np.zeros((cfg.record.image_size, cfg.record.image_size, 3), np.uint8)
           for c in cfg.camera_keys},
        "observation.state": np.zeros((cfg.state_dim,), np.float32),
        "action": np.zeros((cfg.action_dim,), np.float32),
        "task": cfg.dataset.task,
    }
    try:
        ds.add_frame(frame)
        ds.clear_episode_buffer()
        print("✅ 单帧 add_frame 校验通过（键集合与 features 完全匹配）")
    except Exception as exc:
        print(f"❌ add_frame 校验失败：{type(exc).__name__}: {exc}")
        shutil.rmtree(root, ignore_errors=True)
        return 2

    # 先 finalize（关 parquet writer），再删目录 —— 否则 writer 句柄占着，可能删不干净
    from .recorder import finalize_dataset

    finalize_dataset(ds)
    shutil.rmtree(root, ignore_errors=True)
    print(f"（已清理临时目录 {root}）")
    print("\n下一步：加 --config 不带 --dry-run 正式采集。")
    return 0


def cmd_record(
    cfg: CollectConfig,
    read_only_episodes: int | None = None,
    episodes: int | None = None,
) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .hw import Hardware
    from .recorder import finalize_dataset, run_read_only, run_recording

    if episodes is not None:
        if int(episodes) <= 0:
            print(f"❌ --episodes 必须为正，收到 {episodes}", file=sys.stderr)
            return 2
        cfg.dataset.num_episodes = int(episodes)
        print(f"（CLI 覆盖 dataset.num_episodes = {episodes}，只影响本次运行）")

    if read_only_episodes:
        # 只读数据的 action 全 0（不可训练）⇒ 加后缀，防止和正式数据混淆
        if not cfg.dataset.repo_id.endswith("_readonly"):
            cfg.dataset.repo_id = f"{cfg.dataset.repo_id}_readonly"

    root = _dataset_root(cfg)
    if root.exists():
        print(f"❌ 目标目录已存在：{root}\n   换个 repo_id，或先移走旧数据。")
        return 2

    feats = build_features(cfg.camera_keys, cfg.record.image_size, cfg.state_dim, cfg.action_dim,
                           cfg.record.use_videos)

    ds = LeRobotDataset.create(
        repo_id=root.name,
        fps=int(cfg.record.fps),
        features=feats,
        root=str(root),
        robot_type=cfg.dataset.robot_type,
        use_videos=bool(cfg.record.use_videos),
        image_writer_processes=0,
        image_writer_threads=int(cfg.record.image_writer_threads),
    )
    print(f"📁 数据集目录：{root}")

    # 把生效配置 + 说明一起归档，便于日后复现
    dump_config(cfg, root / "collect_config.effective.json")
    (root / "README_collect.md").write_text(
        "# 采集说明\n\n"
        + ("- ⚠️ **只读模式（READ-ONLY）**：全程未发控制指令、未复位；"
           "action 全 0 ⇒ **仅用于链路验证，不可用于训练**\n"
           if read_only_episodes else "")
        + f"- 采集时间：{time.strftime('%F %T')}\n"
        f"- 原始配置：{cfg.source_path}\n"
        f"- 相机键（顺序=写入 info.json 顺序）：{cfg.camera_keys}\n"
        f"- 相机键**字典序**（= LoongForge 的视野槽位顺序）：{sorted(cfg.camera_keys)}\n"
        f"- state_dim={cfg.state_dim} · action_dim={cfg.action_dim}\n"
        f"- task（进 pi05 prompt）：{cfg.dataset.task!r}\n"
        f"- 只落成功条：keep_failed={cfg.record.keep_failed}\n"
        f"- ⚠️ 本次采集的**随机化程度**（人摆件的策略）请手填：______\n",
        encoding="utf-8",
    )

    hw = Hardware(cfg)
    rc = 0
    try:
        # 只读模式不连 SpaceMouse / 键盘：少一个设备占用，也彻底杜绝误发控制指令
        hw.connect(connect_teleop=not read_only_episodes)
        if read_only_episodes:
            summary = run_read_only(cfg, hw, ds, feats, num_episodes=int(read_only_episodes))
        else:
            summary = run_recording(cfg, hw, ds, feats)
        print("\n" + summary.table())
        (root / "collect_summary.txt").write_text(summary.table() + "\n", encoding="utf-8")
    except KeyboardInterrupt:
        print("\n⚠️ 被中断。已完成并落盘的 episode 都在数据集里。")
        rc = 3
    finally:
        try:
            hw.close()
        finally:
            finalize_dataset(ds)

    print("\n下一步验收：")
    print(f"   python -m real_robot_collect.verify --dataset {root}")
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="real_robot_collect",
        description="独立 LeRobot v3.0 采集模块（不依赖 HIL-SERL / LoongForge）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="采集配置 JSON 路径")
    p.add_argument("--selftest", action="store_true", help="只做 import 自检并打印生效配置，不碰硬件")
    p.add_argument("--dry-run", action="store_true", help="空数据集 + 假帧校验，不碰硬件")
    p.add_argument("--read-only", nargs="?", type=int, const=3, default=None, metavar="N",
                   help="只读模式：采 N 条（默认 3）固定时长数据；全程**不发控制指令、不复位**，"
                        "连不上 SpaceMouse/键盘，action 全为 0 ⇒ 仅验证链路，**不可用于训练**")
    p.add_argument("--episodes", type=int, default=None, metavar="N",
                   help="覆盖 dataset.num_episodes（本参数只改这一次运行，不动 JSON）。"
                        "冒烟用 --episodes 5；正式不传即用配置值")
    p.add_argument("--verbose", action="store_true", help="DEBUG 日志（每帧都会打印，很吵）")
    p.add_argument("--log-file", default=None, help="额外把日志写进文件")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"❌ 配置错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    _setup_logging(args.verbose, Path(args.log_file) if args.log_file else None)

    if args.selftest:
        return cmd_selftest(cfg)
    if args.dry_run:
        return cmd_dry_run(cfg)
    return cmd_record(cfg, read_only_episodes=args.read_only, episodes=args.episodes)


if __name__ == "__main__":
    raise SystemExit(main())
