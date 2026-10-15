# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""采集后验收 —— 在**任何**机器上都能跑（只需要 pyarrow + 标准库）。

检查清单（对应 LoongForge 训练侧的硬约束）::

    ① 版本与维度    codebase_version == v3.0；action/state 维度与配置一致
    ② 帧数一致      info.json 的 total_frames == 所有 parquet 的行数之和
    ③ stats 可用    action / observation.state 必须有 q01 与 q99，且 q99 > q01
                    —— LoongForge **不重算**统计量，直接读 meta/stats.json
                       （``data/dataloader.py:112 → dataset.meta.stats``）
                       错了训练侧既不报错也不纠正 ⇒ 静默学错
    ④ 逐条体检      每条的帧数与时长；异常短的条要人工看一眼
    ⑤ RL 遗留字段   **应当为空**（本模块只声明 pi05 需要的字段）
    ⑥ 端到端可读    用官方 ``LeRobotDataset`` 真加载一次（= LoongForge 的读法）。
                    失败 ⇒ 多半是采集结束时漏了 ``finalize()``，parquet footer 没写

用法::

    python -m real_robot_collect.verify --dataset datasets/hs_robot_peg_v1_20261013_150000
    # 也可在训练机上传完后跑同样一条命令
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REQUIRED_STAT_KEYS = ("q01", "q99")
#: 出现这些键说明走的不是本模块（而是 gym_manipulator 的 RL 路径）
RL_LEFTOVER_HINTS = ("next.reward", "next.done", "next.truncated", "complementary_info")


def _read_info(root: Path) -> Dict[str, Any]:
    return json.loads((root / "meta" / "info.json").read_text())


def _parquet_rows(root: Path) -> int:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def _episode_lengths(root: Path) -> List[Tuple[int, int]]:
    """返回 [(episode_index, 帧数), ...]。"""
    import numpy as np
    import pyarrow.parquet as pq

    counts: Dict[int, int] = {}
    for f in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
        t = pq.read_table(f, columns=["episode_index"])
        e = np.asarray(t["episode_index"]).ravel()
        for ep in np.unique(e):
            counts[int(ep)] = counts.get(int(ep), 0) + int((e == ep).sum())
    return sorted(counts.items())


def verify(root: Path) -> int:
    problems: List[str] = []
    warnings: List[str] = []

    if not root.is_dir():
        print(f"❌ 数据集目录不存在: {root}")
        return 2

    info = _read_info(root)
    feats = info.get("features", {})

    # ---------- ① 版本与维度 ----------
    print("=" * 74)
    print(f"数据集: {root}")
    print("=" * 74)
    print(f"[①] codebase_version = {info.get('codebase_version')}   fps = {info.get('fps')}   "
          f"episodes = {info.get('total_episodes')}")
    if info.get("codebase_version") != "v3.0":
        problems.append(f"codebase_version 不是 v3.0：{info.get('codebase_version')}")

    print("\n     features:")
    for k, v in feats.items():
        print(f"       {k:<34} {v['dtype']:<7} {tuple(v['shape'])}")
        if k.startswith("observation.images.") and v["dtype"] not in ("image", "video"):
            problems.append(f"{k} 的 dtype 异常：{v['dtype']}")

    img_keys = [k for k in feats if k.startswith("observation.images.")]
    cam_order = sorted(k[len("observation.images."):] for k in img_keys)
    print(f"\n     相机 {len(img_keys)} 路；**字典序**（= LoongForge 视野槽位顺序）= {cam_order}")
    if len(img_keys) == 1:
        warnings.append("只有 1 路图像；pi05 的 data.num_images 默认 2，多余槽位会用 -1.0 占位 + mask=False")

    # ---------- ⑤ RL 遗留字段 ----------
    leftovers = [k for k in feats for h in RL_LEFTOVER_HINTS if k.startswith(h)]
    if leftovers:
        warnings.append(f"检测到 RL 遗留字段 {leftovers}（本模块不应产生；对 pi05 无害但说明用的不是本模块）")
    else:
        print("\n[⑤] 无 RL 遗留字段 ✅（只有 pi05 需要的 observation.*/action）")

    # ---------- ② 帧数一致 ----------
    rows = _parquet_rows(root)
    total = int(info.get("total_frames", -1))
    print(f"\n[②] info.total_frames = {total}   parquet rows = {rows}   "
          f"{'✅' if total == rows else '❌ 不一致'}")
    if total != rows:
        problems.append(f"total_frames({total}) != parquet rows({rows})：文件传坏或写入中断")

    # ---------- ③ stats ----------
    stats_path = root / "meta" / "stats.json"
    print(f"\n[③] stats: {stats_path.name}")
    if not stats_path.is_file():
        problems.append("meta/stats.json 不存在 —— LoongForge 会拿不到归一化统计量")
    else:
        st = json.loads(stats_path.read_text())
        for key in ("action", "observation.state"):
            if key not in st:
                problems.append(f"stats 缺少 '{key}'")
                continue
            entry = st[key]
            miss = [s for s in REQUIRED_STAT_KEYS if s not in entry]
            if miss:
                problems.append(f"stats['{key}'] 缺少 {miss}")
                continue
            q01 = list(entry["q01"])
            q99 = list(entry["q99"])
            print(f"       {key:<20} dim={len(q01)}")
            print(f"         q01 = {[round(float(x), 4) for x in q01]}")
            print(f"         q99 = {[round(float(x), 4) for x in q99]}")
            viol = [i for i, (a, b) in enumerate(zip(q01, q99)) if not (b > a)]
            if viol:
                problems.append(
                    f"stats['{key}'] 第 {viol} 维 q99 <= q01。"
                    "常见原因：该维在整段数据里恒定（例如夹爪全程不动）。"
                    "q99 归一化的分母会退化成 eps=1e-8 ⇒ 该维归一化后是垃圾。"
                )
            const_dims = [i for i, (a, b) in enumerate(zip(q01, q99)) if abs(b - a) < 1e-9]
            if const_dims:
                warnings.append(f"stats['{key}'] 第 {const_dims} 维恒定（q99==q01），进 prompt 后无信息")
            # 量纲粗检：位置维通常 mm 级（几十~几百），全 0 说明没读到数据
            if all(abs(float(x)) < 1e-6 for x in q01) and all(abs(float(x)) < 1e-6 for x in q99):
                problems.append(f"stats['{key}'] 全 0 —— 采集时很可能没读到真实数据")

    # ---------- ④ 逐条体检 ----------
    eps = _episode_lengths(root)
    print(f"\n[④] 逐条体检（共 {len(eps)} 条）")
    if eps:
        lens = [n for _, n in eps]
        srt = sorted(lens)
        med = srt[len(srt) // 2]
        print(f"       帧数 min/median/max = {srt[0]} / {med} / {srt[-1]}")
        fps = float(info.get("fps", 0)) or 1.0
        print(f"       时长 min/median/max = {srt[0]/fps:.1f}s / {med/fps:.1f}s / {srt[-1]/fps:.1f}s")
        short = [(e, n) for e, n in eps if n < max(3, int(0.25 * med))]
        if short:
            warnings.append(
                f"有 {len(short)} 条明显短于中位数（可能是操作失误或误触 s）：{short[:10]}"
            )
        print(f"       逐条帧数: {lens}")

    # ---------- ⑥ 端到端：能否被 LeRobotDataset 加载 ----------
    #   这是 LoongForge 训练侧的**真实读法**（它直接 import 官方 LeRobotDataset）。
    #   能加载 ⇒ parquet footer 与 meta 完整（即采集结束时 finalize 生效）。
    #   不调 finalize 时，parquet 只写了数据块没写 footer，这里会直接失败。
    print("\n[⑥] 端到端：LeRobotDataset 能否加载")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        print(f"       (跳过) 当前环境没有 lerobot（{type(exc).__name__}）；请在训练机上复核这一项")
    else:
        try:
            ds = LeRobotDataset(repo_id=str(info.get("repo_id") or root.name), root=str(root))
            print(f"       ✅ 加载成功：{ds.num_episodes} 条 / {ds.num_frames} 帧")
            sample = ds[0]
            keys = sorted(sample.keys())
            print(f"       第 0 帧键（{len(keys)} 个）: {keys[:8]}{' ...' if len(keys) > 8 else ''}")
        except Exception as exc:
            problems.append(
                f"LeRobotDataset 加载失败：{type(exc).__name__}: {exc} —— "
                "最常见原因是采集结束时没有调用 dataset.finalize()（parquet footer 未写）"
            )

    # ---------- 训练侧参数提示 ----------
    frames = rows
    print("\n[训练参数提示]（LoongForge）")
    for bs in (32, 64):
        if frames:
            print(f"       bs={bs} 单卡: 1 epoch ≈ {frames // bs} 步")
    print("       ⚠️ 别照抄 LIBERO 线的 16000 步（那是配 101,469 帧的）。")
    print("          建议按 epoch 反推：steps = epochs × 总帧数 / (bs × world_size)")

    # ---------- 结论 ----------
    print("\n" + "=" * 74)
    if problems:
        print(f"❌ 发现 {len(problems)} 个**必须修**的问题：")
        for i, p in enumerate(problems, 1):
            print(f"   {i}. {p}")
    else:
        print("✅ 没有发现必须修的问题")
    if warnings:
        print(f"\n⚠️ {len(warnings)} 条提示（不一定阻塞，但要知情）：")
        for i, w in enumerate(warnings, 1):
            print(f"   {i}. {w}")
    print("=" * 74)
    return 2 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="real_robot_collect.verify")
    ap.add_argument("--dataset", required=True, help="数据集目录")
    a = ap.parse_args(argv)
    return verify(Path(a.dataset).expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
