# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""采集数据可视化 —— 读一条 episode，画成一张 PNG。

依赖：``pyarrow`` + ``pillow`` + ``matplotlib`` + ``numpy``（**不需要 lerobot**）
⇒ 本地 Mac（看图方便）和服务器（数据所在地）都能跑。

用法::

    python -m real_robot_collect.visualize --dataset <数据集目录>
    python -m real_robot_collect.visualize --dataset <数据集目录> --episode 1 --out /tmp/ep1.png

输出默认落在数据集目录的 ``../<数据集名>_ep<k>.png``。
"""

from __future__ import annotations

import argparse
import glob
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 读取（只用 pyarrow，避开 lerobot）
# ---------------------------------------------------------------------------


def _parquet_files(root: Path) -> List[str]:
    return sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))


def _decode_image(cell: Any) -> Optional[np.ndarray]:
    """HF ``datasets`` 的 Image 列存的是 ``{'bytes': ..., 'path': ...}``。"""
    from PIL import Image

    blob = path = None
    if isinstance(cell, dict):
        blob, path = cell.get("bytes"), cell.get("path")
    elif isinstance(cell, (bytes, bytearray)):
        blob = cell
    if blob:
        return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    if path and Path(path).exists():
        return np.asarray(Image.open(path).convert("RGB"))
    return None


def _task_text(root: Path, task_index: Any) -> str:
    """取任务文本。

    ⚠️ 两处容易找错：
      1. ``task`` 字符串**不在** ``data/*.parquet``（那里只有 ``task_index``），
         而在 ``meta/tasks.parquet``（与实测 ``hs_robot_assembly_v1`` 一致）；
      2. ``tasks.parquet`` 里**文本是 pandas index** —— pyarrow 读出来是一列
         ``__index_level_0__``，而 ``task_index`` 列里只有编号（实测：``[0]``）。
    """
    p = root / "meta" / "tasks.parquet"
    if not p.is_file():
        return ""
    try:
        import pyarrow.parquet as pq

        tb = pq.read_table(p)
        name = next((c for c in ("__index_level_0__", "task", "index")
                     if c in tb.column_names), None)
        if name is None:
            cands = [c for c in tb.column_names if c != "task_index"]
            if not cands:
                return ""
            name = cands[0]
        texts = tb.column(name).to_pylist()
        i = int(task_index)
        return str(texts[i]) if 0 <= i < len(texts) else ""
    except Exception:
        return ""


def load_episode(root: Path, ep_index: int = 0) -> Dict[str, Any]:
    """读一条 episode 的全部帧（不依赖 lerobot）。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = _parquet_files(root)
    if not files:
        raise FileNotFoundError(f"{root} 下没有 data/**/*.parquet")

    tb = pa.concat_tables([pq.read_table(f) for f in files])
    ei = np.asarray(tb["episode_index"]).ravel()
    rows = np.where(ei == ep_index)[0]
    if rows.size == 0:
        avail = sorted(set(int(x) for x in np.unique(ei)))
        raise ValueError(f"episode_index={ep_index} 不存在；可用的有 {avail[:20]}…（共 {len(avail)} 条）")

    def col(name: str):
        return tb[name].to_pylist()

    names = set(tb.column_names)
    img_keys = sorted(c for c in tb.column_names if c.startswith("observation.images."))

    out: Dict[str, Any] = {
        "episode_index": ep_index,
        "frames": len(rows),
        "timestamp": np.asarray(
            [col("timestamp")[i] for i in rows], dtype=np.float64
        ) if "timestamp" in names else np.arange(len(rows), dtype=np.float64),
        "state": np.asarray([col("observation.state")[i] for i in rows], dtype=np.float64),
        "action": np.asarray([col("action")[i] for i in rows], dtype=np.float64),
        "images": {k: [_decode_image(col(k)[i]) for i in rows] for k in img_keys},
    }
    if "task" in names:
        out["task"] = col("task")[rows[0]]
    elif "task_index" in names:
        out["task"] = _task_text(root, col("task_index")[rows[0]])
    return out


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------


def _style() -> None:
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.facecolor"] = "white"


def plot(ep: Dict[str, Any], out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    _style()
    n = ep["frames"]
    idx = sorted({0, n // 2, n - 1})           # 首 / 中 / 尾
    cams = list(ep["images"].keys())

    fig = plt.figure(figsize=(16, 10))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.28)

    # ---- 上半：每路相机 3 个时刻 ----
    top = outer[0].subgridspec(len(cams), len(idx), hspace=0.06, wspace=0.04)
    for r, cam in enumerate(cams):
        frames = ep["images"][cam]
        for c, i in enumerate(idx):
            ax = fig.add_subplot(top[r, c])
            img = frames[i]
            if img is None:
                ax.text(0.5, 0.5, "(无图像)", ha="center", va="center")
            else:
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"frame {i}   t={ep['timestamp'][i]:.2f}s", fontsize=10)
            if c == 0:
                ax.set_ylabel(cam.replace("observation.images.", ""),
                              fontsize=11, fontweight="bold")

    # ---- 下半：state / action 曲线（4 个并排子图）----
    bottom = outer[1].subgridspec(1, 4, wspace=0.30)
    t = ep["timestamp"]
    st, ac = ep["state"], ep["action"]

    panels: List[Tuple[str, np.ndarray, List[str]]] = []
    D = st.shape[1]
    panels.append(("state[0:3] 位置 (mm)", st[:, 0:3], ["x", "y", "z"]))
    if D >= 6:
        panels.append(("state[3:6] 姿态 (deg)", st[:, 3:6], ["rx", "ry", "rz"]))
    if D >= 12:
        panels.append(("state[6:12] 力/力矩", st[:, 6:12],
                       ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]))
    panels.append((f"action[0:{ac.shape[1]}]", ac,
                   ["Δx", "Δy", "Δz"][: ac.shape[1]]))

    for c, (label, arr, cols) in enumerate(panels[:4]):
        ax = fig.add_subplot(bottom[c])
        for j in range(arr.shape[1]):
            ax.plot(t, arr[:, j], lw=1.2,
                    label=cols[j] if j < len(cols) else f"[{j}]")
        # 全 0 的维（例如只读模式的 action）标出来
        zero_cols = [cols[j] for j in range(arr.shape[1])
                     if j < len(cols) and np.allclose(arr[:, j], 0.0)]
        if zero_cols and len(zero_cols) == arr.shape[1]:
            ax.plot([], [], " ", label="(全为 0)")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("t (s)", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        ax.tick_params(labelsize=8)

    task = ep.get("task", "")
    fig.suptitle(f"{title}\nepisode {ep['episode_index']} · {n} 帧 · task={task!r}",
                 fontsize=13, y=0.985)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 已输出 {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="real_robot_collect.visualize")
    ap.add_argument("--dataset", required=True, help="数据集目录")
    ap.add_argument("--episode", type=int, default=0, help="episode_index（默认 0）")
    ap.add_argument("--out", default=None, help="输出 PNG 路径")
    a = ap.parse_args(argv)

    root = Path(a.dataset).expanduser().resolve()
    ep = load_episode(root, a.episode)

    out = Path(a.out).expanduser() if a.out else root.parent / f"{root.name}_ep{a.episode}.png"
    info = {}
    try:
        info = json.loads((root / "meta" / "info.json").read_text())
    except Exception:
        pass
    title = f"{root.name}   state={tuple(ep['state'].shape)}  action={tuple(ep['action'].shape)}"
    if info:
        title += f"   fps={info.get('fps')}"

    plot(ep, out, title)
    print(f"   {ep['frames']} 帧 · 相机 {list(ep['images'].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
