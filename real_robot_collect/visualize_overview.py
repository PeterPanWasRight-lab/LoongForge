"""多条 episode 合览图 —— 一行一条，列为：front 相机 / wrist 相机 / 末端轨迹 / action。

用途：一眼判断一批演示的质量
  * 视觉：机械臂在做什么、视野对不对
  * 末端轨迹：走了多远、是否每次形态相似
  * action：是否真有动作（全 0 = 只读模式 / 没记录到人操作）
  * 时长/帧数：是否有异常短或异常长的条目

不依赖 lerobot（pyarrow + PIL + matplotlib），本地 Mac 也能跑。

用法::

    python -m real_robot_collect.visualize_overview --dataset datasets/<名>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from .visualize import _decode_image, _parquet_files, _style  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    _style()  # 中文字体（否则中文标题会变成方块）
    p = argparse.ArgumentParser(description="多条 episode 合览图")
    p.add_argument("--dataset", required=True, help="数据集目录")
    p.add_argument("--out", default=None, help="输出 PNG（默认 <数据集名>_overview.png）")
    p.add_argument("--dpi", type=int, default=105)
    args = p.parse_args(argv)

    root = Path(args.dataset)
    files = _parquet_files(root)
    if not files:
        print(f"❌ {root} 下没有 data/**/*.parquet")
        return 2

    tb = pa.concat_tables([pq.read_table(f) for f in files])
    ei = np.asarray(tb["episode_index"]).ravel().astype(int)
    st = np.asarray(tb["observation.state"].to_pylist(), dtype=np.float64)
    ac = np.asarray(tb["action"].to_pylist(), dtype=np.float64)
    ts = np.asarray(tb["timestamp"].to_pylist(), dtype=np.float64)
    front = tb["observation.images.front"].to_pylist() if "observation.images.front" in tb.column_names else None
    wrist = tb["observation.images.wrist"].to_pylist() if "observation.images.wrist" in tb.column_names else None

    eps = sorted(set(int(x) for x in np.unique(ei)))
    n = len(eps)
    ncol = 4
    fig, axes = plt.subplots(n, ncol, figsize=(4.0 * ncol, 2.5 * n), squeeze=False)

    for r, e in enumerate(eps):
        idx = np.where(ei == e)[0]
        mid = idx[len(idx) // 2]
        t = ts[idx] - ts[idx[0]]

        # ── 两路相机（取中段帧）──
        for c, (cells, name) in enumerate(((front, "front"), (wrist, "wrist"))):
            ax = axes[r][c]
            img = _decode_image(cells[mid]) if cells is not None else None
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(.5, .5, "无图像", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"ep{e}\n{len(idx)} 帧 / {t[-1]:.1f} s", fontsize=9)

        # ── 末端位置（绝对，mm）──
        ax = axes[r][2]
        for j, nm in enumerate("xyz"):
            ax.plot(t, st[idx, j], lw=1.2, label=f"{nm}")
        ax.grid(alpha=.3)
        if r == 0:
            ax.set_title("末端位置（mm，绝对）", fontsize=10)
            ax.legend(fontsize=7, ncol=3, loc="best")

        # ── action（相对增量，mm/帧）──
        ax = axes[r][3]
        for j, nm in enumerate("xyz"):
            ax.plot(t, ac[idx, j], lw=1.2, label=f"Δ{nm}")
        ax.axhline(0, color="k", lw=.5, alpha=.4)
        ax.grid(alpha=.3)
        if r == 0:
            ax.set_title("action（mm/帧，相对）", fontsize=10)
            ax.legend(fontsize=7, ncol=3, loc="best")

    # 一句话统计
    summary = " | ".join(
        f"ep{e}: {int((ei == e).sum())}f" for e in eps
    )
    fig.suptitle(f"{root.name}  —  {n} 条 · {summary}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    out = Path(args.out) if args.out else root.parent / f"{root.name}_overview.png"
    fig.savefig(out, dpi=args.dpi)
    print("written:", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
