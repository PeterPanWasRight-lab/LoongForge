"""运动学一致性检验 —— 用两条独立证据验证 action 与 state / 腕部图像是否自洽。

**检验 A（数值）**：action 是相对增量（`hs_robot.py:524` `target = cached_target + delta`）
⇒ 稳态下应有 ``state[t+1] - state[t] ≈ action[t]``。
控制器有跟踪滞后，所以扫描 lag=0..4 找最佳匹配，并给出残差与相关系数。

**检验 B（视觉）**：腕部相机装在末端（eye-in-hand），末端平动会让图像反向平移。
用相位相关估计帧间整数平移 (dy, dx)，与 action 的 (Δx, Δy, Δz) 求相关矩阵。
    * 若 |corr| 高且方向关系明确 ⇒ 视觉与动作自洽
    * Δz 是深度方向，表现为图像缩放而非平移，故预期与平移相关性弱（是正常现象）

⚠️ 两个检验都**不是在验证绝对正确性**，而是验证"两份独立记录描述了同一件事"。
   它们同时成立，才能排除"state 读了 A 通道、action 写了 B 通道"这类静默错配。

用法::

    python -m real_robot_collect.verify_kinematics --dataset datasets/<名> [--out 图.png]
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


def _gray(cell) -> np.ndarray | None:
    img = _decode_image(cell)
    if img is None:
        return None
    g = img.astype(np.float32)
    if g.ndim == 3:
        g = g @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return g


def _hann(shape) -> np.ndarray:
    wy = np.hanning(shape[0])[:, None]
    wx = np.hanning(shape[1])[None, :]
    return (wy * wx).astype(np.float32)


def phase_shift(a: np.ndarray, b: np.ndarray, win: np.ndarray) -> tuple[float, float]:
    """**亚像素**相位相关：估计 b 相对 a 的平移 (dy, dx)，单位像素。

    帧间位移常常不到 1 px（实测 |dy|≈0.3 px），整数峰值误差太大，
    故在峰值处做抛物线插值取亚像素。
    """
    A = np.fft.fft2((a - a.mean()) * win)
    B = np.fft.fft2((b - b.mean()) * win)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    r = np.fft.ifft2(R).real
    ny, nx = r.shape
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)

    def parab(v0: float, v1: float, v2: float) -> float:
        d = v0 - 2.0 * v1 + v2
        if abs(d) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (v0 - v2) / d, -1.0, 1.0))

    dy = iy + parab(r[(iy - 1) % ny, ix], r[iy, ix], r[(iy + 1) % ny, ix])
    dx = ix + parab(r[iy, (ix - 1) % nx], r[iy, ix], r[iy, (ix + 1) % nx])
    if dy > ny / 2:
        dy -= ny
    if dx > nx / 2:
        dx -= nx
    return float(dy), float(dx)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main(argv: list[str] | None = None) -> int:
    _style()
    p = argparse.ArgumentParser(description="运动学一致性检验（action ↔ state / 腕部图像）")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default=None, help="诊断图输出路径")
    p.add_argument("--max-lag", type=int, default=4)
    args = p.parse_args(argv)

    root = Path(args.dataset)
    files = _parquet_files(root)
    tb = pa.concat_tables([pq.read_table(f) for f in files])
    ei = np.asarray(tb["episode_index"]).ravel().astype(int)
    st = np.asarray(tb["observation.state"].to_pylist(), dtype=np.float64)
    ac = np.asarray(tb["action"].to_pylist(), dtype=np.float64)
    wrist_col = "observation.images.wrist"
    wrist = tb[wrist_col].to_pylist() if wrist_col in tb.column_names else None
    eps = sorted(set(int(x) for x in np.unique(ei)))

    # ================= 检验 A：state[t+1] - state[t] ≈ action[t] =================
    print("=" * 74)
    print("检验 A：action 与 state 差分的一致性   state[t+1]-state[t]  vs  action[t-lag]")
    print("=" * 74)

    ds_list, ac_list = [], []
    for e in eps:
        idx = np.where(ei == e)[0]
        ds_list.append(np.diff(st[idx][:, :3], axis=0))
        ac_list.append(ac[idx][:-1])
    dstate = np.concatenate(ds_list)          # (N, 3)
    act = np.concatenate(ac_list)             # (N, 3)

    print(f"  样本数 {len(dstate)}（相邻帧对）")
    print()
    print("  lag  |  corr(Δx,Δx) corr(Δy,Δy) corr(Δz,Δz) |  残差 RMS (mm)")
    best = None
    for lag in range(args.max_lag + 1):
        if lag == 0:
            d, a = dstate, act
        else:
            d, a = dstate[lag:], act[:-lag]
        c = [_corr(d[:, j], a[:, j]) for j in range(3)]
        rms = float(np.sqrt(np.mean((d - a) ** 2)))
        marker = ""
        if best is None or np.mean(np.abs(c)) > best[1]:
            best = (lag, float(np.mean(np.abs(c))), c, rms)
            marker = "  ←"
        print(f"   {lag:>2}  |    {c[0]:>7.3f}     {c[1]:>7.3f}     {c[2]:>7.3f}  |   {rms:>8.3f}{marker}")

    lag, score, cc, rms0 = best
    print(f"\n  ⇒ 最佳 lag = {lag}，平均 |corr| = {score:.3f}，残差 RMS = {rms0:.3f} mm")
    print(f"     action 标准差 = {np.round(act.std(0), 3)} mm  ⇒ 残差/信号 = "
          f"{np.round(rms0 / max(act.std(), 1e-9), 3)}")
    verdict_a = score > 0.7
    print(f"     {'✅ 通过' if verdict_a else '❌ 未通过'}（阈值 |corr| > 0.7）")

    # ================= 检验 B：腕部图像平移 vs action =================
    print()
    print("=" * 74)
    print("检验 B：腕部图像帧间平移 vs action（eye-in-hand，末端动则图像反向平移）")
    print("=" * 74)

    if wrist is None:
        print("  ⚠️ 数据集里没有腕部相机，跳过检验 B")
        shifts = actB = mat = None
        verdict_b = False
    else:
        # 帧间位移常常 < 1 px（信噪比低）⇒ 用「累积基线」放大信号：
        # 比较 t 与 t+base 的图像位移 vs 同期 action 之和，扫 base = 1,2,3,5,10。
        per_ep = {}
        for e in eps:
            idx = np.where(ei == e)[0]
            g = [_gray(wrist[i]) for i in idx]
            per_ep[e] = (idx, [x for x in g if x is not None])

        names = ["dy_img", "dx_img"]
        cols = ["Δx_arm", "Δy_arm", "Δz_arm"]
        bestB = None
        for base in (1, 2, 3, 5, 10):
            all_shifts, all_act = [], []
            for e in eps:
                idx, grays = per_ep[e]
                if len(grays) < base + 2:
                    continue
                win = _hann(grays[0].shape)
                for i in range(len(grays) - base):
                    all_shifts.append(phase_shift(grays[i], grays[i + base], win))
                    all_act.append(ac[idx][i:i + base].sum(0))
            if not all_shifts:
                continue
            sh = np.array(all_shifts)
            aa = np.array(all_act)
            m = np.array([[_corr(sh[:, i], aa[:, j]) for j in range(3)] for i in range(2)])
            flat = sorted(((abs(m[i, j]), i, j, m[i, j]) for i in range(2) for j in range(3)),
                          reverse=True)
            top = flat[0]
            print(f"\n  基线 base={base:>2} · {len(sh)} 样本 · "
                  f"|dy|={np.abs(sh[:, 0]).mean():.2f} |dx|={np.abs(sh[:, 1]).mean():.2f} px")
            print("               " + "".join(f"{c:>10}" for c in cols))
            for i, nm in enumerate(names):
                print(f"    {nm:<9}" + "".join(f"{v:>10.3f}" for v in m[i]))
            print(f"    最强：{names[top[1]]} ↔ {cols[top[2]]}  corr = {top[3]:+.3f}")
            if bestB is None or top[0] > bestB[0]:
                bestB = (top[0], base, names[top[1]], cols[top[2]], top[3], m, sh, aa)

        if bestB is None:
            print("  ⚠️ 样本不足，跳过")
            shifts = actB = mat = None
            verdict_b = False
        else:
            score_b, base_b, n1, n2, c_b, mat, shifts, actB = bestB
            print(f"\n  ⇒ 最佳基线 base={base_b}：{n1} ↔ {n2}   corr = {c_b:+.3f}")
            verdict_b = score_b > 0.6
            print(f"     {'✅ 通过' if verdict_b else '❌ 未通过'}（阈值 |corr| > 0.6）")
            print("     Δz（竖直）在俯视相机里表现为图像**缩放**而非平移，")
            print("     故与平移列相关弱属正常；真正该对上的是 Δx/Δy ↔ 图像平移。")
            # 单独看"平移 ↔ 平移"最该成立的两对
            pair = [("dy_img", "Δy_arm", _corr(shifts[:, 0], actB[:, 1])),
                    ("dx_img", "Δx_arm", _corr(shifts[:, 1], actB[:, 0]))]
            print("\n     定向检查（相机坐标 ↔ 机械臂坐标）：")
            for a, b, c in pair:
                print(f"       {a} ↔ {b}: corr = {c:+.3f}")

    # ================= 诊断图 =================
    out = Path(args.out) if args.out else root.parent / f"{root.name}_kinematics.png"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0][0]
    ax.scatter(act[:, 0], dstate[:, 0], s=4, alpha=.4, label="Δx")
    ax.scatter(act[:, 1], dstate[:, 1], s=4, alpha=.4, label="Δy")
    ax.scatter(act[:, 2], dstate[:, 2], s=4, alpha=.4, label="Δz")
    lim = max(1.0, np.abs(act).max(), np.abs(dstate).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
    ax.set_xlabel("action[t]  (mm/帧，记录值)")
    ax.set_ylabel("state[t+1] - state[t]  (mm)")
    ax.set_title(f"检验 A：lag={lag}  corr={np.round(cc,2)}  RMS={rms0:.2f} mm")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    ax = axes[0][1]
    for j, nm in enumerate("xyz"):
        ax.plot(act[:400, j], lw=.9, label=f"action Δ{nm}", alpha=.8)
    for j, nm in enumerate("xyz"):
        ax.plot(dstate[:400, j], lw=.9, ls="--", label=f"Δstate {nm}", alpha=.8)
    ax.set_xlabel("帧序（前 400 个帧对）"); ax.set_ylabel("mm")
    ax.set_title("检验 A：逐帧对照（实线=记录 action，虚线=state 差分）")
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=3)

    if shifts is not None:
        ax = axes[1][0]
        ax.scatter(actB[:, 1], shifts[:, 1], s=4, alpha=.4, label="Δy_arm ↔ dx_img")
        ax.scatter(actB[:, 0], shifts[:, 0], s=4, alpha=.4, label="Δx_arm ↔ dy_img")
        ax.set_xlabel("action (mm/帧)"); ax.set_ylabel("图像平移 (px/帧)")
        ax.set_title("检验 B：腕部图像平移 vs action")
        ax.grid(alpha=.3); ax.legend(fontsize=8)

        ax = axes[1][1]
        im = ax.imshow(np.abs(mat), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(["Δx_arm", "Δy_arm", "Δz_arm"])
        ax.set_yticks(range(2)); ax.set_yticklabels(names)
        for i in range(2):
            for j in range(3):
                ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center",
                        color="w" if abs(mat[i, j]) < 0.6 else "k", fontsize=10)
        ax.set_title("|相关矩阵|")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\n诊断图: {out}")

    print("\n" + "=" * 74)
    print(f"结论：检验A {'✅' if verdict_a else '❌'}   检验B "
          f"{'✅' if (shifts is not None and verdict_b) else '❌'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
