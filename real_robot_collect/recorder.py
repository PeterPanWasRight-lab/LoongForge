# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""采集主循环 —— 与 HIL-SERL 解耦的 record 语义。

与 ``lerobot.rl.gym_manipulator --mode record`` 的**四处本质差异**：

1. **只落成功条**（默认）。超时 / 按 q / 急停的 episode 直接 ``clear_episode_buffer()``，
   不进数据集。⇒ 数据集天然干净，不需要事后按 ``next.reward`` 清洗。
   （gym_manipulator 是**每条都 save_episode**，成功与否只能事后从 reward 反推。）
2. **没有 RL 字段**。features 由 ``features.build_features`` 显式声明，
   ``add_frame`` 的严格校验保证 frame 里不可能多出 ``next.reward``/``next.done``/
   ``next.truncated``/``complementary_info.*``。
3. **没有环境/奖励/复位流水线**。复位就是 ``HsRobot.reset_to_position`` 一次调用，
   没有 RobotEnv、没有 AssemblyMonitor、没有 TimeLimitProcessorStep。
4. **不依赖 ``next.done`` 判成功** —— 成功由键盘 ``s`` 事件或超时直接决定，
   在当前循环里就地处理，不依赖事后解析。

键位（源码：``teleop_keyboard.py:394-408``）：
    s = 标记成功并结束       r = 结束并重录（本条不落盘）      q = 结束并放弃
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import CollectConfig
from .features import action_to_robot, action_to_vector, build_frame, state_from_obs
from .hw import Hardware

logger = logging.getLogger(__name__)

#: 本条 episode 的结局
OUTCOME_SUCCESS = "success"
OUTCOME_RERECORD = "rerecord"
OUTCOME_ABORT = "abort"      # 按 q（标记失败）
OUTCOME_TIMEOUT = "timeout"  # 跑满 control_time_s
OUTCOME_KILL = "kill"        # Ctrl+C 全局停止


@dataclass
class EpisodeLog:
    """单条 episode 的现场记录，采集完一并落盘，便于回溯数据质量。"""

    index: int                  # 第几次尝试（从 1 开始）
    outcome: str
    frames: int = 0
    seconds: float = 0.0
    saved: bool = False

    @property
    def eff_fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


@dataclass
class RecordSummary:
    episodes: List[EpisodeLog] = field(default_factory=list)
    saved: int = 0
    attempted: int = 0
    dataset_root: Optional[str] = None

    def table(self) -> str:
        lines = [
            f"{'#':>3}  {'outcome':<9} {'frames':>6} {'sec':>6} {'fps':>6}  {'saved':>5}",
            "-" * 46,
        ]
        for e in self.episodes:
            lines.append(
                f"{e.index:>3}  {e.outcome:<9} {e.frames:>6} {e.seconds:>6.1f} "
                f"{e.eff_fps:>6.1f}  {'YES' if e.saved else '-':>5}"
            )
        lines.append("-" * 46)
        lines.append(f"尝试 {self.attempted} 次 · 落盘 {self.saved} 条")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _read_events(keyboard: Any) -> Dict[str, Any]:
    """读键盘事件（s / r / q）。任何异常都吞掉并返回空 —— 事件通道不应中断采集。

    🚨 **必须先 tick 一次 ``get_action()``**（2026-10-13 实测踩坑）：

    按 ``s`` 时，``_on_press`` 只把它放进 ``event_queue``
    （``teleop_keyboard.py:204``）；真正把它挪进 ``misc_keys_queue`` 的代码
    **写在 ``get_action()`` 内部**（``teleop_keyboard.py:341-347``），
    而 ``get_teleop_events()`` 只从 ``misc_keys_queue`` 读（``:390``）。

    本模块的动作**只取自 SpaceMouse**，从不调 keyboard 的 ``get_action()``
    ⇒ ``misc_keys_queue`` 恒为空 ⇒ ``s``/``r``/``q`` **永远读不到**。

    症状：pynput 日志能看到 ``[KEY_DOWN] Char key: s``，但 episode 不结束、
    ``save_episode()`` 从不触发、数据集 0 条（且**完全不报错**）。

    顺带：``_processed_char_keys`` 的清理也在 ``get_action()`` 里
    （``:300-315``，注释原文「如果不加这段，s/r/q 键按过一次后就会被永久标记为
    已处理，第二次按无效」）—— 所以这里必须**每帧** tick 一次，而不是进 episode 时调一次。

    返回的 dict 已把 ``TeleopEvents`` 枚举键转成它的 ``value`` 字符串
    （如 ``"success"``），便于 ``_truthy()`` 直接查。
    """
    if keyboard is None:
        return {}

    # ★ 关键：tick 一次键盘的 get_action()，让 s/r/q 进入 misc_keys_queue。
    #   返回值（键盘的 delta/gripper）**故意丢弃** —— 动作只来自 SpaceMouse。
    try:
        keyboard.get_action()
    except Exception as exc:
        _WARNED_GET_ACTION.add(str(exc))
        if len(_WARNED_GET_ACTION) == 1:
            logger.warning("keyboard.get_action() tick 失败（s/r/q 会读不到）：%s", exc)

    try:
        ev = keyboard.get_teleop_events() or {}
        return {str(getattr(k, "value", k)): bool(v) for k, v in ev.items()}
    except Exception as exc:
        logger.debug("读遥操事件失败（忽略）: %s", exc)
        return {}


#: 只警告一次，避免每帧刷屏
_WARNED_GET_ACTION: set = set()


def _reset_keyboard_state(keyboard: Any) -> None:
    """清空键盘的事件/夹爪队列 —— **每条 episode 开始前必须调**。

    否则：用户在「复位 / 摆件」期间按下的 ``s``/``r``/``q`` 会留在队列里，
    下一条的第一帧就读到它 ⇒ 本条 **0 帧就结束**，甚至把一条空数据落盘。

    ``reset_state`` 清 ``gripper_queue`` + ``misc_keys_queue`` + ``_processed_char_keys``
    （``teleop_keyboard.py:436-450``）—— 其中 ``misc_keys_queue`` 正是 s/r/q 的队列。
    """
    if keyboard is None:
        return
    fn = getattr(keyboard, "reset_state", None)
    if callable(fn):
        try:
            fn()
        except Exception as exc:  # pragma: no cover
            logger.debug("reset_state 失败（忽略）: %s", exc)


def _truthy(events: Dict[str, Any], *names: str) -> bool:
    """``TeleopEvents`` 枚举的 value 是字符串（如 'success'）；兼容传枚举本身。"""
    for n in names:
        if events.get(n):
            return True
    return False


def _pace(loop_start: float, dt: float) -> None:
    """按 fps 限速。用 sleep 而不是忙等 —— 这里不是实时控制环，
    真机的控制由 HRC 控制器在底层跑，我们只负责以稳定频率喂指令与采数据。"""
    remaining = dt - (time.perf_counter() - loop_start)
    if remaining > 0:
        time.sleep(remaining)


# ---------------------------------------------------------------------------
# 单条 episode
# ---------------------------------------------------------------------------


def record_one(
    cfg: CollectConfig,
    hw: Hardware,
    dataset: Any,
    features: Dict[str, Dict[str, Any]],
    episode_ordinal: int,
    verbose_every: int = 30,
) -> tuple[str, int, float]:
    """录一条 episode，返回 ``(outcome, frames, seconds)``。

    **不负责落盘** —— 落盘/丢弃由 :func:`run_recording` 按 outcome 决定。
    这样"什么算成功"和"写不写盘"是两件事，可以独立改。
    """
    rec = cfg.record
    dt = 1.0 / float(rec.fps)
    task = cfg.dataset.task

    # 清掉复位/摆件期间残留的按键（否则一个残留的 s 会让本条 0 帧就结束）
    _reset_keyboard_state(hw.keyboard)

    frames = 0
    t0 = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # 一帧只读一次硬件（图像与 state 必须来自同一时刻）
        obs = hw.read_raw_obs()
        images = hw.preprocess_images(obs)
        state = state_from_obs(
            obs,
            include_gripper=rec.include_gripper,
            collect_force_torque=rec.collect_force_torque,
        )

        teleop_action = hw.spacemouse.get_action() if hw.spacemouse is not None else {}
        hw.robot.send_action(action_to_robot(teleop_action))
        action_vec = action_to_vector(teleop_action, include_gripper=rec.include_gripper)

        frame = build_frame(images, state, action_vec, task, features)
        dataset.add_frame(frame)
        frames += 1

        if verbose_every > 0 and frames % verbose_every == 0:
            # 力开着时顺便打出来 —— 若恒为 0 说明传感器没通，现场能立刻发现
            ft_txt = (
                "  F/T=" + str(np.round(state[6:12], 2).tolist())
                if rec.collect_force_torque
                else ""
            )
            logger.info(
                "  ep#%d  %d 帧  %.1fs  当前位姿 xyz=%s  夹爪=%s%s",
                episode_ordinal,
                frames,
                time.perf_counter() - t0,
                np.round(state[:3], 1).tolist(),
                round(float(state[-1]), 2) if rec.include_gripper else "n/a",
                ft_txt,
            )

        # ---- 事件（s / r / q）----
        events = _read_events(hw.keyboard)
        if _truthy(events, "rerecord_episode"):
            logger.info("  🔄 r 按下 —— 本条作废，重录")
            return OUTCOME_RERECORD, frames, time.perf_counter() - t0
        if _truthy(events, "success"):
            logger.info("  ✅ s 按下 —— 标记成功")
            return OUTCOME_SUCCESS, frames, time.perf_counter() - t0
        if _truthy(events, "key_punish"):
            logger.info("  ❌ q 按下 —— 标记失败并结束")
            return OUTCOME_ABORT, frames, time.perf_counter() - t0
        if _truthy(events, "terminate_episode"):
            # 理论上 s/r/q 三键都会同时置 terminate_episode；到这里说明是别处置的
            logger.info("  ⏹ 收到 terminate_episode（未伴随 s/r/q）—— 按放弃处理")
            return OUTCOME_ABORT, frames, time.perf_counter() - t0

        # ---- 超时 ----
        if time.perf_counter() - t0 >= rec.control_time_s:
            logger.info("  ⏱ 达到 control_time_s=%.1fs 上限", rec.control_time_s)
            return OUTCOME_TIMEOUT, frames, time.perf_counter() - t0

        _pace(loop_start, dt)


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------


def run_recording(
    cfg: CollectConfig,
    hw: Hardware,
    dataset: Any,
    features: Dict[str, Dict[str, Any]],
) -> RecordSummary:
    """跑满 ``dataset.num_episodes`` 条**成功**数据。

    流程::

        复位 -> 人摆件 -> 录一条 -> s 成功则落盘 / r 则重录 / q 或超时则丢弃 -> 下一轮

    Ctrl+C 安全退出：已完成并落盘的 episode 全部保留，当前条丢弃。
    """
    rec = cfg.record
    target = int(cfg.dataset.num_episodes)
    summary = RecordSummary()
    kill = False

    logger.info("=" * 72)
    logger.info("开始采集：目标 %d 条成功数据 · 数据集 %s", target, cfg.dataset.repo_id)
    logger.info("  fps=%s · 图像 %dx%d · state_dim=%d · action_dim=%d · use_videos=%s",
                rec.fps, rec.image_size, rec.image_size, cfg.state_dim, cfg.action_dim, rec.use_videos)
    logger.info("  单条上限 %.1fs · 复位等待 %.1fs · 失败条保留=%s",
                rec.control_time_s, rec.reset_time_s, rec.keep_failed)
    logger.info("  键位：s=成功  r=重录  q=放弃  Ctrl+C=停止")
    logger.info("=" * 72)

    while summary.saved < target and not kill:
        ordinal = summary.attempted + 1
        summary.attempted += 1

        logger.info("--- 第 %d 条（已落盘 %d/%d）: 复位 ---", ordinal, summary.saved, target)
        try:
            hw.reset_to_start()
        except Exception as exc:
            logger.error("复位失败，跳过本条: %s", exc)
            summary.episodes.append(EpisodeLog(ordinal, "reset_failed"))
            continue
        time.sleep(max(0.0, rec.reset_time_s))

        # 保险：确保 buffer 干净（rerecord 路径已经清过，这里再兜一次）
        _safe_clear(dataset)

        try:
            outcome, frames, seconds = record_one(cfg, hw, dataset, features, ordinal)
        except KeyboardInterrupt:
            logger.warning("收到 Ctrl+C —— 丢弃当前条并退出（已落盘的 %d 条保留）", summary.saved)
            _safe_clear(dataset)
            summary.episodes.append(EpisodeLog(ordinal, OUTCOME_KILL, 0, 0.0))
            kill = True
            break

        log = EpisodeLog(ordinal, outcome, frames, seconds)

        if outcome == OUTCOME_SUCCESS:
            dataset.save_episode()
            log.saved = True
            summary.saved += 1
            logger.info("  💾 已落盘（第 %d 条），累计 %d/%d", summary.saved, summary.saved, target)
        elif outcome == OUTCOME_RERECORD:
            _safe_clear(dataset)
            logger.info("  ↩️ 本条未落盘（重录）")
        else:
            if rec.keep_failed and frames > 0:
                dataset.save_episode()
                log.saved = True
                summary.saved += 1
                logger.info("  💾 失败条按配置保留（keep_failed=true）")
            else:
                _safe_clear(dataset)
                logger.info("  🗑 本条未落盘（outcome=%s）", outcome)

        summary.episodes.append(log)

    # 正常采满目标条数 ⇒ 把机械臂送回起始位再收尾（用户 2026-10-13 要求：
    # "采集完最后一条数据后，机械臂应该复位到起始位置后再结束程序"）。
    # ⚠️ Ctrl+C 中止的路径**不复位** —— 尊重"立刻停下"的意图：
    #    用户 Ctrl+C 往往是因为出了状况，此时再让机械臂自己动一次是危险的。
    if not kill:
        logger.info("🏁 已采满 %d/%d 条 —— 把机械臂送回起始位", summary.saved, target)
        try:
            hw.reset_to_start()
            logger.info("   已回到起始位，可以收工")
        except Exception as exc:  # pragma: no cover
            logger.warning("收尾复位失败（不影响已落盘的数据）：%s", exc)

    summary.dataset_root = str(getattr(dataset, "root", cfg.dataset_root_abs()))
    return summary


def run_read_only(
    cfg: CollectConfig,
    hw: Hardware,
    dataset: Any,
    features: Dict[str, Dict[str, Any]],
    num_episodes: int = 3,
    seconds_per_episode: float = 2.0,
) -> RecordSummary:
    """**只读采集** —— 全程**不调用** ``send_action`` / ``reset_to_position``。

    用途：机械臂不允许运动（被别人占用 / 未复位 / 无安全员在场）时，仍能验证除
    「控制下发」以外的整条链路：

        ① 能否稳定读观测（相机×2 + 位姿 + [力]）
        ② 帧组装与 schema（features ↔ ``add_frame`` 的严格校验）
        ③ 落盘链路（``save_episode`` → parquet + meta + stats）
        ④ 收尾（``finalize``）与产物可读性（``verify``）

    ⚠️ 产出的 **action 全为 0**、state 是真实读数但机械臂静止 ⇒
       这份数据**只能用于链路验证，绝不可用于训练**。
    ⚠️ 不碰 SpaceMouse / 键盘（调用方应传 ``connect_teleop=False``）。
    """
    rec = cfg.record
    fps = int(rec.fps)
    n_frames = max(1, int(round(fps * seconds_per_episode)))
    dt = 1.0 / float(fps)
    zero_action = np.zeros((cfg.action_dim,), dtype=np.float32)
    summary = RecordSummary()

    logger.info("=" * 72)
    logger.info("🔒 只读模式（READ-ONLY）：全程不发控制指令、不复位")
    logger.info("   计划 %d 条 × %d 帧（%.1f s @ %d fps）· action 全 0",
                num_episodes, n_frames, seconds_per_episode, fps)
    logger.info("   state_dim=%d（位姿6 + 力6%s）· images=%s",
                cfg.state_dim, "" if rec.include_gripper else "，不含夹爪", cfg.camera_keys)
    logger.info("=" * 72)

    for ep in range(1, num_episodes + 1):
        _safe_clear(dataset)
        t0 = time.perf_counter()
        frames = 0
        for _ in range(n_frames):
            loop_start = time.perf_counter()

            # ★★ 只读：只有 read_*，没有任何 send_action / reset_to_position ★★
            obs = hw.read_raw_obs()
            images = hw.preprocess_images(obs)
            state = state_from_obs(
                obs,
                include_gripper=rec.include_gripper,
                collect_force_torque=rec.collect_force_torque,
            )

            frame = build_frame(images, state, zero_action, cfg.dataset.task, features)
            dataset.add_frame(frame)
            frames += 1

            if frames % max(1, fps) == 0:
                logger.info("  ep#%d  %d/%d 帧  位姿 xyz=%s%s",
                            ep, frames, n_frames, np.round(state[:3], 1).tolist(),
                            ("  F/T=" + str(np.round(state[6:12], 2).tolist()))
                            if rec.collect_force_torque else "")
            _pace(loop_start, dt)

        dataset.save_episode()
        log = EpisodeLog(ep, "read_only", frames, time.perf_counter() - t0, saved=True)
        summary.episodes.append(log)
        summary.saved += 1
        summary.attempted += 1
        logger.info("  💾 第 %d 条已落盘（%d 帧 / %.1f s / 实测 %.1f fps）",
                    ep, frames, log.seconds, log.eff_fps)

    summary.dataset_root = str(getattr(dataset, "root", cfg.dataset_root_abs()))
    return summary


def _safe_clear(dataset: Any) -> None:
    try:
        dataset.clear_episode_buffer()
    except Exception as exc:  # pragma: no cover
        logger.debug("clear_episode_buffer 失败（忽略）: %s", exc)


def finalize_dataset(dataset: Any) -> None:
    """收尾：停图像写入线程 → **finalize 数据集**。

    ⚠️ 第二步**必须做**。``LeRobotDataset.finalize()`` 的 docstring 原话：

        This function needs to be called after data collection/conversion,
        else **footer metadata won't be written to the parquet files**.
        The dataset won't be valid and can't be loaded.

    实现是 ``self._close_writer()`` + ``self.meta._close_writer()``
    （``lerobot_dataset.py:1054``）；而 ``_close_writer`` 里除了关 parquet writer，
    还会 ``_flush_metadata_buffer()``（``:158``）。**parquet 的 writer 是长开的**，
    不 close 就只写出数据块、不写 footer ⇒ ``data/chunk-*/file-*.parquet`` 读不出来
    （``__del__`` 里有兜底 close，但那依赖垃圾回收时机，异常路径下不可靠）。

    顺序：先停图像线程（``save_episode`` 内部已 ``_wait_image_writer()``，
    这里只是干净收尾），再 finalize。两者互不影响，但先停线程更安全。
    """
    stop = getattr(dataset, "stop_image_writer", None)
    if callable(stop):
        try:
            stop()
            logger.info("图像写入线程已停止")
        except Exception as exc:  # pragma: no cover
            logger.warning("stop_image_writer 失败: %s", exc)

    fin = getattr(dataset, "finalize", None)
    if callable(fin):
        try:
            fin()
            logger.info("数据集已 finalize（parquet footer + meta 已落盘）")
        except Exception as exc:  # pragma: no cover
            logger.error("finalize 失败 —— 数据集可能不可读: %s", exc)
