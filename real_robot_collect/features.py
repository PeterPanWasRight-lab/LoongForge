# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""特征声明与帧组装 —— 本模块唯一决定数据集 schema 的地方。

设计原则（与 HIL-SERL 解耦的关键）：
    本模块 **不 import 任何 loongforge.* 或 lerobot 的 RL 组件**。
    数据集里写什么字段，完全由 :func:`build_features` 显式声明。
    ``LeRobotDataset.add_frame`` 会做严格校验：frame 的键集合必须**恰好等于**
    ``features - DEFAULT_FEATURES``（见 ``datasets/utils.py:validate_frame``）。
    ⇒ 因此 **不可能混进 RL 遗留字段**（next.reward / next.done / next.truncated /
    complementary_info.*），这是相对 gym_manipulator record 路径的核心改进。

字段约定（与 LoongForge pi05 的输入契约对齐）：
    observation.images.<cam>   (3, H, W) uint8，帧里传 HWC（validate 两种都收，见
                               ``validate_feature_image_or_video``：expected (C,H,W)，
                               actual 可为 (c,h,w) 或 (h,w,c)）
    observation.state          (D,) float32
    action                     (A,) float32
    task                       str（``add_frame`` 强制要求，不是 feature 的一部分）

图像 dtype 语义：
    ``use_videos=True``  → dtype "video"，逐 episode 编码 mp4（需要 ffmpeg/torchcodec）
    ``use_videos=False`` → dtype "image"，逐帧 PNG 内嵌进 parquet（**自包含、无编码依赖**）
    本项目真机线用 ``use_videos=False``：边缘端没有视频编码依赖风险，
    且 LoongForge 侧的 lerobot 0.5.0 对 "image" 是一等公民
    （``lerobot_dataset.py:1211`` 把 ["image","video"] 并列处理）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 特征声明
# ---------------------------------------------------------------------------

#: 状态向量各维的名字（写进 info.json 的 names，便于人读；不影响训练）。
#: **顺序必须与 :func:`state_from_obs` 的拼接顺序严格一致**：
#:     位姿 6 → 力/力矩 6 → 夹爪 0/1
#: 这个「位姿在前」的顺序与历史数据集 ``hs_robot_assembly_v1`` 的
#: ``[位姿 6, F/T 6]`` 前缀兼容（它是 ``ForceTorqueProcessorStep`` 用
#: ``torch.cat([state, force_torque])`` 拼出来的）。
STATE_NAMES: List[str] = [
    "x", "y", "z", "rx", "ry", "rz",
    "Fx", "Fy", "Fz", "Tx", "Ty", "Tz",
    "gripper",
]
#: action 各维名字。注意第 4 维是夹爪的**二值/连续指令**，语义与 state 的夹爪维不同
ACTION_NAMES_4 = ["delta_x", "delta_y", "delta_z", "gripper"]

#: 力/力矩维数（``get_r(20..25)`` = Fx,Fy,Fz,Tx,Ty,Tz）
FORCE_DIM = 6
#: ``HsRobot.get_observation()`` 里力数据的键名（``hs_robot.py:370``，**是短名**，
#: 不是 ``observation.force_torque``）
FORCE_KEY = "force_torque"


def build_features(
    camera_keys: Sequence[str],
    image_size: int,
    state_dim: int,
    action_dim: int,
    use_videos: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """构造 LeRobot features 声明。

    Args:
        camera_keys: 相机键名列表，**顺序会被写入 info.json**。
            注意 LoongForge 读取时按 **key 字典序**决定视野槽位
            （``pi05_transform.py:133``），所以这里虽然按 config 顺序声明，
            但为了可预测，建议相机键名本身就按字典序命名（如 front < wrist）。
        image_size: 方形边长；图像会 resize 到 (image_size, image_size, 3)。
        state_dim: ``observation.state`` 维度 = 6(位姿) + 6(力/力矩) + 0/1(夹爪)。
            注意 **力那 6 维恒定占位**（不采集时填 0），见 ``RecordCfg`` 的说明。
        action_dim: ``action`` 维度。本线仅 Δxyz(+夹爪) ⇒ 3 或 4。
        use_videos: False ⇒ dtype "image"（PNG 内嵌 parquet）。
    """
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if state_dim <= 0 or action_dim <= 0:
        raise ValueError(f"state_dim/action_dim must be positive, got {state_dim}/{action_dim}")

    feats: Dict[str, Dict[str, Any]] = {}

    for cam in camera_keys:
        # info.json 里 image 的 shape 记作 (C, H, W)（与实测产物一致）
        feats[f"observation.images.{cam}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        }

    state_names = STATE_NAMES[:state_dim] if state_dim <= len(STATE_NAMES) else None
    feats["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": state_names,
    }

    action_names = ACTION_NAMES_4[:action_dim] if action_dim <= len(ACTION_NAMES_4) else None
    feats["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": action_names,
    }
    return feats


# ---------------------------------------------------------------------------
# 帧组装
# ---------------------------------------------------------------------------

#: HsRobot 笛卡尔模式的位姿键（``observation_features`` 里名字就是 loc_i.pos）
POSE_KEYS: List[str] = [f"loc_{i}.pos" for i in range(6)]


def state_from_obs(
    obs: Mapping[str, Any],
    include_gripper: bool,
    collect_force_torque: bool = False,
) -> np.ndarray:
    """从 ``HsRobot.get_observation()`` 的输出里取状态向量。

    拼接顺序（**与 :data:`STATE_NAMES` 严格对应**）::

        [x, y, z, rx, ry, rz]     位姿，mm + deg，绝对量
      + [Fx, Fy, Fz, Tx, Ty, Tz]  力/力矩（不采集时恒 0，但**照样占位**）
      + [gripper]                 可选，0/1 逻辑态

    ``loc_0..5.pos`` = [x, y, z (mm), rx, ry, rz (deg)]（**绝对量**，``get_observation``
    读的是 HRCController 的真实反馈，含延迟）。

    力/力矩的开关语义见 ``RecordCfg.collect_force_torque``：关掉只影响**内容**，
    不影响**维度** —— 这样 state 的 shape 恒定，训练侧与执行端都不用跟着改。

    ⚠️ **绝对量**：与 LIBERO 的 state（绝对 eef 位姿）同性质；action 才是增量。
    """
    missing = [k for k in POSE_KEYS if k not in obs]
    if missing:
        raise KeyError(
            f"get_observation() 缺少位姿键 {missing}；"
            f"实际键={sorted(obs.keys())}。请确认 robot.control_mode == 0（笛卡尔）。"
        )
    parts = [np.asarray([obs[k] for k in POSE_KEYS], dtype=np.float32)]

    if collect_force_torque:
        ft = obs.get(FORCE_KEY)
        if ft is None:
            raise KeyError(
                f"collect_force_torque=True，但 get_observation() 里没有 '{FORCE_KEY}'；"
                f"实际键={sorted(obs.keys())}。"
                "常见原因：Hardware.connect() 把 robot.force_sensor_enabled 设成了 False，"
                "或力传感器读取失败走了回落分支（hs_robot.py:372-374）。"
            )
        arr = np.asarray(ft, dtype=np.float32).reshape(-1)
        if arr.size != FORCE_DIM:
            raise ValueError(f"'{FORCE_KEY}' 期望 {FORCE_DIM} 维，实际 {arr.size} 维")
        parts.append(arr)
    else:
        parts.append(np.zeros(FORCE_DIM, dtype=np.float32))

    if include_gripper:
        if "gripper.pos" not in obs:
            raise KeyError("get_observation() 缺少 'gripper.pos'，但 include_gripper=True")
        parts.append(np.asarray([obs["gripper.pos"]], dtype=np.float32))
    return np.concatenate(parts, axis=0)


def action_to_vector(teleop_action: Mapping[str, Any], include_gripper: bool) -> np.ndarray:
    """把 SpaceMouse 的动作字典压成 action 向量。

    SpaceMouse 返回 ``{delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw,
    gripper?}``（``teleop_spacemouse.py:91``）。
    **本线只取 Δxyz(+夹爪)** —— 姿态维在配置里被 ``rotation_scale: 0`` 置零，
    取进来也是常数，所以直接不声明、不写盘。

    ⚠️ 夹爪维的语义：SpaceMouse 给的是控制指令（``gripper_state``），
    不是 state 那一路的关节量，两者不存在导数关系。
    """
    for k in ("delta_x", "delta_y", "delta_z"):
        if k not in teleop_action:
            raise KeyError(f"teleop action 缺少 {k}；实际键={sorted(teleop_action.keys())}")
    parts = [
        np.asarray(
            [teleop_action["delta_x"], teleop_action["delta_y"], teleop_action["delta_z"]],
            dtype=np.float32,
        )
    ]
    if include_gripper:
        # SpaceMouse 那一侧键名是 "gripper"；缺省时用 0.0（= 不动）
        parts.append(np.asarray([teleop_action.get("gripper", 0.0)], dtype=np.float32))
    return np.concatenate(parts, axis=0)


def action_to_robot(teleop_action: Mapping[str, Any]) -> Dict[str, float]:
    """把遥操动作翻译成 ``HsRobot.send_action`` 期望的字典。

    ``HsRobot.send_action`` 的契约（``hs_robot.py:428``）::

        {'delta_x','delta_y','delta_z','delta_roll','delta_pitch','delta_yaw','gripper.pos'}

    ⚠️ 三处键名差异，**这是最容易写错的地方**：
        1. 遥操侧是 ``gripper``，robot 侧要 ``gripper.pos``
        2. robot 侧要求 ``delta_roll/pitch/yaw`` 存在（即使为 0）
        3. 夹爪的值：``HsRobot.send_action`` 对 **负数直接跳过（保持硬件状态）**，
           只有 >= 0 才按 ``> 0.5`` 二值化触发 IO 脉冲（``hs_robot.py:452-456``）
           ⇒ 因此这里对夹爪做 **clamp 到 [0, 1]**，不要让它出现负数。
    """
    out: Dict[str, float] = {
        "delta_x": float(teleop_action.get("delta_x", 0.0)),
        "delta_y": float(teleop_action.get("delta_y", 0.0)),
        "delta_z": float(teleop_action.get("delta_z", 0.0)),
        "delta_roll": float(teleop_action.get("delta_roll", 0.0)),
        "delta_pitch": float(teleop_action.get("delta_pitch", 0.0)),
        "delta_yaw": float(teleop_action.get("delta_yaw", 0.0)),
    }
    if "gripper" in teleop_action or "gripper.pos" in teleop_action:
        raw = teleop_action.get("gripper", teleop_action.get("gripper.pos", 0.0))
        try:
            g = float(raw)
        except (TypeError, ValueError):
            g = 0.0
        out["gripper.pos"] = float(min(max(g, 0.0), 1.0))
    return out


def build_frame(
    images: Mapping[str, np.ndarray],
    state: np.ndarray,
    action: np.ndarray,
    task: str,
    features: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """组装一帧。键集合**必须**恰好等于 ``features - DEFAULT_FEATURES`` + ``task``。

    多余的键会被 ``validate_frame`` 直接拒绝（这是我们想要的：宁可报错也不要脏字段）。
    """
    frame: Dict[str, Any] = {}

    for key, spec in features.items():
        if spec["dtype"] in ("image", "video"):
            cam = key[len("observation.images.") :]
            if cam not in images:
                raise KeyError(f"帧缺少相机 '{cam}'（features 声明了 {key}）")
            img = np.asarray(images[cam])
            if img.dtype != np.uint8:
                raise ValueError(
                    f"相机 '{cam}' 期望 uint8，实际 {img.dtype}；"
                    "请在采集侧转换后再 add_frame（LeRobot 的 image writer 会校验 0~255）"
                )
            if img.ndim != 3:
                raise ValueError(f"相机 '{cam}' 期望 HWC 3 维，实际 shape={img.shape}")
            frame[key] = img
        elif key == "observation.state":
            frame[key] = np.asarray(state, dtype=np.float32).reshape(-1)
        elif key == "action":
            frame[key] = np.asarray(action, dtype=np.float32).reshape(-1)
        else:
            raise NotImplementedError(f"features 里出现未支持的键 {key!r}（dtype={spec['dtype']}）")

    # 形状自检：把校验前置到这里，出错时报的错比 add_frame 里清晰
    for key in ("observation.state", "action"):
        want = tuple(features[key]["shape"])
        got = frame[key].shape
        if got != want:
            raise ValueError(f"{key} 形状不符：声明 {want}，实际 {got}")

    if not isinstance(task, str) or not task.strip():
        raise ValueError("task 必须是非空字符串（它会进 pi05 的 prompt）")
    frame["task"] = task
    return frame
