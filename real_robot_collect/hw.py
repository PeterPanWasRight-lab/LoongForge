# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""硬件组装 + 图像预处理 —— 薄封装，只依赖 lerobot（fork），不依赖 loongforge。

依赖的 lerobot 组件（全部是**驱动层**，与 HIL-SERL / RL 无关）：
    HsRobot / HsRobotConfig        robots/hs_robot
    RealSenseCameraConfig          cameras/realsense
    SpaceMouseTeleop(+Config)      teleoperators/spacemouse
    KeyboardEndEffectorTeleop      teleoperators/keyboard
    TeleopEvents                   teleoperators/utils

**不依赖**：gym_manipulator / RobotEnv / reward_classifier / AssemblyMonitor /
处理器流水线（processor.*）。这就是“与 HIL-SERL 解耦”的准确含义 —— 复用驱动，
不复用 RL 环境。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .config import CameraCfg, CollectConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 图像预处理（采集与执行必须共用这一个函数）
# ---------------------------------------------------------------------------


def preprocess_image(
    img_hwc_uint8: np.ndarray,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    size: int = 224,
) -> np.ndarray:
    """uint8 HWC -> (crop) -> resize -> uint8 HWC。

    ⚠️ **执行端的 runner 必须 import 这个函数**，不要自己再写一遍。
    采集与执行若用不同的 crop/插值/round，会引入静默的分布偏移 ——
    这正是 ``prefer_controller_ee`` 那类问题的同一个教训。

    处理顺序刻意与 fork 的处理器链一致（``VanillaObservationProcessorStep``
    先 HWC uint8 -> CHW float32 [0,1]，再由 ``ImageCropResizeProcessorStep``
    做 ``F.crop`` + ``F.resize``）：
        这里先把 uint8 归一化到 [0,1] 的 CHW float，crop/resize 后再还原 uint8。

    crop_box: ``(top, left, height, width)``（torchvision 的顺序，不是 x0y0x1y1）。
    """
    if img_hwc_uint8.dtype != np.uint8:
        raise ValueError(f"preprocess_image 期望 uint8，实际 {img_hwc_uint8.dtype}")
    if img_hwc_uint8.ndim != 3 or img_hwc_uint8.shape[2] not in (1, 3):
        raise ValueError(f"preprocess_image 期望 HWC（通道 1 或 3），实际 {img_hwc_uint8.shape}")

    need = crop_box is not None or (img_hwc_uint8.shape[0] != size or img_hwc_uint8.shape[1] != size)
    if not need:
        return np.ascontiguousarray(img_hwc_uint8)

    import torch
    from torchvision.transforms import functional as F

    t = torch.from_numpy(np.ascontiguousarray(img_hwc_uint8)).permute(2, 0, 1).float().div_(255.0)
    if crop_box is not None:
        top, left, h, w = (int(v) for v in crop_box)
        H, W = t.shape[-2], t.shape[-1]
        if top < 0 or left < 0 or h <= 0 or w <= 0 or top + h > H or left + w > W:
            raise ValueError(
                f"crop_box={crop_box} 越界（图像 {H}x{W}）。"
                "注意顺序是 (top, left, height, width)。"
            )
        t = F.crop(t, top, left, h, w)
    t = F.resize(t, [size, size])
    out = (t.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return np.ascontiguousarray(out.permute(1, 2, 0).numpy())


# ---------------------------------------------------------------------------
# 配置对象构造（显式，不经过 draccus）
# ---------------------------------------------------------------------------


def build_robot_config(cfg: CollectConfig) -> Any:
    from lerobot.robots.hs_robot import HsRobotConfig

    cams: Dict[str, Any] = {}
    for c in cfg.robot.cameras:
        cams[c.name] = build_camera_config(c)

    kwargs: Dict[str, Any] = {
        "ip": cfg.robot.ip,
        "port": cfg.robot.port,
        "control_mode": 0,
        "gripper_io_port": cfg.robot.gripper_io_port,
        "cameras": cams,
    }
    if cfg.robot.workspace_limits:
        kwargs["workspace_limits"] = dict(cfg.robot.workspace_limits)
    return HsRobotConfig(**kwargs)


def build_camera_config(c: CameraCfg) -> Any:
    rot = _resolve_rotation(c.rotation)
    if c.type == "intelrealsense":
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        if not c.serial_number_or_name:
            raise ValueError(f"相机 '{c.name}' 是 intelrealsense，必须给 serial_number_or_name")
        kw: Dict[str, Any] = {
            "serial_number_or_name": str(c.serial_number_or_name),
            "fps": c.fps,
            "width": c.width,
            "height": c.height,
        }
        if rot is not None:
            kw["rotation"] = rot
        return RealSenseCameraConfig(**kw)

    if c.type == "opencv":
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        if c.index_or_path is None:
            raise ValueError(f"相机 '{c.name}' 是 opencv，必须给 index_or_path")
        kw = {"index_or_path": c.index_or_path, "fps": c.fps, "width": c.width, "height": c.height}
        if rot is not None:
            kw["rotation"] = rot
        return OpenCVCameraConfig(**kw)

    raise NotImplementedError(f"不支持的相机 type={c.type!r}（支持 intelrealsense / opencv）")


def _resolve_rotation(name: Optional[str]) -> Any:
    """把 ``"ROTATE_90"`` 这样的字符串转成 fork 的 Cv2Rotation 枚举；None 原样返回。"""
    if not name:
        return None
    try:
        from lerobot.cameras.configs import Cv2Rotation
    except Exception:  # pragma: no cover - 兼容不同布局
        from lerobot.cameras.utils import Cv2Rotation  # type: ignore

    if not hasattr(Cv2Rotation, name):
        raise ValueError(f"未知 rotation={name!r}；可选 {[m for m in dir(Cv2Rotation) if not m.startswith('_')]}")
    return getattr(Cv2Rotation, name)


def build_spacemouse(cfg: CollectConfig) -> Optional[Any]:
    if not cfg.spacemouse.enabled:
        return None
    from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig
    from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop

    sm = cfg.spacemouse
    conf = SpaceMouseTeleopConfig(
        deadzone=sm.deadzone,
        rotation_deadzone=sm.rotation_deadzone,
        filter_alpha=sm.filter_alpha,
        position_scale=sm.position_scale,
        rotation_scale=sm.rotation_scale,
        max_position_offset=sm.max_position_offset,
        max_rotation_offset=sm.max_rotation_offset,
        update_rate_hz=sm.update_rate_hz,
        max_angle_delta_deg=sm.max_angle_delta_deg,
        use_gripper=sm.use_gripper,
        button0_mode=sm.button0_mode,
        button1_mode=sm.button1_mode,
        emergency_buttons=tuple(sm.emergency_buttons),
        emergency_hold_time=sm.emergency_hold_time,
        output_mode=sm.output_mode,
        axis_mapping=tuple(sm.axis_mapping),
        axis_inversion=tuple(sm.axis_inversion),
    )
    return SpaceMouseTeleop(conf)


def build_keyboard(cfg: CollectConfig) -> Optional[Any]:
    """构造键盘事件设备。

    ``KeyboardEndEffectorTeleop``（``name="keyboard_ee"``）负责 s/r/q 事件；
    它在 ``__init__`` 里就建好了 ``misc_keys_queue``，事件由 pynput 监听线程投递。
    """
    if not cfg.keyboard.enabled:
        return None
    from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardEndEffectorTeleopConfig
    from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardEndEffectorTeleop

    conf = KeyboardEndEffectorTeleopConfig(
        step_size=cfg.keyboard.step_size,
        use_gripper=cfg.keyboard.use_gripper,
    )
    return KeyboardEndEffectorTeleop(conf)


# ---------------------------------------------------------------------------
# 组装 + 连接
# ---------------------------------------------------------------------------


class Hardware:
    """RAII 风格的硬件句柄。``connect()`` / ``close()`` 成对使用。"""

    def __init__(self, cfg: CollectConfig) -> None:
        self.cfg = cfg
        self.robot = None
        self.spacemouse = None
        self.keyboard = None
        self._connected = False

    def connect(self, connect_teleop: bool = True) -> None:
        from lerobot.robots.utils import make_robot_from_config

        logger.info("连接机器人 %s:%s ...", self.cfg.robot.ip, self.cfg.robot.port)
        self.robot = make_robot_from_config(build_robot_config(self.cfg))
        self.robot.connect()
        logger.info("机器人 + %d 路相机已连接", len(self.cfg.robot.cameras))

        self._configure_force_sensor()
        self._enable_robot()

        if connect_teleop:
            self.spacemouse = build_spacemouse(self.cfg)
            if self.spacemouse is not None:
                self.spacemouse.connect()
                logger.info("SpaceMouse 已连接（rotation_scale=%s）", self.cfg.spacemouse.rotation_scale)
            self.keyboard = build_keyboard(self.cfg)
            if self.keyboard is not None:
                self.keyboard.connect()
                logger.info("键盘事件设备已连接（s=成功 / r=重录 / q=放弃）")
        self._connected = True

    def _enable_robot(self) -> None:
        """使能机械臂 —— **不做这一步，所有运动指令都会静默失效**。

        🚨 依据（2026-10-13 实测）：``hs_robot.py:241`` 把
        ``self.hsrobot.enable()`` **注释掉了**，只留一句
        ``logger.info("Robot enabled successfully")``（**假日志**）。
        于是 connect 之后机械臂处于「未使能」状态：

          * 读位姿 ``get_cartesian_position()`` 正常 —— 完全看不出问题
          * 但 ``move_to_cartesian_position()`` / ``send_action()`` **全部无效且不报错**
          * 实测症状：复位（含"先上抬 20mm"）零位移、推 SpaceMouse 零位移，
            而 ``_cached_target_position`` 却正常累积 ⇒ 指令发出去了，只是没人执行。

        使能本身**不产生运动**（伺服就位、保持当前位姿），故默认开启；
        由 ``robot.enable_on_connect`` 控制。
        """
        import time

        assert self.robot is not None

        if not self.cfg.robot.enable_on_connect:
            logger.warning(
                "robot.enable_on_connect=false —— 跳过使能。"
                "若机械臂没有由外部（示教器/其它程序）使能，运动指令将全部无效"
            )
            return

        hsrobot = getattr(self.robot, "hsrobot", None)
        if hsrobot is None or not hasattr(hsrobot, "enable"):
            logger.warning("HsRobot 上没有 hsrobot.enable()（fork 版本可能不同）；已跳过使能")
            return

        try:
            hsrobot.enable()
            time.sleep(0.5)
            logger.info("✅ 已调用 hsrobot.enable() —— 机械臂已使能（否则运动指令静默失效）")
        except Exception as exc:  # pragma: no cover
            logger.warning("hsrobot.enable() 调用失败：%s —— 运动指令可能无效", exc)
            return

        # ★ 使能之后必须重设运动起始点，否则运动通道仍然是死的
        self._restart_cartesian_start()

    def _restart_cartesian_start(self) -> None:
        """使能之后**重新设置笛卡尔运动起始点**。

        🚨 为什么非做不可（2026-10-13 实测）：``HsRobot.connect()`` 里的顺序是

            HRC.connect(ip, port, "cartesian")
                └─ 内部调 start_cartesian_position(当前位姿)   # 设「运动起始点」

        而这一步发生在**机械臂使能之前** ⇒ 底层 ``HRC_start_d_pos`` 失败
        （C 库会往 stdout 打印 ``[error] 设置运动起始点失败``），
        之后所有 ``move_to_cartesian_position`` **全部被静默忽略**。

        ``move_to_cartesian_position`` 的 docstring 写得很清楚：
        「**必须先调用 start_cartesian_position 设置运动模式后才能使用此函数**」。

        所以正确顺序是「先 enable，再设起始点」—— 本方法补的就是第二步。
        ``start_cartesian_position`` 对已是 cartesian 模式的实例是幂等的
        （``hsrosi/controller.py:369`` 只在模式**不同**时才抛错），可安全重复调用。
        """
        hrc = getattr(self.robot, "HRC", None)
        if hrc is None or not hasattr(hrc, "start_cartesian_position"):
            logger.warning("HRC 上没有 start_cartesian_position（fork 版本可能不同）；跳过起始点重设")
            return
        try:
            pos = hrc.get_cartesian_position()
            pos = [float(v) for v in pos]
            hrc.start_cartesian_position(pos)
            logger.info("✅ 已重设笛卡尔运动起始点 = %s（运动通道激活）",
                        [round(v, 3) for v in pos])
        except Exception as exc:  # pragma: no cover
            logger.warning("重设笛卡尔起始点失败：%s —— 运动指令可能仍然无效", exc)

    def _configure_force_sensor(self) -> None:
        """按 ``record.collect_force_torque`` 开关力/力矩读取。

        ⚠️ 为什么改**实例属性**而不是构造参数：``HsRobotConfig`` **没有**
        ``add_force_torque_to_observation`` 字段 —— 源码是
        ``self.force_sensor_enabled = getattr(config, 'add_force_torque_to_observation', True)``
        （``hs_robot.py:60``），缺字段时 getattr 恒返回 True，**传配置项也关不掉**。
        所以唯一可靠的办法是构造之后覆盖实例属性。

        关掉的效果（``hs_robot.py:358``）：
          * 不再做 ``get_r(20..25)`` 共 6 次 IO 读（每帧省 6 次通信）
          * ``get_observation()`` 不再放 ``force_torque`` 键
          * 我们的 ``state_from_obs`` 于是填 0 —— 维度照样占位
        """
        want = bool(self.cfg.record.collect_force_torque)
        assert self.robot is not None
        if not hasattr(self.robot, "force_sensor_enabled"):
            logger.warning(
                "HsRobot 没有 force_sensor_enabled 属性（fork 版本可能不同）；"
                "力开关未生效，将按 fork 默认行为读取"
            )
            return
        self.robot.force_sensor_enabled = want
        if want:
            logger.info("力/力矩：采集（get_r(20..25)，6 维）→ state[6:12]")
        else:
            logger.info("力/力矩：不采集（已关闭 IO 读取，state[6:12] 恒填 0）")

    def close(self) -> None:
        for name, dev in (("keyboard", self.keyboard), ("spacemouse", self.spacemouse), ("robot", self.robot)):
            if dev is None:
                continue
            try:
                dev.disconnect()
            except Exception as exc:  # 断开失败不应掩盖真正的错误
                logger.warning("断开 %s 失败: %s", name, exc)
        self._connected = False

    def __enter__(self) -> "Hardware":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- 便捷读取 --------------------------------------------------------

    def read_raw_obs(self) -> Mapping[str, Any]:
        """读一次 ``get_observation()``。

        ⚠️ **一帧只调一次**。``get_observation()`` 会读 HRC 位姿 + 每路相机
        ``async_read()``（若 ``collect_force_torque=True`` 还要加 6 次 ``get_r(20..25)``），
        重复调用既慢，又会让"图像与 state 来自不同时刻"。所以主循环读一次，
        再同时喂给 :meth:`preprocess_images` 和 :func:`features.state_from_obs`。
        """
        assert self.robot is not None
        return self.robot.get_observation()

    def preprocess_images(self, obs: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        """从 **同一次** obs 里切出所有相机并跑统一的预处理，返回 ``{短名: HWC uint8}``。

        ⚠️ ``HsRobot.get_observation()`` 里的相机键是**短名**（"front"/"wrist"），
        不是 ``observation.images.front``。这里做一次显式映射。
        """
        crop = self.cfg.record.crop_params or {}
        out: Dict[str, np.ndarray] = {}
        for cam in self.cfg.camera_keys:
            if cam not in obs:
                raise KeyError(
                    f"get_observation() 没有相机 '{cam}'；实际键={sorted(obs.keys())}。"
                    "请检查 robot.cameras 的键名与相机是否真的连上。"
                )
            box = crop.get(cam)
            out[cam] = preprocess_image(
                np.asarray(obs[cam]),
                crop_box=tuple(box) if box else None,
                size=self.cfg.record.image_size,
            )
        return out

    def reset_to_start(self) -> None:
        """把机器人送回起始位姿（每条形之前调用）。"""
        assert self.robot is not None
        ok = self.robot.reset_to_position(
            list(self.cfg.record.reset_position), gripper_state=float(self.cfg.record.reset_gripper)
        )
        if not ok:
            logger.warning("reset_to_position 返回 False —— 复位可能没成功，这条数据要留意")


# ---------------------------------------------------------------------------
# 自检（不需要硬件）
# ---------------------------------------------------------------------------


def selftest_imports() -> List[str]:
    """逐项 import，返回失败项清单（空 = 全通过）。

    用途：在机器人主机上先跑 ``collect.py --selftest``，确认所有驱动可 import，
    再去碰硬件。避免"连上机器人才发现某个 import 路径不对"。
    """
    checks = [
        ("lerobot.robots.hs_robot", "HsRobot, HsRobotConfig"),
        ("lerobot.robots.utils", "make_robot_from_config"),
        ("lerobot.cameras.realsense.configuration_realsense", "RealSenseCameraConfig"),
        ("lerobot.teleoperators.spacemouse.configuration_spacemouse", "SpaceMouseTeleopConfig"),
        ("lerobot.teleoperators.spacemouse.teleop_spacemouse", "SpaceMouseTeleop"),
        ("lerobot.teleoperators.keyboard.configuration_keyboard", "KeyboardEndEffectorTeleopConfig"),
        ("lerobot.teleoperators.keyboard.teleop_keyboard", "KeyboardEndEffectorTeleop"),
        ("lerobot.teleoperators.utils", "TeleopEvents"),
        ("lerobot.datasets.lerobot_dataset", "LeRobotDataset"),
    ]
    failures: List[str] = []
    for mod, attrs in checks:
        try:
            m = __import__(mod, fromlist=["*"])
            for a in [x.strip() for x in attrs.split(",")]:
                if not hasattr(m, a):
                    failures.append(f"{mod}.{a} 不存在")
        except Exception as exc:
            failures.append(f"{mod}: {type(exc).__name__}: {exc}")
    return failures
