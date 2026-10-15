# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""配置加载 —— 只用标准库，不依赖 draccus / loongforge。

设计取舍：
    故意 **不用** draccus 解析。理由有三：
      1. draccus 的 ChoiceRegistry 会在缺字段时抛奇怪的错，现场 debug 困难；
      2. 本模块只支持少量机器人/相机/遥操组合，显式字段比动态注册更好读；
      3. 与 HIL-SERL 解耦 ⇒ 不共用它那套 env/processor/reward 的配置结构。

JSON 结构见 ``configs/peg_v1.json``。所有字段都在这里显式声明 + 校验。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 子配置
# ---------------------------------------------------------------------------


@dataclass
class CameraCfg:
    """单台相机的声明。type 目前只支持 intelrealsense / opencv。"""

    name: str
    type: str
    width: int = 640
    height: int = 480
    fps: int = 30
    serial_number_or_name: Optional[str] = None   # realsense 用
    index_or_path: Optional[int] = None           # opencv 用
    rotation: Optional[str] = None                # 例如 "ROTATE_90"；None = 不旋转


@dataclass
class RobotCfg:
    type: str = "hs_robot"
    ip: str = ""
    port: int = 23234
    control_mode: int = 0            # 0 = 笛卡尔（本模块只支持 0）
    gripper_io_port: int = 19
    #: connect 后是否调用底层使能 ``hsrobot.enable()``。
    #: 🚨 必须保持 true —— ``hs_robot.py:241`` 把 ``self.hsrobot.enable()`` **注释掉了**，
    #: 不使能时读位姿正常、但**所有运动指令静默无效**（2026-10-13 实测：
    #: 复位/上抬 20mm/推杆全程无位移，日志零报错）。
    enable_on_connect: bool = True
    workspace_limits: Optional[Dict[str, float]] = None
    cameras: List[CameraCfg] = field(default_factory=list)


@dataclass
class SpaceMouseCfg:
    """3D 鼠标配置。字段名与 fork 的 ``SpaceMouseTeleopConfig`` 对齐。"""

    enabled: bool = True
    deadzone: float = 0.01
    rotation_deadzone: float = 0.04
    filter_alpha: float = 0.5
    position_scale: float = 600.0
    #: ⚠️ 置 0 ⇒ 摇杆不控姿态，action 里就不含姿态增量（本线选择）
    rotation_scale: float = 0.0
    max_position_offset: float = 30.0
    max_rotation_offset: float = 30.0
    update_rate_hz: float = 500.0
    max_angle_delta_deg: float = 10.0
    use_gripper: bool = False
    button0_mode: str = "translate"
    button1_mode: str = "rotate"
    emergency_buttons: List[int] = field(default_factory=lambda: [0, 1])
    emergency_hold_time: float = 1.0
    output_mode: str = "incremental"
    axis_mapping: List[int] = field(default_factory=lambda: [1, 0, 2, 5, 3, 4])
    axis_inversion: List[int] = field(default_factory=lambda: [-1, 1, 1, -1, 1, 1])


@dataclass
class KeyboardCfg:
    """键盘事件设备。s=成功 / r=重录 / q=放弃（``teleop_keyboard.py:394-408``）。"""

    enabled: bool = True
    type: str = "keyboard_ee"
    step_size: float = 2.5
    use_gripper: bool = False


@dataclass
class RecordCfg:
    fps: int = 15
    image_size: int = 224
    #: False ⇒ dtype "image"（PNG 内嵌 parquet，自包含、无视频编码依赖）
    use_videos: bool = False
    #: 单条 episode 的墙钟上限（秒）。**超时的 episode 默认不落盘**（见 keep_failed）
    control_time_s: float = 25.0
    #: 每条形之间的复位等待（秒）
    reset_time_s: float = 1.5
    #: 复位目标位姿（6 个数，mm + deg）。必须与训练/执行一致，中途不许改
    reset_position: List[float] = field(default_factory=list)
    #: 复位时的夹爪指令（0.0 = 张开，1.0 = 闭合），遵循 ``HsRobot`` 的 >=0 约定
    reset_gripper: float = 0.0
    #: 采集时的裁剪框，顺序是 (top, left, height, width)
    #  —— 源码依据：``F.crop(image, *params)`` = torchvision.crop(img, top, left, h, w)
    #     （fork 的 ``hil_processor.py:233``）
    crop_params: Optional[Dict[str, List[int]]] = None
    #: 是否记录夹爪维（同时影响 state 与 action 的维度）
    include_gripper: bool = True
    #: 是否**采集**力/力矩（`get_r(20..25)` 六维）。
    #:
    #: ⚠️ 关掉**不会**改变 state 的维度 —— 那 6 维依然占位，只是恒填 0。
    #: 这样 state 的 shape 不随开关变化，训练侧与执行端都不必跟着改；
    #: 同时会关闭机械臂的 IO 读取（省 6 次 IO/帧，也不受传感器误读影响）。
    collect_force_torque: bool = False
    #: 图像写入线程数。0 = 同步写（2 路 224×224 PNG 可能拖慢 15 fps，建议 4）
    image_writer_threads: int = 4
    #: 是否保留失败（超时/按 q）的 episode。SFT 建议 False —— 只要成功演示
    keep_failed: bool = False
    #: 显示相机预览（需要 DISPLAY；ssh 无 X 时会自动跳过）
    display: bool = False


@dataclass
class DatasetCfg:
    repo_id: str = "hs_robot_collect"
    #: 相对路径时相对于“运行目录”；建议写绝对路径以免落错地方
    root: str = "datasets"
    task: str = ""
    num_episodes: int = 50
    robot_type: str = "hs_robot"
    #: 是否给 repo_id 追加时间戳后缀（沿用现有习惯：repo_id_YYYYMMDD_HHMMSS）
    timestamp_suffix: bool = True


@dataclass
class CollectConfig:
    robot: RobotCfg
    record: RecordCfg
    dataset: DatasetCfg
    spacemouse: SpaceMouseCfg = field(default_factory=SpaceMouseCfg)
    keyboard: KeyboardCfg = field(default_factory=KeyboardCfg)
    #: 保存到磁盘的原始 JSON（便于随数据集归档）
    source_path: Optional[str] = None

    # ------------------------------------------------------------------
    # 派生量
    # ------------------------------------------------------------------

    @property
    def camera_keys(self) -> List[str]:
        return [c.name for c in self.robot.cameras]

    @property
    def state_dim(self) -> int:
        """``[位姿 6] + [力/力矩 6] + [夹爪 0/1]`` ⇒ **12 或 13**（恒定）。

        力/力矩那 6 维**永远占位**，与 ``record.collect_force_torque`` 无关 ——
        开关只决定它是真实读数还是 0。理由：state 的 shape 若随开关变化，
        训练侧的 ``model.state_dim``、执行端组 state 的代码、以及已采数据
        就都要跟着改；固定维度可以让「先采、后启用」零成本切换。

        拼接顺序（必须与 :func:`features.state_from_obs` 一致，也与历史数据集
        ``hs_robot_assembly_v1`` 的 ``[位姿 6, F/T 6]`` 前缀兼容）。
        """
        return 6 + 6 + (1 if self.record.include_gripper else 0)

    @property
    def action_dim(self) -> int:
        return 4 if self.record.include_gripper else 3

    def dataset_root_abs(self) -> Path:
        """数据集根目录的绝对路径（``root`` 为相对路径时相对 CWD）。"""
        return Path(self.dataset.root).expanduser().resolve()


# ---------------------------------------------------------------------------
# 加载与校验
# ---------------------------------------------------------------------------


def _as_camera_cfg(name: str, d: Dict[str, Any]) -> CameraCfg:
    return CameraCfg(
        name=name,
        type=str(d.get("type", "intelrealsense")),
        width=int(d.get("width", 640)),
        height=int(d.get("height", 480)),
        fps=int(d.get("fps", 30)),
        serial_number_or_name=d.get("serial_number_or_name"),
        index_or_path=d.get("index_or_path"),
        rotation=d.get("rotation"),
    )


def load_config(path: str | os.PathLike) -> CollectConfig:
    """从 JSON 读配置并做一轮**前置校验**。校验失败直接抛，不留到硬件阶段。"""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    raw = json.loads(p.read_text())

    # --- robot ---
    r = dict(raw.get("robot") or {})
    if not r:
        raise ValueError("配置缺少 robot 段")
    if not str(r.get("ip", "")).strip():
        raise ValueError("robot.ip 必填（历史配置出现过 3 个 IP，上传前必须 ping 确认）")
    # ⚠️ 必须 pop 掉，否则会落到下面"未支持字段"的检查里（本地实测踩过）
    cmode = int(r.pop("control_mode", 0))
    if cmode != 0:
        raise ValueError(
            f"本模块只支持 robot.control_mode == 0（笛卡尔）。当前 {cmode}；"
            "关节模式的状态位姿键是 joint_i.pos，需要另外适配。"
        )
    cams_raw = r.pop("cameras", {}) or {}
    if not cams_raw:
        raise ValueError("robot.cameras 不能为空（pi05 需要至少 1 路图像，本线是 2 路）")
    robot_cfg = RobotCfg(
        type=str(r.pop("type", "hs_robot")),
        ip=str(r.pop("ip")),
        port=int(r.pop("port", 23234)),
        control_mode=0,
        gripper_io_port=int(r.pop("gripper_io_port", 19)),
        enable_on_connect=bool(r.pop("enable_on_connect", True)),
        workspace_limits=r.pop("workspace_limits", None),
        cameras=[_as_camera_cfg(k, v) for k, v in cams_raw.items()],
    )
    if r:
        raise ValueError(f"robot 段出现未支持字段: {sorted(r)}")

    # --- record ---
    rec_raw = dict(raw.get("record") or {})
    record_cfg = RecordCfg(**rec_raw) if rec_raw else RecordCfg()
    if not record_cfg.reset_position:
        raise ValueError("record.reset_position 必填：每条形都要回到这个位姿，缺了数据分布会不干净")
    if len(record_cfg.reset_position) != 6:
        raise ValueError(f"record.reset_position 必须是 6 个数（mm+deg），当前 {len(record_cfg.reset_position)} 个")
    if record_cfg.fps <= 0:
        raise ValueError("record.fps 必须为正")
    if record_cfg.image_size < 64:
        raise ValueError(f"record.image_size={record_cfg.image_size} 太小；pi05 输入 224，建议直接采 224")
    if record_cfg.crop_params:
        for cam, box in record_cfg.crop_params.items():
            if len(box) != 4:
                raise ValueError(
                    f"crop_params[{cam}] 必须是 4 个数且顺序为 (top, left, height, width)，当前 {box}"
                )
        unknown = set(record_cfg.crop_params) - set(cams_raw.keys())
        if unknown:
            raise ValueError(
                f"crop_params 里的相机名 {sorted(unknown)} 不在 robot.cameras 中；"
                "注意键要用**相机短名**（front），不是 observation.images.front"
            )

    # --- dataset ---
    ds_raw = dict(raw.get("dataset") or {})
    dataset_cfg = DatasetCfg(**ds_raw) if ds_raw else DatasetCfg()
    if not dataset_cfg.task.strip():
        raise ValueError(
            "dataset.task 必填：它会进 pi05 的 prompt（用自然语言，别用下划线串）"
        )
    if dataset_cfg.num_episodes <= 0:
        raise ValueError("dataset.num_episodes 必须为正")

    # --- teleop ---
    sm_raw = dict(raw.get("spacemouse") or {})
    spacemouse_cfg = SpaceMouseCfg(**sm_raw) if sm_raw else SpaceMouseCfg()
    if spacemouse_cfg.rotation_scale != 0.0:
        raise ValueError(
            f"spacemouse.rotation_scale={spacemouse_cfg.rotation_scale} != 0。"
            "本线的 action 空间**不含姿态增量**（features 里没声明），"
            "开着它会让机器人动姿态但数据里记不到 ⇒ 训练/执行不一致。"
            "要控姿态请先改 features.ACTION_NAMES_4 并调 action_dim。"
        )
    kb_raw = dict(raw.get("keyboard") or {})
    keyboard_cfg = KeyboardCfg(**kb_raw) if kb_raw else KeyboardCfg()
    if not keyboard_cfg.enabled:
        raise ValueError(
            "keyboard.enabled=false 时无法触发 s(成功)/r(重录)/q(放弃)，"
            "SpaceMouse 那一侧 get_episode_end_status() 是空实现（return None）。"
            "必须保留键盘事件设备。"
        )

    return CollectConfig(
        robot=robot_cfg,
        record=record_cfg,
        dataset=dataset_cfg,
        spacemouse=spacemouse_cfg,
        keyboard=keyboard_cfg,
        source_path=str(p),
    )


def dump_config(cfg: CollectConfig, path: str | os.PathLike) -> None:
    """把生效配置（含默认值展开后）落盘，随数据集归档，便于复现。"""
    def _dc(o: Any) -> Any:
        if hasattr(o, "__dataclass_fields__"):
            return {k: _dc(getattr(o, k)) for k in o.__dataclass_fields__}
        if isinstance(o, list):
            return [_dc(x) for x in o]
        if isinstance(o, dict):
            return {k: _dc(v) for k, v in o.items()}
        return o

    Path(path).write_text(json.dumps(_dc(cfg), ensure_ascii=False, indent=2))
