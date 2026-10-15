# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""独立 LeRobot v3.0 真机采集模块（HS 臂）。

与 HIL-SERL 解耦：只依赖 lerobot 的**驱动层**（hs_robot / 相机 / SpaceMouse /
键盘 / LeRobotDataset），不依赖 gym_manipulator、RobotEnv、reward_classifier、
处理器流水线，也不依赖 LoongForge。

公开 API::

    from real_robot_collect import load_config, run_recording, Hardware, build_features
"""

from .config import (
    CameraCfg,
    CollectConfig,
    DatasetCfg,
    KeyboardCfg,
    RecordCfg,
    RobotCfg,
    SpaceMouseCfg,
    dump_config,
    load_config,
)
from .features import build_features, build_frame, state_from_obs
from .hw import Hardware, preprocess_image
from .recorder import RecordSummary, finalize_dataset, record_one, run_recording

__all__ = [
    "CameraCfg",
    "CollectConfig",
    "DatasetCfg",
    "Hardware",
    "KeyboardCfg",
    "RecordCfg",
    "RecordSummary",
    "RobotCfg",
    "SpaceMouseCfg",
    "build_features",
    "build_frame",
    "dump_config",
    "finalize_dataset",
    "load_config",
    "preprocess_image",
    "record_one",
    "run_recording",
    "state_from_obs",
]

__version__ = "0.1.0"
