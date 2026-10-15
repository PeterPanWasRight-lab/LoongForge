# real_robot_collect —— 独立 LeRobot v3.0 真机采集模块

> 2026-10-13 ｜ 从 HIL-SERL 的采集路径独立出来的最小采集器。
> **只依赖 lerobot 驱动层**（hs_robot / 相机 / SpaceMouse / 键盘 / LeRobotDataset），
> **不依赖** `gym_manipulator`、`RobotEnv`、`reward_classifier`、处理器流水线，也不依赖 LoongForge。

---

## 0. 先看这一节：这个模块解决什么问题

前一代采集走的是 HIL-SERL 的 `gym_manipulator --mode record`。它是为 **RL gym 环境**写的，
借来做 SFT 数据有四个结构性副作用：

| # | 前一代（gym_manipulator record） | 本模块 |
|---|---|---|
| 1 | **每条都 `save_episode()`**，成功与否只反映在 `next.reward` 里 ⇒ 失败演示混进 SFT | **只落成功条**，超时/按 q 的直接 `clear_episode_buffer()` ⇒ 数据集天然干净 |
| 2 | 产物带 `next.reward` / `next.done` / `next.truncated` / `complementary_info.*` | frame 的键集合由 `features.py` 显式声明 + `add_frame` 严格校验 ⇒ **不可能多出字段** |
| 3 | 需要 RobotEnv + AssemblyMonitor + 处理器流水线 + TimeLimit 等一整套 RL 组件 | 复位 = `HsRobot.reset_to_position()` 一次调用；无环境、无奖励、无时间限制处理器 |
| 4 | `next.done = terminated or truncated`，**无法区分成功与超时**（`gym_manipulator.py:1610`） | 结局在循环内就地判定（`success`/`rerecord`/`abort`/`timeout`），不依赖事后解析 |

**行为的等价之处**：驱动器完全一样（同一台机械臂、同样两路 RealSense、同一个 SpaceMouse），
键位也一样（`s`/`r`/`q`，来自 `teleop_keyboard.py:394-408`），所以**现场手感不变**。

---

## 1. 目录结构

```
real_robot_collect/
├── __init__.py          导出公开 API
├── config.py            配置 dataclass + JSON 加载 + **前置校验**（不含硬件）
├── features.py          ★ 唯一决定数据集 schema 的地方（特征声明 + 帧组装）
├── hw.py                硬件组装（robot/相机/遥操）+ **图像预处理**（执行端必须共用）
├── recorder.py          采集主循环 + s/r/q 事件 + 只落成功条
├── collect.py           CLI 入口（--selftest / --dry-run / 正式采集）
├── verify.py            采集后验收（版本/维度/帧数/stats/逐条体检/RL 残留）
└── configs/peg_v1.json  轴孔装配的示例配置（数值取自 102.56 上正在用的 record.json）
```

---

## 2. 数据契约（`features.py` 说了算）

数据集里**只会出现**下面这些键（`task` 是 `add_frame` 强制要求，不算 feature）：

| 键 | dtype | shape | 含义 |
|---|---|---|---|
| `observation.images.front` | `image`（PNG 内嵌 parquet） | `(3, 224, 224)` | 前置相机 |
| `observation.images.wrist` | 同上 | `(3, 224, 224)` | 腕部相机 |
| `observation.state` | `float32` | `(12,)` | **[x,y,z (mm), rx,ry,rz (deg), Fx,Fy,Fz,Tx,Ty,Tz]** —— **绝对量** |
| `action` | `float32` | `(3,)` | **[Δx, Δy, Δz]** —— **相对增量**（控制器归一化输入，非物理量） |
| `task` | `str` | — | `"insert the peg into the hole"`（进 pi05 的 prompt） |

索引类（`timestamp` / `frame_index` / `episode_index` / `index` / `task_index`）
由 `LeRobotDataset` 的 `DEFAULT_FEATURES` 自动管理，不需要我们写。

**三条容易踩的约束**：

1. ⭐ **力那 6 维永远占位** —— `record.collect_force_torque` 只决定它是真实读数还是 0，
   **维度不随它变化**。所以将来打开力采集，训练侧 `model.state_dim` 和执行端组 state
   的代码都不用跟着改，**也不用重采**：

   | 开关 | state[6:12] | 机械臂 IO |
   |---|---|---|
   | `false`（当前默认） | 恒 `0.0` | **不读**（每帧省 6 次 `get_r`） |
   | `true` | `get_r(20..25)` 真实力/力矩 | 读（日志会打印 F/T，便于现场确认传感器通不通） |

   拼接顺序与历史数据集 `hs_robot_assembly_v1` 的 `[位姿 6, F/T 6]` 前缀兼容。
   **切开关不用重采**：打开就能用；关着时那 6 维是常数，对 prompt 无副作用。
2. **夹爪维由 `record.include_gripper` 同时开关 state 和 action** ——
   源码里 `use_gripper` 一份开关管两处（`gym_manipulator.py:543` 与 `:697`）。
   **本线当前取 `false`** ⇒ state **12** 维（位姿 6 + 力 6）/ action **3** 维
   —— 与历史数据集 `hs_robot_assembly_v1` 的 state 完全一致，也正好等于训练时要设的
   `model.action_dim=3`。

   ⚠️ **为什么关掉**：夹爪目前**没有任何设备能控制它**（`keyboard.use_gripper: false`，
   且 `recorder.py` 只从 SpaceMouse 取动作、从不调 `keyboard.get_action()`）。
   夹爪全程不动 ⇒ 那一维恒定 ⇒ `q99 == q01` ⇒ 归一化后是常数
   （`verify.py` 会判为**必须修的问题**）。

   **将来要做夹爪演示，三件事一起做**：① `include_gripper: true`；
   ② `keyboard.use_gripper: true`；③ **补 `recorder.py`：把 `keyboard.get_action()["gripper"]`
   合并进 `teleop_action`**（少了 ③，前两个开了也不生效）。
   注意：改动会变 state/action 维度 ⇒ **旧数据不能混用**，要重采。
3. **相机命名决定视野槽位** —— LoongForge 读图像时按 **key 字典序**排序
   （`pi05_transform.py:133`），所以 `front < wrist` 的顺序是**由名字决定的**，
   训练与执行两侧必须同名同序。`--selftest` 会把字典序打出来。

---

## 2.5 🚨 使能 —— 不做这步，机械臂一步都不会动

`hs_robot.py:241` 把 `self.hsrobot.enable()` **注释掉了**，只留一句
`logger.info("Robot enabled successfully")`（**假日志**）。后果：

| | 未使能时 |
|---|---|
| 读位姿 `get_cartesian_position()` | ✅ 正常 —— **完全看不出问题** |
| `move_to_cartesian_position()` / `send_action()` | ❌ **全部无效，且不报错** |
| 日志 | 零 ERROR / 零 WARNING（完全静默） |

2026-10-13 实测症状：复位流程（含"先垂直往上走 20 mm"）零位移、推 SpaceMouse 零位移，
而 `_cached_target_position` 却在正常累积 ⇒ **指令发出去了，只是没人执行**。

⇒ 本模块在 `Hardware.connect()` 里补上 `hsrobot.enable()`，由
`robot.enable_on_connect` 控制（**默认 true，别关**）。使能本身不产生运动
（伺服就位、保持当前位姿）。

**若使能后仍然不动**，按这个顺序查：**示教器是否处于【远程 / 自动】模式**（手动模式不接受
程序指令）→ 是否处于急停 → 伺服是否上电。

### 诊断工具

| 脚本 | 用途 |
|---|---|
| `enable_probe.py` | 使能 + 运动通道验证（`--move-test` 会真动 5 mm 再回来；`--no-enable` 做对照） |
| `reset_preview.py` | 复位预演：6 维边界检查 + 复位前后位移（换工位/改 `reset_position` 后先跑） |
| `verify.py` | 数据集验收（`total_frames == parquet rows`、stats 合理性、官方 lerobot 能否加载） |
| `visualize.py` | 单条 episode 可视化（不依赖 lerobot） |

---

## 3. 用法（前三步都不碰硬件）

```bash
cd /home/gm/Project/Test/lerobot_test          # fork 的仓库根目录
export PYTHONPATH=$PWD:$PYTHONPATH

# ⓪ 纯逻辑回归测试（只依赖 numpy，不需要 lerobot/torch/硬件）—— 39 项断言
python real_robot_collect/test_logic.py

# ① 环境自检：import 驱动 + 打印将生效的配置与 features
python -m real_robot_collect.collect --selftest --config configs/peg_v1.json

# ② 空跑：真建一个空数据集 + 塞一帧假数据验证键集合，然后删掉
python -m real_robot_collect.collect --dry-run --config configs/peg_v1.json

# ③ 正式采集（连硬件）
python -m real_robot_collect.collect --config configs/peg_v1.json --log-file /tmp/collect.log
```

**⓪ 已在开发机实测通过 39/39**（覆盖：配置解析、8 个必须被拦住的错配置、
**维度公式自洽**（state = 6 + 6 + 夹爪?）、特征声明（逐位 names：前 6 位位姿 / 第 7-12 位力）、
帧按键/形状/dtype 校验、状态是绝对量、力开关的两个分支（真读 / 恒 0，且**维度不变**）、
`include_gripper` 两个取值的显式路径（12/3 与 13/4）、负夹爪 clamp、
以及 8 个"多写字段、形状不符或开关与数据不一致必须报错"的负例）。
这是本模块**唯一被真实验证过的部分**。

采集后验收（**在采集机或训练机都能跑**）：

```bash
python -m real_robot_collect.verify --dataset datasets/hs_robot_peg_v1_20261013_150000
```

**退出码**：`0` 正常 · `2` 配置/环境错误 · `3` 被 Ctrl+C 中断（已落盘的数据都在）。

---

## 4. 现场键位（与 HIL-SERL 一致）

| 操作 | 键 / 设备 | 效果 |
|---|---|---|
| 平移 | SpaceMouse 推拉 | 机器人在基座系平移 |
| 姿态 | —— | **无效**（`rotation_scale: 0`，且 action 空间不含姿态维） |
| 夹爪 | —— | **无效**（`keyboard.use_gripper: false`，且 `recorder.py` 不读键盘动作 ⇒ **只改配置也打不开**，需补代码） |
| **成功** | **`s`** | 结束本条并落盘 |
| **重录** | **`r`** | 结束本条并**丢弃**（`clear_episode_buffer`），不进数据集 |
| **放弃** | `q` | 结束本条并丢弃，额外打失败标记 |
| 备用移动 | —— | **无效**（本模块只从 SpaceMouse 取动作；键盘仅供 s/r/q 事件） |
| 急停 | SpaceMouse 按钮 `0`+`1` 同按 1 s | 硬件层急停 |
| 全局停止 | `Ctrl+C` | 丢弃当前条，**已落盘的条全部保留** |

> 为什么必须有键盘设备：`SpaceMouseTeleop.get_episode_end_status()` 是**空实现**
> （`spacemouse_utils.py:331` 直接 `return None`）⇒ 光靠 SpaceMouse 触发不了 s/r/q。
> `config.py` 里对 `keyboard.enabled=false` 做了硬拦截。

**每条的标准动作**：复位（自动，`reset_position`）→ 人摆件 → 推/压完成装配 → `s`；
录废了按 `r`。

---

## 5. 部署（放 LoongForge 仓库，跑在边缘端）

这个模块的**代码位置**与**运行位置**是分开的：

```
LoongForge/real_robot_collect/            ← 你说的位置（版本管理）
        │  rsync / scp
        ▼
102.56:/home/gm/Project/Test/lerobot_test/real_robot_collect/   ← 真正运行的位置
        （因为 hs_robot / 相机 / SpaceMouse 都挂在 102.56）
```

⚠️ **LoongForge 仓库所在机器（107.56）跑不了它** —— 那里没有 hs_robot 驱动、没有相机、没有机械臂。
同步命令（在 107.56 上执行）：

```bash
rsync -av --progress \
  /home/gm/data/LoongForge/LoongForge/real_robot_collect/ \
  gm@172.16.102.56:/home/gm/Project/Test/lerobot_test/real_robot_collect/
```

采集完的数据按 `results/50条采集Runbook_2026-10-13.md` §6 反向传回 107.56 的
`/home/gm/data/LoongForge/artifacts/`。**注意**：`configs/peg_v1.json` 里 `dataset.root`
写的是 102.56 上的绝对路径，传到 107.56 前不用改（那份只管采集端）。

---

## 6. ⚠️ 未验证项（务必先跑 `--selftest` 和 `--dry-run`）

本模块是**接口级**写的，作者**没有在真实硬件上跑过**（机械臂/相机/SpaceMouse 都在现场）。
以下每一项都可能在首跑时暴露，按可能性从高到低：

| # | 风险点 | 首跑怎么暴露 | 兜底 |
|---|---|---|---|
| 1 | import 路径与 fork 实际不符（fork 是魔改版） | `--selftest` 直接列出失败项 | 按报错改 `hw.py:selftest_imports` 里的路径 |
| 2 | `HsRobotConfig` / `SpaceMouseTeleopConfig` 的字段名与我读到的版本不一致 | `--selftest` 之后构造配置时抛 `TypeError` | 报错会指出多余/缺失字段名 |
| 3 | `KeyboardEndEffectorTeleopConfig` 需要额外字段 | 同上 | 同上 |
| 4 | `LeRobotDataset.create` 的参数名不同 | `--dry-run` 抛 `TypeError` | 同上 |
| 5 | `add_frame` 的形状校验不接受 HWC | `--dry-run` 的单帧校验会失败 | `validate_feature_image_or_video` 已确认接受 HWC，但以防万一 |
| 6 | 相机的键名不是 `front`/`wrist` | 运行时 `KeyError`，报错会打出实际键 | 改配置的相机名 |
| 7 | fps 达不到 15（PNG 编码/IO 慢） | 日志里 `eff_fps` 低于设定值 | 提高 `image_writer_threads`，或降 fps |
| 8 | 复位的 `reset_position` 与现场不符 | 机器人跑向错误位置 | **先空跑一次不给动作**，确认复位点 |

**建议的首跑顺序**：`--selftest` → `--dry-run` → 把 `num_episodes` 改成 **1**、`control_time_s` 改 **5**
跑一条最短的 → 再上正式参数。

---

## 7. 与执行端的接口（重要）

`hw.preprocess_image()` 是**采集与执行必须共用**的函数：

```python
from real_robot_collect.hw import preprocess_image
img = preprocess_image(obs["front"], crop_box=(150, 250, 250, 250), size=224)
```

它复刻了 fork 处理器链的顺序（uint8 HWC → [0,1] CHW float → crop → resize → uint8 HWC），
`crop_box` 的顺序是 **(top, left, height, width)**（源码：`F.crop(img, top, left, h, w)`）。
执行端的 runner **不要自己再写一遍** —— 采集与执行的 crop/插值/round 不一致会引入
静默的分布偏移，这正是 `prefer_controller_ee` 那类问题的同一个教训。

---

## 8. 已知的设计取舍

- **`use_videos: false`（PNG 内嵌 parquet）**：自包含、无视频编码依赖
  （边缘端不需要 ffmpeg/torchcodec），LoongForge 侧的 lerobot 0.5.0 对 `dtype:"image"`
  是一等公民（`lerobot_dataset.py:1211` 把 `["image","video"]` 并列处理）。
  代价是体积比 mp4 大、parquet 单文件较大。
- **不加姿态增量**：`action_dim=3` 是 fork 里**硬编码**的（`gym_manipulator.py:683`
  注释「暂时使用3维x、y、z与夹爪」）。要加姿态必须同时改代码与 action 空间，
  **并且要重采数据**。
- **`next.done` 不用来判成功**：本模块不用它；前一代路径里它是
  `terminated or truncated`（`gym_manipulator.py:1610`），成功与超时都置 True。
