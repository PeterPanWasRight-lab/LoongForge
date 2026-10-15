"""real_robot_collect 的可离线验证部分（config / features）回归测试。

不依赖 lerobot、torch、硬件 —— 只测"纯逻辑"那两块：
  * config.py   配置解析 + 前置校验（8 个该被拦住的错配置）
  * features.py 特征声明 + 帧组装（键集合、形状、dtype、绝对/相对量透传）

用法（任意 CWD）::

    python real_robot_collect/test_logic.py

现场首跑前的顺序建议：test_logic.py -> collect.py --selftest -> --dry-run -> 正式采集。
退出码 0 = 全通过。
"""

import json
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from real_robot_collect.config import dump_config, load_config  # noqa: E402
from real_robot_collect.features import (  # noqa: E402
    action_to_robot,
    action_to_vector,
    build_features,
    build_frame,
    state_from_obs,
)

CFG_PATH = str(_HERE / "configs" / "peg_v1.json")
fails, oks = [], []


def chk(name, cond, extra=""):
    (oks if cond else fails).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")


def with_patch(patch):
    base = json.loads(open(CFG_PATH).read())
    for sec, kv in patch.items():
        base[sec] = {**base.get(sec, {}), **kv}
    open("/tmp/bad.json", "w").write(json.dumps(base))
    return "/tmp/bad.json"


print("=== 1) load_config ===")
cfg = load_config(CFG_PATH)
chk("解析成功", cfg is not None)
print("     camera_keys =", cfg.camera_keys, "| 字典序 =", sorted(cfg.camera_keys))
print(f"     state_dim = {cfg.state_dim}  action_dim = {cfg.action_dim}")
print(f"     include_gripper = {cfg.record.include_gripper}   "
      f"collect_force_torque = {cfg.record.collect_force_torque}")
chk("state_dim 公式自洽 (6 + 6 + 夹爪?)",
    cfg.state_dim == 6 + 6 + (1 if cfg.record.include_gripper else 0), f"实际 {cfg.state_dim}")
chk("action_dim 公式自洽 (3 + 夹爪?)",
    cfg.action_dim == (4 if cfg.record.include_gripper else 3), f"实际 {cfg.action_dim}")
print("     dataset dir =", cfg.dataset_root_abs() / cfg.dataset.repo_id)
dump_config(cfg, "/tmp/eff.json")
chk("dump_config 可回读", json.load(open("/tmp/eff.json"))["record"]["fps"] == cfg.record.fps)

print("=== 2) 前置校验：下列错配置都必须被拦住 ===")
bad_cases = [
    ("rotation_scale != 0", {"spacemouse": {"rotation_scale": 0.5}}),
    ("task 为空", {"dataset": {"task": "   "}}),
    ("reset_position 位数错", {"record": {"reset_position": [1, 2, 3]}}),
    ("crop 键名写成 observation.images.front",
     {"record": {"crop_params": {"observation.images.front": [1, 2, 3, 4]}}}),
    ("control_mode != 0", {"robot": {"control_mode": 1}}),
    ("ip 为空", {"robot": {"ip": ""}}),
    ("robot 段有未知字段", {"robot": {"nonsense": 1}}),
    ("num_episodes <= 0", {"dataset": {"num_episodes": 0}}),
]
for label, patch in bad_cases:
    p = with_patch(patch)
    try:
        load_config(p)
        chk(label, False, "<- 应该报错却没报")
    except Exception as e:
        chk(label, True, f"-> {type(e).__name__}: {str(e)[:60]}")

print("=== 3) build_features / build_frame 正例 ===")
feats = build_features(cfg.camera_keys, cfg.record.image_size, cfg.state_dim, cfg.action_dim, False)
for k, v in feats.items():
    print(f"     {k:<32} {v['dtype']:<6} {tuple(v['shape'])}")
chk("features 恰好 4 项（无 RL 字段）", len(feats) == 4)
chk("图像 dtype == image", all(v["dtype"] == "image" for k, v in feats.items() if "images" in k))
chk("features 形状 == cfg.state_dim", tuple(feats["observation.state"]["shape"]) == (cfg.state_dim,),
    str(tuple(feats["observation.state"]["shape"])))
chk("state names 前 6 位 = 位姿", feats["observation.state"]["names"][:6] ==
    ["x", "y", "z", "rx", "ry", "rz"], str(feats["observation.state"]["names"]))
chk("state names 第 7-12 位 = 力/力矩", feats["observation.state"]["names"][6:12] ==
    ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])

img = np.zeros((224, 224, 3), np.uint8)
obs = {f"loc_{i}.pos": float(10 * i) for i in range(6)}
obs["gripper.pos"] = 1.0
obs["force_torque"] = np.arange(6, dtype=np.float32) + 100.0  # 100..105，便于辨认

# 用**当前配置**的 include_gripper 组装，保证与 features 维度一致
s_ft = state_from_obs(obs, cfg.record.include_gripper, collect_force_torque=True)
chk(f"state shape == ({cfg.state_dim},) 与配置一致", s_ft.shape == (cfg.state_dim,), str(s_ft.shape))
chk("力维透传（100..105）", bool(np.allclose(s_ft[6:12], obs["force_torque"])),
    str(np.round(s_ft[6:12], 1)))

# 与当前配置无关地显式覆盖 include_gripper=True（13 维）路径
s13 = state_from_obs(obs, True, collect_force_torque=True)
chk("include_gripper=True ⇒ 13 维、末位是夹爪",
    s13.shape == (13,) and float(s13[-1]) == 1.0, str(s13.shape))

s_no = state_from_obs(obs, cfg.record.include_gripper, collect_force_torque=False)
chk("关掉力后维度**不变**（开关只影响内容）", s_no.shape == s_ft.shape == (cfg.state_dim,), str(s_no.shape))
chk("关掉力后 state[6:12] 全 0", bool(np.allclose(s_no[6:12], 0.0)), str(np.round(s_no[6:12], 1)))

acts = {"delta_x": .5, "delta_y": -.2, "delta_z": 0., "delta_roll": 0.,
        "delta_pitch": 0., "delta_yaw": 0., "gripper": .3}
av = action_to_vector(acts, cfg.record.include_gripper)
fr = build_frame({"front": img, "wrist": img}, s_ft, av, cfg.dataset.task, feats)
chk(f"action shape == ({cfg.action_dim},)", fr["action"].shape == (cfg.action_dim,),
    str(fr["action"].shape))
chk("frame 键 == features + task", set(fr) == set(feats) | {"task"}, str(sorted(fr)))
chk("位姿绝对值透传未改动（x=0, y=10）",
    float(fr["observation.state"][0]) == 0.0 and float(fr["observation.state"][1]) == 10.0)
chk("负夹爪被 clamp 到 0", action_to_robot({"gripper": -1.0})["gripper.pos"] == 0.0)
chk("robot dict 含 6 个 delta 键 + gripper.pos",
    set(action_to_robot(acts)) == {"delta_x", "delta_y", "delta_z", "delta_roll",
                                   "delta_pitch", "delta_yaw", "gripper.pos"})

print("=== 4) 负例：多写字段 / 形状不符 / 开关与数据不一致 必须报错 ===")
neg_cases = [
    ("多一个 RL 字段 next.reward",
     lambda: build_frame({"front": img, "wrist": img}, s_ft, av, cfg.dataset.task,
                         {**feats, "next.reward": {"dtype": "float32", "shape": (1,), "names": None}})),
    (f"state 维度 5 != {cfg.state_dim}",
     lambda: build_frame({"front": img, "wrist": img}, np.zeros(5, np.float32), av, cfg.dataset.task, feats)),
    ("图像是 float 未转 uint8",
     lambda: build_frame({"front": img.astype(np.float32), "wrist": img}, s_ft, av, cfg.dataset.task, feats)),
    ("缺一路相机",
     lambda: build_frame({"front": img}, s_ft, av, cfg.dataset.task, feats)),
    ("task 为空",
     lambda: build_frame({"front": img, "wrist": img}, s_ft, av, "  ", feats)),
    ("obs 缺 loc_5.pos",
     lambda: state_from_obs({k: v for k, v in obs.items() if k != "loc_5.pos"}, True)),
    ("开着力开关但 obs 里没有 force_torque",
     lambda: state_from_obs({k: v for k, v in obs.items() if k != "force_torque"}, True, True)),
    ("force_torque 维度不是 6",
     lambda: state_from_obs({**obs, "force_torque": np.zeros(3, np.float32)}, True, True)),
]
for label, fn in neg_cases:
    try:
        fn()
        chk(label, False, "<- 应该报错却没报")
    except Exception as e:
        chk(label, True, f"-> {type(e).__name__}")

print("=== 5) include_gripper=False 的显式路径 ⇒ 12 / 3 维（力那 6 维仍占位）===")
s12 = state_from_obs(obs, False, collect_force_torque=True)
a3 = action_to_vector(acts, False)
f12 = build_features(cfg.camera_keys, 224, 12, 3, False)
fr12 = build_frame({"front": img, "wrist": img}, s12, a3, cfg.dataset.task, f12)
chk("state 12 维 == 位姿6 + 力6", fr12["observation.state"].shape == (12,))
chk("12 维时最后一位是 Tz（不是夹爪）", f12["observation.state"]["names"][-1] == "Tz")
chk("action 3 维", fr12["action"].shape == (3,))
chk("action_names=3 且不含 gripper", f12["action"]["names"] == ["delta_x", "delta_y", "delta_z"])

print(f"\n===== 结果: {len(oks)} PASS / {len(fails)} FAIL =====")
if fails:
    print("失败项:")
    for f in fails:
        print("   -", f)
sys.exit(1 if fails else 0)
