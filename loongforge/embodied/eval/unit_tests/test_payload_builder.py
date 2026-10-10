# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-model PayloadBuilder + orchestrator capability matching.

These pin the bit-exact behavior of the refactored eval pipeline against the
original scattered logic (``_build_model_state`` in adapters, the runner's
``_canonical_to_policy_payload``, the server's ``_build_image_input``) so a
regression in any of the three layers is caught here rather than in a
downstream benchmark run.
"""

from __future__ import annotations

import numpy as np
import pytest

from loongforge.embodied.eval.adapters.libero import LiberoAdapter
from loongforge.embodied.eval.factories.registry import (
    MODEL_FACTORY_REGISTRY,
    _auto_import_factory_modules,
)
from loongforge.embodied.eval.orchestrator.config import resolve_action_decoder_key
from loongforge.embodied.eval.orchestrator.runners.libero_runner import (
    _build_rpc_payload,
)
from loongforge.embodied.eval.payload_builders import build_payload_builder
from loongforge.embodied.eval.payload_builders.registry import (
    PAYLOAD_BUILDER_REGISTRY,
    _auto_import_payload_builder_modules,
)


def _synthetic_env_obs() -> dict:
    """Build a minimal LIBERO-shaped obs for adapter unit tests."""
    return {
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.5, -0.3, 0.4], dtype=np.float32),
        "robot0_eef_quat": np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.7], dtype=np.float32),
    }


def _synthetic_context() -> dict:
    """Build the controller-context dict LiberoRunner would supply."""
    return {
        "instruction": "pick up the cup",
        "episode_id": "ep0",
        "episode_step": 0,
        "ee_pos": np.array([0.5, -0.3, 0.4], dtype=np.float32),
        "ee_ori_mat": np.array(
            [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]], dtype=np.float32
        ),
    }


def test_registry_keys_are_consistent():
    """Every Factory must have a matching PayloadBuilder (and vice versa).

    Orchestrator startup will assert this; the unit test lets the misalignment
    surface fast when a new model is added but only half-wired.

    Factories run on the model-server side (require ``torch``); this test
    therefore skips in benchmark client envs (simplerenv / libero / calvin /
    maniskill / robotwin) that intentionally lack torch — the check only
    means anything when both registries can be populated.
    """
    pytest.importorskip("torch", reason="factory registry requires torch (server-side env)")
    _auto_import_factory_modules()
    _auto_import_payload_builder_modules()
    factory_keys = set(MODEL_FACTORY_REGISTRY)
    builder_keys = set(PAYLOAD_BUILDER_REGISTRY)
    assert factory_keys == builder_keys, (
        f"factory - builder = {factory_keys - builder_keys}; "
        f"builder - factory = {builder_keys - factory_keys}"
    )


def test_pi05_payload_builder_defaults_and_state_none():
    """Pi05 emits no state and packs images as ``[primary, wrist]``."""
    adapter = LiberoAdapter("libero_object")
    canonical = adapter.obs_to_canonical(_synthetic_env_obs(), _synthetic_context())
    pb = build_payload_builder("pi05", yaml_model={}, yaml_server={}, yaml_benchmark={})
    kwargs = pb.build(canonical, {"benchmark_name": "libero"})
    assert isinstance(kwargs["images"], list) and len(kwargs["images"]) == 2
    assert kwargs["instructions"] == ["pick up the cup"]
    assert kwargs["state"] is None
    # Pi05 capabilities.
    assert pb.action_encoding == "axis_angle"
    assert pb.action_dim == 7


def test_resolve_action_decoder_key_pi05_identity():
    """Pi05 axis_angle × LIBERO axis_angle should compose to identity (empty key)."""
    adapter = LiberoAdapter("libero_object")
    pb = build_payload_builder("pi05")
    assert resolve_action_decoder_key(pb, adapter) == ""


def test_resolve_action_decoder_key_xvla_ee6d_to_axis_angle():
    """XVLA ee6d × LIBERO axis_angle auto-composes to ``ee6d_to_axis_angle``."""
    adapter = LiberoAdapter("libero_object")
    pb = build_payload_builder("xvla", yaml_model={"state_encoding": "ee6d"})
    assert resolve_action_decoder_key(pb, adapter) == "ee6d_to_axis_angle"


def test_build_rpc_payload_includes_rpc_control_fields():
    """The RPC dict wraps model kwargs with episode / cache control fields."""
    adapter = LiberoAdapter("libero_object")
    canonical = adapter.obs_to_canonical(_synthetic_env_obs(), _synthetic_context())
    pb = build_payload_builder("xvla", yaml_model={"state_encoding": "ee6d", "domain_id": 3})
    ctx = {"benchmark_name": "libero", "episode_id": "ep0", "episode_step": 5}
    rpc = _build_rpc_payload(pb, canonical, ctx, disable_action_cache=True)
    # RPC control.
    assert rpc["episode_id"] == "ep0"
    assert rpc["episode_step"] == 5
    assert rpc["disable_action_cache"] is True
    assert rpc["return_action_chunk"] is False
    # Model kwargs.
    assert rpc["domain_id"] == 3
    assert rpc["state"].shape == (20,)
    assert isinstance(rpc["images"], list)


def test_yaml_override_respects_type_annotation_whitelist():
    """YAML keys without a matching type annotation must not clobber attributes."""
    pb = build_payload_builder(
        "xvla",
        yaml_model={
            "state_encoding": "ee6d",   # annotated → override
            "unnorm_key": "custom_key",  # unannotated → goes to _yaml_model, not attr
            "not_a_real_field": "junk",
        },
    )
    assert pb.state_encoding == "ee6d"
    # unnorm_key is populated via explicit constructor logic, not annotation
    # override — but it should still make it in through the yaml_model dict.
    assert pb.unnorm_key == "custom_key"
    # Unrelated field must not leak onto the instance.
    assert not hasattr(pb, "not_a_real_field")


# ---------------------------------------------------------------------------
# CALVIN + ManiSkill + SimplerEnv coverage (added when the refactor extended
# past LIBERO to the other 3 A-form benchmarks).
# ---------------------------------------------------------------------------


def test_maniskill_pi05_passthrough_reads_joint_from_state_raw():
    """ManiSkill ``passthrough`` forwards adapter-emitted 8D Panda qpos as-is."""
    from loongforge.embodied.eval.adapters.maniskill import ManiSkillAdapter

    adapter = ManiSkillAdapter()
    env_obs = {
        "image": {"base_camera": {"rgb": np.zeros((224, 224, 3), dtype=np.uint8)}},
        "agent": {
            "qpos": np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.04, 0.05], dtype=np.float32
            ),
            "qvel": np.zeros(9, dtype=np.float32),
        },
    }
    canonical = adapter.obs_to_canonical(
        env_obs, {"instruction": "pick up", "episode_id": "ep", "episode_step": 0}
    )
    pb = build_payload_builder("pi05", yaml_model={"state_encoding": "passthrough"})
    kwargs = pb.build(canonical, {"benchmark_name": "maniskill"})
    assert kwargs["state"].shape == (8,)
    assert np.allclose(kwargs["state"][:7], np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
    # Gripper column is the mean of the two finger joints.
    assert np.isclose(float(kwargs["state"][7]), 0.045, atol=1e-6)


def test_action_decoder_key_calvin_and_maniskill():
    """M×N capability matching produces the expected registry keys for the new benchmarks."""
    from loongforge.embodied.eval.adapters.calvin import CalvinAdapter
    from loongforge.embodied.eval.adapters.maniskill import ManiSkillAdapter
    from loongforge.embodied.eval.adapters.simplerenv import SimplerEnvAdapter

    pb_xvla = build_payload_builder("xvla", yaml_model={"state_encoding": "ee6d"})
    # xvla × CALVIN
    assert resolve_action_decoder_key(pb_xvla, CalvinAdapter()) == "ee6d_to_calvin_abs"
    # xvla × SimplerEnv
    assert resolve_action_decoder_key(pb_xvla, SimplerEnvAdapter()) == "ee6d_to_simpler_abs_euler"
    # xvla × ManiSkill
    assert resolve_action_decoder_key(pb_xvla, ManiSkillAdapter()) == "ee6d_to_axis_angle"
    # pi05 × anything → identity (pi05.action_encoding == adapter.action_space)
    pb_pi = build_payload_builder("pi05")
    assert resolve_action_decoder_key(pb_pi, ManiSkillAdapter()) == ""


# ---------------------------------------------------------------------------
# RoboTwin (form-B bridge) coverage — proprio + stateful action decoders.
# ---------------------------------------------------------------------------


def _robotwin_endpose():
    """Synthetic dual-arm endpose (identity quats, half-open grippers)."""
    return {
        "left_endpose": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "right_endpose": np.array([-0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "left_gripper": 0.25,
        "right_gripper": 0.75,
    }


def test_ee6d_dual_decoder_matches_helper():
    """The ee6d_dual ActionDecoder equals the per-row ``_robotwin_ee6d_row_to_ee`` helper."""
    from loongforge.embodied.eval.action_decoders import build_action_decoder
    from loongforge.embodied.eval.action_decoders.ee6d import _robotwin_ee6d_row_to_ee

    raw = np.random.RandomState(0).randn(20).astype(np.float32)
    decoder = build_action_decoder("ee6d_robotwin_ee_dual")
    # Chunk-based interface: [H, D] in -> [H, D'] out.
    out = decoder(raw.reshape(1, -1), {})
    assert out.shape == (1, 16)
    assert np.allclose(out[0], _robotwin_ee6d_row_to_ee(raw), atol=1e-6)


def test_pi05_aloha_proprio_and_decoder_chunk_cache():
    """pi05 aloha_pi proprio = adapt_to_pi(joint); decoder anchors abs to chunk state."""
    from loongforge.embodied.eval.action_decoders import build_action_decoder
    from loongforge.embodied.eval.action_decoders.joint import (
        adapt_to_pi_encode_actions,
        delta_to_absolute_actions,
    )
    from loongforge.embodied.eval.payload_builders.pi05 import adapt_to_pi_decode_state

    joint0 = np.linspace(-0.5, 0.5, 14).astype(np.float32)
    canonical = {
        "instruction": "adjust bottle",
        "images": {"primary": np.zeros((10, 10, 3), np.uint8), "left": np.zeros((10, 10, 3), np.uint8),
                   "right": np.zeros((10, 10, 3), np.uint8), "head": np.zeros((10, 10, 3), np.uint8), "wrist": None},
        "state_raw": {"joint": joint0, "endpose": None},
    }
    pb = build_payload_builder("pi05", yaml_model={"state_encoding": "aloha_pi"})
    kwargs = pb.build(canonical, {"benchmark_name": "robotwin"})
    assert np.allclose(kwargs["state"], adapt_to_pi_decode_state(joint0), atol=1e-6)

    decoder = build_action_decoder("pi05_aloha_robotwin")
    decoder.reset()
    raw = np.full(14, 0.01, dtype=np.float32)
    # Fresh chunk: anchor = adapt_to_pi(joint0). Chunk-based [1, 14] -> [1, 14].
    pi_state0 = adapt_to_pi_decode_state(joint0)
    env0 = decoder(raw.reshape(1, -1), {"pi_state": pi_state0, "is_fresh_chunk": True})[0]
    expected0 = adapt_to_pi_encode_actions(
        delta_to_absolute_actions(raw, adapt_to_pi_decode_state(joint0))
    )
    assert np.allclose(env0, expected0, atol=1e-6)
    # Cache hit with a *different* pi_state: anchor must stay at the fresh-chunk state.
    joint1 = joint0 + 0.2
    pi_state1 = adapt_to_pi_decode_state(joint1)
    env1 = decoder(raw.reshape(1, -1), {"pi_state": pi_state1, "is_fresh_chunk": False})[0]
    expected1 = adapt_to_pi_encode_actions(
        delta_to_absolute_actions(raw, adapt_to_pi_decode_state(joint0))
    )
    assert np.allclose(env1, expected1, atol=1e-6)
