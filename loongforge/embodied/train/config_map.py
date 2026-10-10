# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""model-name → (YAML path, ModelConfig class, DataConfig class) routing.

MODEL_SCHEMA is the single place that binds a model name to:
  - its shared YAML file (with ``model:`` / ``data:`` sections), and
  - the two typed config classes that the YAML sections are merged into.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Type

from loongforge.embodied.model.pi05.model_configuration_pi05 import Pi05ModelConfig
from loongforge.embodied.data.datasets.pi05.transforms.data_configuration_pi05 import Pi05DataConfig

_CONFIGS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "configs" / "models" / "embodied"
)


@dataclass(frozen=True)
class ModelSchema:
    """Binding of a model name to its YAML file and typed config classes."""

    yaml_file: str
    model_config_cls: Type
    data_config_cls: Type


MODEL_SCHEMA = {
    "pi05": ModelSchema("pi05.yaml", Pi05ModelConfig, Pi05DataConfig),
}

def get_model_schema(model_name: str) -> ModelSchema:
    """Look up the ModelSchema for a model name."""
    key = model_name.lower().replace("-", "_")
    if key not in MODEL_SCHEMA:
        available = ", ".join(sorted(MODEL_SCHEMA.keys()))
        raise ValueError(f"Unknown model-name '{model_name}'. Available: [{available}]")
    return MODEL_SCHEMA[key]


def get_config_path(model_name: str) -> str:
    """Look up config YAML path by model name."""
    schema = get_model_schema(model_name)
    path = _CONFIGS_DIR / schema.yaml_file
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return str(path)
