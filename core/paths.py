"""Where the repository keeps its configuration. Stdlib only.

Every module resolves its config through here rather than off its own
`__file__`, so moving a module between packages can never silently repoint it at
a different YAML.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = ROOT / "config"

# The model registry.
MODELS_PATH = CONFIG_DIR / "models.yaml"

# The welfare module's design.
WELFARE_CONFIG_PATH = CONFIG_DIR / "welfare.yaml"

# The item bank. The welfare attributes are a positively-framed restatement of
# these items and are validated for exact coverage against them.
SCALES_DIR = CONFIG_DIR / "scales"
BATTERY_PATH = SCALES_DIR / "llm_self_scales_adapted.json"
VARIANTS_PATH = SCALES_DIR / "item_variants.json"
WELFARE_ATTRIBUTES_PATH = SCALES_DIR / "welfare_attributes.json"
