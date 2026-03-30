"""Model registry loader."""

import yaml

from ...common.paths import get_repo_root


def load_model_registry() -> list[dict]:
    """Load model registry from YAML file."""
    registry_path = get_repo_root() / "configs" / "model_registry.yaml"
    with open(registry_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []
