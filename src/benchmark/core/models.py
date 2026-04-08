"""ModelEntry dataclass and model loading."""

from dataclasses import dataclass, field
from pathlib import Path

from ...common.paths import resolve_repo_relative_path
from .registry import load_model_registry


@dataclass
class ModelEntry:
    family: str
    variant: str
    hyperparameters: str
    dataset: str  # "COCO-Person" or "COCO-80"
    num_classes: int
    fmt: str  # "Int8" or "W4A8"
    resolution: int
    model_path: str  # absolute path to model file
    config_path: str  # absolute path to base config YAML
    overrides: dict = field(default_factory=dict)
    framework: str = "tf"  # "tf" or "torch"
    input_data_type: str = "uint8"
    output_data_type: str = "int8"


def _num_classes_for_dataset(dataset: str) -> int:
    if dataset == "COCO-80":
        return 80
    return 1


def load_models() -> list[ModelEntry]:
    """Build ModelEntry list from the model registry."""
    entries: list[ModelEntry] = []
    for reg in load_model_registry():
        model_path = resolve_repo_relative_path(reg["model"])
        config_path = resolve_repo_relative_path(reg["config"])

        if not config_path.exists():
            print(f"[WARN] Config not found, skipping: {config_path}")
            continue
        if not model_path.exists():
            print(f"[WARN] Model not found, skipping: {model_path}")
            continue

        entries.append(
            ModelEntry(
                family=reg["family"],
                variant=reg["variant"],
                hyperparameters=reg["hyperparameters"],
                dataset=reg["dataset"],
                num_classes=_num_classes_for_dataset(reg["dataset"]),
                fmt=reg["fmt"],
                resolution=reg["resolution"],
                model_path=str(model_path.resolve()),
                config_path=str(config_path.resolve()),
                overrides=reg.get("overrides", {}),
                framework=reg.get("framework", "tf"),
                input_data_type=reg.get("input_data_type", "uint8"),
                output_data_type=reg.get("output_data_type", "int8"),
            )
        )
    return entries
