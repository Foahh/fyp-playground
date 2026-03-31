"""Benchmark-specific path constants and resolved paths per mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..common.paths import get_datasets_dir, get_repo_root, get_stedgeai_path

_VALID_BENCHMARK_MODES = frozenset({"underdrive", "nominal", "overdrive"})


def _benchmark_results_subdir(mode: str) -> str:
    if mode == "overdrive":
        return "benchmark_overdrive"
    if mode == "nominal":
        return "benchmark_nominal"
    if mode == "underdrive":
        return "benchmark_underdrive"
    raise ValueError(f"Invalid benchmark mode {mode!r}; expected one of {sorted(_VALID_BENCHMARK_MODES)}")


BASE_DIR = get_repo_root()
DATASETS_DIR = get_datasets_dir()
MODELZOO_DIR = BASE_DIR / "external" / "stm32ai-modelzoo" / "object_detection"
SERVICES_DIR = BASE_DIR / "external" / "stm32ai-modelzoo-services" / "object_detection"
RESULTS_DIR = BASE_DIR / "results"

BENCHMARK_PARSED_DIR = RESULTS_DIR

STEDGEAI_PATH = get_stedgeai_path()

N6_WORKDIR = RESULTS_DIR / "n6_workdir"

COCO_PERSON_TFS_TEST = str(DATASETS_DIR / "coco_2017_person" / "test")
COCO_80_TFS_TEST = str(DATASETS_DIR / "coco_2017_80_classes" / "test")
COCO_PERSON_ANNOTATIONS = str(
    DATASETS_DIR / "coco" / "annotations" / "instances_val2017_person.json"
)
COCO_80_ANNOTATIONS = str(
    DATASETS_DIR / "coco" / "annotations" / "instances_val2017.json"
)
COCO_IMAGES = str(DATASETS_DIR / "coco" / "images" / "val2017")

METRIC_PARSED_CSV_PATH = BENCHMARK_PARSED_DIR / "benchmark_parsed.csv"


@dataclass(frozen=True)
class BenchmarkPaths:
    """Resolved paths for one benchmark mode (underdrive / nominal / overdrive)."""

    benchmark_dir: Path
    csv_path: Path
    benchmark_log: Path
    power_measure_csv_path: Path


def benchmark_paths_for_mode(mode: str) -> BenchmarkPaths:
    """Return paths for a single benchmark mode."""
    m = mode.strip().lower()
    if m not in _VALID_BENCHMARK_MODES:
        raise ValueError(
            f"Invalid benchmark mode {mode!r}; expected one of {sorted(_VALID_BENCHMARK_MODES)}"
        )
    bd = RESULTS_DIR / _benchmark_results_subdir(m)
    return BenchmarkPaths(
        benchmark_dir=bd,
        csv_path=bd / "benchmark_results.csv",
        benchmark_log=bd / "benchmark.log",
        power_measure_csv_path=bd / "power_measure.csv",
    )
