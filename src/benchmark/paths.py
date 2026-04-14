"""Benchmark-specific path constants and resolved paths per mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..common.paths import (
    get_datasets_dir,
    get_repo_root,
    get_results_dir,
    get_stedgeai_path,
)

_VALID_BENCHMARK_MODES = frozenset({"underdrive", "nominal", "overdrive"})


def _normalize_benchmark_mode(mode: str) -> str:
    m = mode.strip().lower()
    if m not in _VALID_BENCHMARK_MODES:
        raise ValueError(
            f"Invalid benchmark mode {mode!r}; expected one of {sorted(_VALID_BENCHMARK_MODES)}"
        )
    return m


BASE_DIR = get_repo_root()
DATASETS_DIR = get_datasets_dir()
MODELZOO_DIR = BASE_DIR / "external" / "stm32ai-modelzoo" / "object_detection"
SERVICES_DIR = BASE_DIR / "external" / "stm32ai-modelzoo-services" / "object_detection"
RESULTS_DIR = get_results_dir()


def benchmark_results_csv_path(mode: str) -> Path:
    """``results/benchmark_{mode}_results.csv`` (underdrive / nominal / overdrive)."""
    m = _normalize_benchmark_mode(mode)
    return RESULTS_DIR / f"benchmark_{m}_results.csv"


def power_measure_csv_path(mode: str) -> Path:
    """``results/power_measure_{mode}.csv``."""
    m = _normalize_benchmark_mode(mode)
    return RESULTS_DIR / f"power_measure_{m}.csv"


def benchmark_log_path(mode: str) -> Path:
    """``results/benchmark_{mode}.log`` — STEdgeAI / benchmark audit log for one clock mode."""
    m = _normalize_benchmark_mode(mode)
    return RESULTS_DIR / f"benchmark_{m}.log"

BENCHMARK_PARSED_DIR = RESULTS_DIR
GENERATED_NETWORK_DIR = RESULTS_DIR / "network"
GENERATE_RESULT_CSV_PATH = RESULTS_DIR / "generate_result.csv"
GENERATE_LOG_PATH = RESULTS_DIR / "generate.log"

STEDGEAI_PATH = get_stedgeai_path()

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

    csv_path: Path
    benchmark_log: Path
    power_measure_csv_path: Path


def benchmark_paths_for_mode(mode: str) -> BenchmarkPaths:
    """Return paths for a single benchmark mode."""
    m = _normalize_benchmark_mode(mode)
    return BenchmarkPaths(
        csv_path=benchmark_results_csv_path(m),
        benchmark_log=benchmark_log_path(m),
        power_measure_csv_path=power_measure_csv_path(m),
    )
