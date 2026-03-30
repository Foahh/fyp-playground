"""Benchmark-specific path constants."""

import sys
from pathlib import Path

from ..common.paths import get_datasets_dir, get_repo_root, get_stedgeai_path


def _detect_benchmark_mode_from_argv() -> str:
    """Detect benchmark mode from CLI args."""
    mode = "nominal"
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token == "--mode" and i + 1 < len(argv):
            mode = argv[i + 1].strip().lower()
            break
        if token.startswith("--mode="):
            mode = token.split("=", 1)[1].strip().lower()
            break

    if mode in ("override", "overdrive"):
        return "overdrive"
    if mode in ("norminal", "nominal"):
        return "nominal"
    return "nominal"


BASE_DIR = get_repo_root()
DATASETS_DIR = get_datasets_dir()
MODELZOO_DIR = BASE_DIR / "external" / "stm32ai-modelzoo" / "object_detection"
SERVICES_DIR = BASE_DIR / "external" / "stm32ai-modelzoo-services" / "object_detection"
RESULTS_DIR = BASE_DIR / "results"

BENCHMARK_MODE = _detect_benchmark_mode_from_argv()
BENCHMARK_DIR = RESULTS_DIR / (
    "benchmark_overdrive" if BENCHMARK_MODE == "overdrive" else "benchmark_nominal"
)
BENCHMARK_PARSED_DIR = RESULTS_DIR

CSV_PATH = BENCHMARK_DIR / "benchmark_results.csv"
BENCHMARK_LOG = BENCHMARK_DIR / "benchmark.log"
POWER_MEASURE_CSV_PATH = BENCHMARK_DIR / "power_measure.csv"

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
