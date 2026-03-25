"""Paths, class lists, CSV columns, and explicit model registry."""

import os
import platform
from pathlib import Path
from typing import Optional, Tuple


def get_stedgeai_path() -> str:
    base = os.environ.get("STEDGEAI_CORE_DIR", "")
    system = platform.system()
    if system == "Windows":
        return os.path.join(base, "Utilities", "windows", "stedgeai.exe")
    elif system == "Linux":
        return os.path.join(base, "Utilities", "linux", "stedgeai")
    elif system == "Darwin":
        return os.path.join(base, "Utilities", "mac", "stedgeai")
    else:
        raise OSError(f"Unsupported platform: {system}")


# ── Paths ──

BASE_DIR = Path(__file__).resolve().parents[2]
MODELZOO_DIR = BASE_DIR / "external" / "stm32ai-modelzoo" / "object_detection"
SERVICES_DIR = BASE_DIR / "external" / "stm32ai-modelzoo-services" / "object_detection"
RESULTS_DIR = BASE_DIR / "results"
BENCHMARK_DIR = RESULTS_DIR / "benchmark"

CSV_PATH = BENCHMARK_DIR / "benchmark_results.csv"
ERROR_LOG = BENCHMARK_DIR / "benchmark_errors.log"
STDOUT_LOG = BENCHMARK_DIR / "benchmark_stdout.log"
POWER_MEASURE_CSV_PATH = BENCHMARK_DIR / "power-measure.csv"

STEDGEAI_PATH = get_stedgeai_path()

N6_WORKDIR = BENCHMARK_DIR / "n6_workdir"

# ── Dataset paths ──

COCO_PERSON_TFS_TEST = str(BASE_DIR / "datasets" / "coco_2017_person" / "test")
COCO_80_TFS_TEST = str(
    BASE_DIR / "datasets" / "coco_2017_80_classes" / "test"
)
COCO_PERSON_ANNOTATIONS = str(
    BASE_DIR / "datasets" / "coco" / "annotations" / "instances_val2017_person.json"
)
COCO_80_ANNOTATIONS = str(
    BASE_DIR / "datasets" / "coco" / "annotations" / "instances_val2017.json"
)
COCO_IMAGES = str(BASE_DIR / "datasets" / "coco" / "images" / "val2017")

# ── COCO 80 class names ──

COCO_80_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# SSD families require output_chpos = chfirst; all others use chlast.
SSD_FAMILIES = {
    "ssdlite_mobilenetv1_pt",
    "ssdlite_mobilenetv2_pt",
    "ssdlite_mobilenetv3large_pt",
    "ssdlite_mobilenetv3small_pt",
}

CSV_COLUMNS = [
    "model_family",
    "model_variant",
    "hyperparameters",
    "dataset",
    "format",
    "resolution",
    "internal_ram_kib",
    "external_ram_kib",
    "weights_flash_kib",
    "inference_time_ms",
    "inf_per_sec",
    "ap_50",
    "avg_power_mW",
]

CSV_COLUMNS_NO_POWER = [c for c in CSV_COLUMNS if c != "avg_power_mW"]

METRIC_PARSED_CSV_PATH = BENCHMARK_DIR / "metric_parsed.csv"


def get_power_serial_config() -> Tuple[Optional[str], int]:
    """Serial device for INA228 Arduino CSV (separate from ST-LINK UART used by stedgeai)."""
    port = os.environ.get("BENCHMARK_POWER_SERIAL", "").strip()
    if not port:
        return None, 115200
    try:
        baud = int(os.environ.get("BENCHMARK_POWER_BAUD", "115200"))
    except ValueError:
        baud = 115200
    return port, baud


_DEFAULT_POWER_DISCARD_MS = 1.0


def get_power_edge_discard_ms() -> Tuple[float, float]:
    """
    Milliseconds to drop from each contiguous inference-high segment (start and end)
    when computing avg power. Uses timestamps from the INA228 CSV (ts_us).

    Env:
      BENCHMARK_POWER_DISCARD_START_MS / BENCHMARK_POWER_DISCARD_END_MS — per-edge
        (default 1 ms each when unset).
      BENCHMARK_POWER_DISCARD_EDGE_MS — sets both start and end to the same value when the
        two specific vars are not set (convenience for symmetric discard).
      Set START_MS=0 and END_MS=0 explicitly to disable edge trimming.
    """
    if (
        "BENCHMARK_POWER_DISCARD_EDGE_MS" in os.environ
        and "BENCHMARK_POWER_DISCARD_START_MS" not in os.environ
        and "BENCHMARK_POWER_DISCARD_END_MS" not in os.environ
    ):
        try:
            v = float(os.environ["BENCHMARK_POWER_DISCARD_EDGE_MS"].strip())
            v = max(0.0, v)
            return v, v
        except ValueError:
            pass

    def _one(key: str) -> float:
        raw = os.environ.get(key, "").strip()
        if not raw:
            return _DEFAULT_POWER_DISCARD_MS
        try:
            return max(0.0, float(raw))
        except ValueError:
            return _DEFAULT_POWER_DISCARD_MS

    return _one("BENCHMARK_POWER_DISCARD_START_MS"), _one("BENCHMARK_POWER_DISCARD_END_MS")

# ── Path helpers for the model registry ──

_ZOO = Path("external") / "stm32ai-modelzoo" / "object_detection"
_CONFIGS = Path("configs")
_UL = (
    Path("external")
    / "ultralytics"
    / "examples"
    / "YOLOv8-STEdgeAI"
    / "stedgeai_models"
    / "object_detection"
)

_MD_ssdlite_mobilenetv1_pt = _ZOO / "ssdlite_mobilenetv1_pt/README.md"
_MD_ssdlite_mobilenetv2_pt = _ZOO / "ssdlite_mobilenetv2_pt/README.md"
_MD_ssdlite_mobilenetv3large_pt = _ZOO / "ssdlite_mobilenetv3large_pt/README.md"
_MD_ssdlite_mobilenetv3small_pt = _ZOO / "ssdlite_mobilenetv3small_pt/README.md"
_MD_st_yolodv2milli_pt = _ZOO / "st_yolodv2milli_pt/README.md"
_MD_st_yolodv2tiny_pt = _ZOO / "st_yolodv2tiny_pt/README.md"
_MD_st_yololcv1 = _ZOO / "st_yololcv1/README.md"
_MD_st_yoloxn = _ZOO / "st_yoloxn/README.md"
_MD_yolov8n = _ZOO / "yolov8n/README.md"
_MD_yolov11n = _ZOO / "yolov11n/README.md"

# ── Model Registry ──
# Each entry explicitly maps a model file to its base config file.
# Paths are relative to BASE_DIR and resolved at load time.

MODEL_REGISTRY: list[dict] = [
    # ── ssdlite_mobilenetv1_pt ──
    {
        "config": _ZOO / "ssdlite_mobilenetv1_pt/Public_pretrainedmodel_public_dataset/coco/ssdlite_mobilenetv1_pt_coco_300/ssdlite_mobilenetv1_pt_coco_300_config.yaml",
        "model": _ZOO / "ssdlite_mobilenetv1_pt/Public_pretrainedmodel_public_dataset/coco/ssdlite_mobilenetv1_pt_coco_300/ssdlite_mobilenetv1_pt_coco_300_qdq_int8.onnx",
        "family": "ssdlite_mobilenetv1_pt",
        "readme": _MD_ssdlite_mobilenetv1_pt,
        "variant": "ssdlite_mobilenetv1_pt_coco_300",
        "hyperparameters": "",
        "dataset": "COCO-80",
        "fmt": "Int8",
        "resolution": 300,
        "overrides": {"evaluation": {"output_chpos": "chfirst"}},
    },
    {
        "config": _ZOO / "ssdlite_mobilenetv1_pt/ST_pretrainedmodel_public_dataset/coco_person/ssdlite_mobilenetv1_pt_coco_person_300/ssdlite_mobilenetv1_pt_coco_person_300_config.yaml",
        "model": _ZOO / "ssdlite_mobilenetv1_pt/ST_pretrainedmodel_public_dataset/coco_person/ssdlite_mobilenetv1_pt_coco_person_300/ssdlite_mobilenetv1_pt_coco_person_300_qdq_int8.onnx",
        "family": "ssdlite_mobilenetv1_pt",
        "readme": _MD_ssdlite_mobilenetv1_pt,
        "variant": "ssdlite_mobilenetv1_pt_coco_person_300",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 300,
        "overrides": {"evaluation": {"output_chpos": "chfirst"}},
    },
    # ── ssdlite_mobilenetv2_pt ──
    {
        "config": _ZOO / "ssdlite_mobilenetv2_pt/Public_pretrainedmodel_public_dataset/coco/ssdlite_mobilenetv2_pt_coco_300/ssdlite_mobilenetv2_pt_coco_300_config.yaml",
        "model": _ZOO / "ssdlite_mobilenetv2_pt/Public_pretrainedmodel_public_dataset/coco/ssdlite_mobilenetv2_pt_coco_300/ssdlite_mobilenetv2_pt_coco_300_qdq_int8.onnx",
        "family": "ssdlite_mobilenetv2_pt",
        "readme": _MD_ssdlite_mobilenetv2_pt,
        "variant": "ssdlite_mobilenetv2_pt_coco_300",
        "hyperparameters": "",
        "dataset": "COCO-80",
        "fmt": "Int8",
        "resolution": 300,
        "overrides": {"evaluation": {"output_chpos": "chfirst"}},
    },
    {
        "config": _ZOO / "ssdlite_mobilenetv2_pt/ST_pretrainedmodel_public_dataset/coco_person/ssdlite_mobilenetv2_pt_coco_person_300/ssdlite_mobilenetv2_pt_coco_person_300_config.yaml",
        "model": _ZOO / "ssdlite_mobilenetv2_pt/ST_pretrainedmodel_public_dataset/coco_person/ssdlite_mobilenetv2_pt_coco_person_300/ssdlite_mobilenetv2_pt_coco_person_300_qdq_int8.onnx",
        "family": "ssdlite_mobilenetv2_pt",
        "readme": _MD_ssdlite_mobilenetv2_pt,
        "variant": "ssdlite_mobilenetv2_pt_coco_person_300",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 300,
        "overrides": {"evaluation": {"output_chpos": "chfirst"}},
    },
    # ── ssdlite_mobilenetv3large_pt ──
    {
        "config": _ZOO / "ssdlite_mobilenetv3large_pt/Public_pretrainedmodel_public_dataset/coco/ssdlite_mobilenetv3large_pt_coco_300/ssdlite_mobilenetv3large_pt_coco_300_config.yaml",
        "model": _ZOO / "ssdlite_mobilenetv3large_pt/Public_pretrainedmodel_public_dataset/coco/ssdlite_mobilenetv3large_pt_coco_300/ssdlite_mobilenetv3large_pt_coco_300_qdq_int8.onnx",
        "family": "ssdlite_mobilenetv3large_pt",
        "readme": _MD_ssdlite_mobilenetv3large_pt,
        "variant": "ssdlite_mobilenetv3large_pt_coco_300",
        "hyperparameters": "",
        "dataset": "COCO-80",
        "fmt": "Int8",
        "resolution": 300,
        "overrides": {"evaluation": {"output_chpos": "chfirst"}},
    },
    {
        "config": _ZOO / "ssdlite_mobilenetv3large_pt/ST_pretrainedmodel_public_dataset/coco_person/ssdlite_mobilenetv3large_pt_coco_person_300/ssdlite_mobilenetv3large_pt_coco_person_300_config.yaml",
        "model": _ZOO / "ssdlite_mobilenetv3large_pt/ST_pretrainedmodel_public_dataset/coco_person/ssdlite_mobilenetv3large_pt_coco_person_300/ssdlite_mobilenetv3large_pt_coco_person_300_qdq_int8.onnx",
        "family": "ssdlite_mobilenetv3large_pt",
        "readme": _MD_ssdlite_mobilenetv3large_pt,
        "variant": "ssdlite_mobilenetv3large_pt_coco_person_300",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 300,
        "overrides": {"evaluation": {"output_chpos": "chfirst"}},
    },
    # ── ssdlite_mobilenetv3small_pt ──
    {
        "config": _ZOO / "ssdlite_mobilenetv3small_pt/Public_pretrainedmodel_public_dataset/coco/ssdlite_mobilenetv3small_pt_coco_300/ssdlite_mobilenetv3small_pt_coco_300_config.yaml",
        "model": _ZOO / "ssdlite_mobilenetv3small_pt/Public_pretrainedmodel_public_dataset/coco/ssdlite_mobilenetv3small_pt_coco_300/ssdlite_mobilenetv3small_pt_coco_300_qdq_int8.onnx",
        "family": "ssdlite_mobilenetv3small_pt",
        "readme": _MD_ssdlite_mobilenetv3small_pt,
        "variant": "ssdlite_mobilenetv3small_pt_coco_300",
        "hyperparameters": "",
        "dataset": "COCO-80",
        "fmt": "Int8",
        "resolution": 300,
        "overrides": {"evaluation": {"output_chpos": "chfirst"}},
    },
    {
        "config": _ZOO / "ssdlite_mobilenetv3small_pt/ST_pretrainedmodel_public_dataset/coco_person/ssdlite_mobilenetv3small_pt_coco_person_300/ssdlite_mobilenetv3small_pt_coco_person_300_config.yaml",
        "model": _ZOO / "ssdlite_mobilenetv3small_pt/ST_pretrainedmodel_public_dataset/coco_person/ssdlite_mobilenetv3small_pt_coco_person_300/ssdlite_mobilenetv3small_pt_coco_person_300_qdq_int8.onnx",
        "family": "ssdlite_mobilenetv3small_pt",
        "readme": _MD_ssdlite_mobilenetv3small_pt,
        "variant": "ssdlite_mobilenetv3small_pt_coco_person_300",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 300,
        "overrides": {"evaluation": {"output_chpos": "chfirst"}},
    },
    # ── st_yolodv2milli_pt (COCO-80) ──
    {
        "config": _ZOO / "st_yolodv2milli_pt/ST_pretrainedmodel_public_dataset/coco/st_yolodv2milli_actrelu_pt_coco_192/st_yolodv2milli_actrelu_pt_coco_192_config.yaml",
        "model": _ZOO / "st_yolodv2milli_pt/ST_pretrainedmodel_public_dataset/coco/st_yolodv2milli_actrelu_pt_coco_192/st_yolodv2milli_actrelu_pt_coco_192_qdq_int8.onnx",
        "family": "st_yolodv2milli_pt",
        "readme": _MD_st_yolodv2milli_pt,
        "variant": "st_yolodv2milli_actrelu_pt_coco_192",
        "hyperparameters": "actrelu",
        "dataset": "COCO-80",
        "fmt": "Int8",
        "resolution": 192,
    },
    {
        "config": _ZOO / "st_yolodv2milli_pt/ST_pretrainedmodel_public_dataset/coco/st_yolodv2milli_actrelu_pt_coco_320/st_yolodv2milli_actrelu_pt_coco_320_config.yaml",
        "model": _ZOO / "st_yolodv2milli_pt/ST_pretrainedmodel_public_dataset/coco/st_yolodv2milli_actrelu_pt_coco_320/st_yolodv2milli_actrelu_pt_coco_320_qdq_int8.onnx",
        "family": "st_yolodv2milli_pt",
        "readme": _MD_st_yolodv2milli_pt,
        "variant": "st_yolodv2milli_actrelu_pt_coco_320",
        "hyperparameters": "actrelu",
        "dataset": "COCO-80",
        "fmt": "Int8",
        "resolution": 320,
    },
    {
        "config": _ZOO / "st_yolodv2milli_pt/ST_pretrainedmodel_public_dataset/coco/st_yolodv2milli_actrelu_pt_coco_640/st_yolodv2milli_actrelu_pt_coco_640_config.yaml",
        "model": _ZOO / "st_yolodv2milli_pt/ST_pretrainedmodel_public_dataset/coco/st_yolodv2milli_actrelu_pt_coco_640/st_yolodv2milli_actrelu_pt_coco_640_qdq_int8.onnx",
        "family": "st_yolodv2milli_pt",
        "readme": _MD_st_yolodv2milli_pt,
        "variant": "st_yolodv2milli_actrelu_pt_coco_640",
        "hyperparameters": "actrelu",
        "dataset": "COCO-80",
        "fmt": "Int8",
        "resolution": 640,
    },
    # ── st_yolodv2milli_pt (COCO-Person) ──
    {
        "config": _ZOO / "st_yolodv2milli_pt/ST_pretrainedmodel_public_dataset/coco_person/st_yolodv2milli_actrelu_pt_coco_person_192/st_yolodv2milli_actrelu_pt_coco_person_192_config.yaml",
        "model": _ZOO / "st_yolodv2milli_pt/ST_pretrainedmodel_public_dataset/coco_person/st_yolodv2milli_actrelu_pt_coco_person_192/st_yolodv2milli_actrelu_pt_coco_person_192_qdq_int8.onnx",
        "family": "st_yolodv2milli_pt",
        "readme": _MD_st_yolodv2milli_pt,
        "variant": "st_yolodv2milli_actrelu_pt_coco_person_192",
        "hyperparameters": "actrelu",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 192,
    },
    {
        "config": _ZOO / "st_yolodv2milli_pt/ST_pretrainedmodel_public_dataset/coco_person/st_yolodv2milli_actrelu_pt_coco_person_256/st_yolodv2milli_actrelu_pt_coco_person_256_config.yaml",
        "model": _ZOO / "st_yolodv2milli_pt/ST_pretrainedmodel_public_dataset/coco_person/st_yolodv2milli_actrelu_pt_coco_person_256/st_yolodv2milli_actrelu_pt_coco_person_256_qdq_int8.onnx",
        "family": "st_yolodv2milli_pt",
        "readme": _MD_st_yolodv2milli_pt,
        "variant": "st_yolodv2milli_actrelu_pt_coco_person_256",
        "hyperparameters": "actrelu",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 256,
    },
    {
        "config": _ZOO / "st_yolodv2milli_pt/ST_pretrainedmodel_public_dataset/coco_person/st_yolodv2milli_actrelu_pt_coco_person_320/st_yolodv2milli_actrelu_pt_coco_person_320_config.yaml",
        "model": _ZOO / "st_yolodv2milli_pt/ST_pretrainedmodel_public_dataset/coco_person/st_yolodv2milli_actrelu_pt_coco_person_320/st_yolodv2milli_actrelu_pt_coco_person_320_qdq_int8.onnx",
        "family": "st_yolodv2milli_pt",
        "readme": _MD_st_yolodv2milli_pt,
        "variant": "st_yolodv2milli_actrelu_pt_coco_person_320",
        "hyperparameters": "actrelu",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 320,
    },
    # ── st_yolodv2tiny_pt (COCO-80) ──
    {
        "config": _ZOO / "st_yolodv2tiny_pt/ST_pretrainedmodel_public_dataset/coco/st_yolodv2tiny_actrelu_pt_coco_192/st_yolodv2tiny_actrelu_pt_coco_192_config.yaml",
        "model": _ZOO / "st_yolodv2tiny_pt/ST_pretrainedmodel_public_dataset/coco/st_yolodv2tiny_actrelu_pt_coco_192/st_yolodv2tiny_actrelu_pt_coco_192_qdq_int8.onnx",
        "family": "st_yolodv2tiny_pt",
        "readme": _MD_st_yolodv2tiny_pt,
        "variant": "st_yolodv2tiny_actrelu_pt_coco_192",
        "hyperparameters": "actrelu",
        "dataset": "COCO-80",
        "fmt": "Int8",
        "resolution": 192,
    },
    {
        "config": _ZOO / "st_yolodv2tiny_pt/ST_pretrainedmodel_public_dataset/coco/st_yolodv2tiny_actrelu_pt_coco_288/st_yolodv2tiny_actrelu_pt_coco_288_config.yaml",
        "model": _ZOO / "st_yolodv2tiny_pt/ST_pretrainedmodel_public_dataset/coco/st_yolodv2tiny_actrelu_pt_coco_288/st_yolodv2tiny_actrelu_pt_coco_288_qdq_int8.onnx",
        "family": "st_yolodv2tiny_pt",
        "readme": _MD_st_yolodv2tiny_pt,
        "variant": "st_yolodv2tiny_actrelu_pt_coco_288",
        "hyperparameters": "actrelu",
        "dataset": "COCO-80",
        "fmt": "Int8",
        "resolution": 288,
    },
    {
        "config": _ZOO / "st_yolodv2tiny_pt/ST_pretrainedmodel_public_dataset/coco/st_yolodv2tiny_actrelu_pt_coco_640/st_yolodv2tiny_actrelu_pt_coco_640_config.yaml",
        "model": _ZOO / "st_yolodv2tiny_pt/ST_pretrainedmodel_public_dataset/coco/st_yolodv2tiny_actrelu_pt_coco_640/st_yolodv2tiny_actrelu_pt_coco_640_qdq_int8.onnx",
        "family": "st_yolodv2tiny_pt",
        "readme": _MD_st_yolodv2tiny_pt,
        "variant": "st_yolodv2tiny_actrelu_pt_coco_640",
        "hyperparameters": "actrelu",
        "dataset": "COCO-80",
        "fmt": "Int8",
        "resolution": 640,
    },
    # ── st_yolodv2tiny_pt (COCO-Person) ──
    {
        "config": _ZOO / "st_yolodv2tiny_pt/ST_pretrainedmodel_public_dataset/coco_person/st_yolodv2tiny_actrelu_pt_coco_person_192/st_yolodv2tiny_actrelu_pt_coco_person_192_config.yaml",
        "model": _ZOO / "st_yolodv2tiny_pt/ST_pretrainedmodel_public_dataset/coco_person/st_yolodv2tiny_actrelu_pt_coco_person_192/st_yolodv2tiny_actrelu_pt_coco_person_192_qdq_int8.onnx",
        "family": "st_yolodv2tiny_pt",
        "readme": _MD_st_yolodv2tiny_pt,
        "variant": "st_yolodv2tiny_actrelu_pt_coco_person_192",
        "hyperparameters": "actrelu",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 192,
    },
    {
        "config": _ZOO / "st_yolodv2tiny_pt/ST_pretrainedmodel_public_dataset/coco_person/st_yolodv2tiny_actrelu_pt_coco_person_256/st_yolodv2tiny_actrelu_pt_coco_person_256_config.yaml",
        "model": _ZOO / "st_yolodv2tiny_pt/ST_pretrainedmodel_public_dataset/coco_person/st_yolodv2tiny_actrelu_pt_coco_person_256/st_yolodv2tiny_actrelu_pt_coco_person_256_qdq_int8.onnx",
        "family": "st_yolodv2tiny_pt",
        "readme": _MD_st_yolodv2tiny_pt,
        "variant": "st_yolodv2tiny_actrelu_pt_coco_person_256",
        "hyperparameters": "actrelu",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 256,
    },
    {
        "config": _ZOO / "st_yolodv2tiny_pt/ST_pretrainedmodel_public_dataset/coco_person/st_yolodv2tiny_actrelu_pt_coco_person_288/st_yolodv2tiny_actrelu_pt_coco_person_288_config.yaml",
        "model": _ZOO / "st_yolodv2tiny_pt/ST_pretrainedmodel_public_dataset/coco_person/st_yolodv2tiny_actrelu_pt_coco_person_288/st_yolodv2tiny_actrelu_pt_coco_person_288_qdq_int8.onnx",
        "family": "st_yolodv2tiny_pt",
        "readme": _MD_st_yolodv2tiny_pt,
        "variant": "st_yolodv2tiny_actrelu_pt_coco_person_288",
        "hyperparameters": "actrelu",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 288,
    },
    # ── st_yololcv1 (COCO-Person, Int8) ──
    {
        "config": _ZOO / "st_yololcv1/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yololcv1_192/st_yololcv1_192_config.yaml",
        "model": _ZOO / "st_yololcv1/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yololcv1_192/st_yololcv1_192_int8.tflite",
        "family": "st_yololcv1",
        "readme": _MD_st_yololcv1,
        "variant": "st_yololcv1_192",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 192,
    },
    {
        "config": _ZOO / "st_yololcv1/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yololcv1_224/st_yololcv1_224_config.yaml",
        "model": _ZOO / "st_yololcv1/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yololcv1_224/st_yololcv1_224_int8.tflite",
        "family": "st_yololcv1",
        "readme": _MD_st_yololcv1,
        "variant": "st_yololcv1_224",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 224,
    },
    {
        "config": _ZOO / "st_yololcv1/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yololcv1_256/st_yololcv1_256_config.yaml",
        "model": _ZOO / "st_yololcv1/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yololcv1_256/st_yololcv1_256_int8.tflite",
        "family": "st_yololcv1",
        "readme": _MD_st_yololcv1,
        "variant": "st_yololcv1_256",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 256,
    },
    # ── st_yololcv1 (COCO-Person, W4A8) ──
    {
        "config": _ZOO / "st_yololcv1/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yololcv1_192/st_yololcv1_192_config.yaml",
        "model": _ZOO / "st_yololcv1/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yololcv1_192/st_yololcv1_192_qdq_w4_74.3%_w8_25.7%_a8_100%_map_33.94.onnx",
        "family": "st_yololcv1",
        "readme": _MD_st_yololcv1,
        "variant": "st_yololcv1_192",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "W4A8",
        "resolution": 192,
    },
    {
        "config": _ZOO / "st_yololcv1/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yololcv1_224/st_yololcv1_224_config.yaml",
        "model": _ZOO / "st_yololcv1/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yololcv1_224/st_yololcv1_224_qdq_w4_50.53%_w8_49.47%_a8_100%_map_34.99.onnx",
        "family": "st_yololcv1",
        "readme": _MD_st_yololcv1,
        "variant": "st_yololcv1_224",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "W4A8",
        "resolution": 224,
    },
    {
        "config": _ZOO / "st_yololcv1/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yololcv1_256/st_yololcv1_256_config.yaml",
        "model": _ZOO / "st_yololcv1/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yololcv1_256/st_yololcv1_256_qdq_w4_50.53%_w8_49.47%_a8_100%_map_36.87.onnx",
        "family": "st_yololcv1",
        "readme": _MD_st_yololcv1,
        "variant": "st_yololcv1_256",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "W4A8",
        "resolution": 256,
    },
    # ── st_yoloxn (COCO-Person public, Int8) ──
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_192/st_yoloxn_d033_w025_192_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_192/st_yoloxn_d033_w025_192_int8.tflite",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d033_w025_192",
        "hyperparameters": "d033_w025",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 192,
    },
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_256/st_yoloxn_d033_w025_256_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_256/st_yoloxn_d033_w025_256_int8.tflite",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d033_w025_256",
        "hyperparameters": "d033_w025",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 256,
    },
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_320/st_yoloxn_d033_w025_320_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_320/st_yoloxn_d033_w025_320_int8.tflite",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d033_w025_320",
        "hyperparameters": "d033_w025",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 320,
    },
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_416/st_yoloxn_d033_w025_416_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_416/st_yoloxn_d033_w025_416_int8.tflite",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d033_w025_416",
        "hyperparameters": "d033_w025",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 416,
    },
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d050_w040_256/st_yoloxn_d050_w040_256_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d050_w040_256/st_yoloxn_d050_w040_256_int8.tflite",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d050_w040_256",
        "hyperparameters": "d050_w040",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 256,
    },
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d100_w025_480/st_yoloxn_d100_w025_480_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d100_w025_480/st_yoloxn_d100_w025_480_int8.tflite",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d100_w025_480",
        "hyperparameters": "d100_w025",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 480,
    },
    # ── st_yoloxn (COCO-Person public, W4A8) ──
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_192/st_yoloxn_d033_w025_192_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_192/st_yoloxn_d033_w025_192_qdq_w4_83.16%_w8_16.84%_a8_100%_map_37.34.onnx",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d033_w025_192",
        "hyperparameters": "d033_w025",
        "dataset": "COCO-Person",
        "fmt": "W4A8",
        "resolution": 192,
    },
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_256/st_yoloxn_d033_w025_256_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_256/st_yoloxn_d033_w025_256_qdq_w4_83.16%_w8_16.84%_a8_100%_map_44.43.onnx",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d033_w025_256",
        "hyperparameters": "d033_w025",
        "dataset": "COCO-Person",
        "fmt": "W4A8",
        "resolution": 256,
    },
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_320/st_yoloxn_d033_w025_320_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_320/st_yoloxn_d033_w025_320_qdq_w4_59.47%_w8_40.53%_a8_100%_map_50.61.onnx",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d033_w025_320",
        "hyperparameters": "d033_w025",
        "dataset": "COCO-Person",
        "fmt": "W4A8",
        "resolution": 320,
    },
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_416/st_yoloxn_d033_w025_416_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d033_w025_416/st_yoloxn_d033_w025_416_qdq_w4_76.19%_w8_23.81%_a8_100%_map_53.97.onnx",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d033_w025_416",
        "hyperparameters": "d033_w025",
        "dataset": "COCO-Person",
        "fmt": "W4A8",
        "resolution": 416,
    },
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d050_w040_256/st_yoloxn_d050_w040_256_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d050_w040_256/st_yoloxn_d050_w040_256_qdq_w4_62.53%_w8_37.47%_a8_100%_map_49.2.onnx",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d050_w040_256",
        "hyperparameters": "d050_w040",
        "dataset": "COCO-Person",
        "fmt": "W4A8",
        "resolution": 256,
    },
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d100_w025_480/st_yoloxn_d100_w025_480_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_person/st_yoloxn_d100_w025_480/st_yoloxn_d100_w025_480_qdq_w4_46.51%_w8_53.49%_a8_100%_map_60.42.onnx",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d100_w025_480",
        "hyperparameters": "d100_w025",
        "dataset": "COCO-Person",
        "fmt": "W4A8",
        "resolution": 480,
    },
    # ── st_yoloxn (COCO-80, Int8) ──
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_80_classes/st_yoloxn_d100_w025_480/st_yoloxn_d100_w025_480_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_public_dataset/coco_2017_80_classes/st_yoloxn_d100_w025_480/st_yoloxn_d100_w025_480_qdq_int8.onnx",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d100_w025_480_80cls",
        "hyperparameters": "d100_w025",
        "dataset": "COCO-80",
        "fmt": "Int8",
        "resolution": 480,
    },
    # ── st_yoloxn (custom dataset / ST-Person, Int8) ──
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_custom_dataset/st_person/st_yoloxn_d033_w025_416/st_yoloxn_d033_w025_416_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_custom_dataset/st_person/st_yoloxn_d033_w025_416/st_yoloxn_d033_w025_416_int8.tflite",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d033_w025_416_st",
        "hyperparameters": "d033_w025",
        "dataset": "ST-Person",
        "fmt": "Int8",
        "resolution": 416,
    },
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_custom_dataset/st_person/st_yoloxn_d067_w025_416/st_yoloxn_d067_w025_416_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_custom_dataset/st_person/st_yoloxn_d067_w025_416/st_yoloxn_d067_w025_416_int8.tflite",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d067_w025_416_st",
        "hyperparameters": "d067_w025",
        "dataset": "ST-Person",
        "fmt": "Int8",
        "resolution": 416,
    },
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_custom_dataset/st_person/st_yoloxn_d100_w025_416/st_yoloxn_d100_w025_416_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_custom_dataset/st_person/st_yoloxn_d100_w025_416/st_yoloxn_d100_w025_416_int8.tflite",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d100_w025_416_st",
        "hyperparameters": "d100_w025",
        "dataset": "ST-Person",
        "fmt": "Int8",
        "resolution": 416,
    },
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_custom_dataset/st_person/st_yoloxn_d100_w025_480/st_yoloxn_d100_w025_480_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_custom_dataset/st_person/st_yoloxn_d100_w025_480/st_yoloxn_d100_w025_480_int8.tflite",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d100_w025_480_st",
        "hyperparameters": "d100_w025",
        "dataset": "ST-Person",
        "fmt": "Int8",
        "resolution": 480,
    },
    # ── st_yoloxn (custom dataset / ST-Person, W4A8) ──
    {
        "config": _ZOO / "st_yoloxn/ST_pretrainedmodel_custom_dataset/st_person/st_yoloxn_d100_w025_480/st_yoloxn_d100_w025_480_config.yaml",
        "model": _ZOO / "st_yoloxn/ST_pretrainedmodel_custom_dataset/st_person/st_yoloxn_d100_w025_480/st_yoloxn_d100_w025_480_qdq_w4_78.84%_w8_21.16%_a8_100%_map_47.33.onnx",
        "family": "st_yoloxn",
        "readme": _MD_st_yoloxn,
        "variant": "st_yoloxn_d100_w025_480_st",
        "hyperparameters": "d100_w025",
        "dataset": "ST-Person",
        "fmt": "W4A8",
        "resolution": 480,
    },
    # ── yolov8n (all share yolov8n_256 base config, model_path overridden) ──
    {
        "config": _ZOO / "yolov8n/yolov8n_256_config.yaml",
        "model": _UL / "yolov8n_192_quant_pc_uf_od_coco-person-st.tflite",
        "family": "yolov8n",
        "readme": _MD_yolov8n,
        "variant": "yolov8n_192",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 192,
    },
    {
        "config": _ZOO / "yolov8n/yolov8n_256_config.yaml",
        "model": _UL / "yolov8n_256_quant_pc_uf_od_coco-person-st.tflite",
        "family": "yolov8n",
        "readme": _MD_yolov8n,
        "variant": "yolov8n_256",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 256,
    },
    {
        "config": _ZOO / "yolov8n/yolov8n_256_config.yaml",
        "model": _UL / "yolov8n_320_quant_pc_uf_od_coco-person-st.tflite",
        "family": "yolov8n",
        "readme": _MD_yolov8n,
        "variant": "yolov8n_320",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 320,
    },
    {
        "config": _ZOO / "yolov8n/yolov8n_256_config.yaml",
        "model": _UL / "yolov8n_416_quant_pc_uf_od_coco-person-st.tflite",
        "family": "yolov8n",
        "readme": _MD_yolov8n,
        "variant": "yolov8n_416",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 416,
    },
    # ── yolo11n ──
    {
        "config": _ZOO / "yolov11n/yolov11n_256_config.yaml",
        "model": _UL / "yolo11/yolo11n_256_quant_pc_uf_od_coco-person-st.tflite",
        "family": "yolo11n",
        "readme": _MD_yolov11n,
        "variant": "yolo11n_256",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 256,
    },
    # ── TinyissimoYOLO v8
    {
        "config": _CONFIGS / "tinyissimoyolo_v8_192_config.yaml",
        "model": Path("results")
        / "model"
        / "tinyissimoyolo_v8_192"
        / "weights"
        / "best_saved_model"
        / "best_int8.tflite",
        "family": "tinyissimoyolo_v8",
        "variant": "tinyissimoyolo_v8_192",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 192,
    },
    # ── yolo26 ──
    {
        "config": _CONFIGS / "yolo26_config.yaml",
        "model": _UL / "yolo26/yolo26_256_qdq_int8_od_coco-person-st.onnx",
        "family": "yolo26",
        "variant": "yolo26_256",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 256,
    },
    {
        "config": _CONFIGS / "yolo26_config.yaml",
        "model": _UL / "yolo26/yolo26_320_qdq_int8_od_coco-person-st.onnx",
        "family": "yolo26",
        "variant": "yolo26_320",
        "hyperparameters": "",
        "dataset": "COCO-Person",
        "fmt": "Int8",
        "resolution": 320,
    },
]


def ensure_dirs():
    """Create output directories. Call once from __main__."""
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    os.makedirs(N6_WORKDIR, exist_ok=True)
