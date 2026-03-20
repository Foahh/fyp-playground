"""Paths, class lists, family sets, CSV columns, and remote model definitions."""

import os
import platform
from pathlib import Path


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


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELZOO_DIR = BASE_DIR / "stm32ai-modelzoo" / "object_detection"
SERVICES_DIR = BASE_DIR / "stm32ai-modelzoo-services" / "object_detection"
OUTPUT_DIR = BASE_DIR / "results"

CSV_PATH = OUTPUT_DIR / "benchmark_results.csv"
ERROR_LOG = OUTPUT_DIR / "benchmark_errors.log"
STDOUT_LOG = OUTPUT_DIR / "benchmark_stdout.log"

STEDGEAI_PATH = get_stedgeai_path()

N6_WORKDIR = OUTPUT_DIR / "n6_workdir"

# Dataset paths
COCO_PERSON_TFS_TEST = str(BASE_DIR / "datasets" / "coco_2017_person" / "test")
COCO_PERSON_ANNOTATIONS = str(
    BASE_DIR / "datasets" / "coco" / "annotations" / "instances_val2017_person.json"
)
COCO_80_ANNOTATIONS = str(
    BASE_DIR / "datasets" / "coco" / "annotations" / "instances_val2017.json"
)
COCO_IMAGES = str(BASE_DIR / "datasets" / "coco" / "images" / "val2017")

# COCO 80 class names
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

# In-scope families
IN_SCOPE_FAMILIES = [
    "ssdlite_mobilenetv1_pt",
    "ssdlite_mobilenetv2_pt",
    "ssdlite_mobilenetv3large_pt",
    "ssdlite_mobilenetv3small_pt",
    "st_yolodv2milli_pt",
    "st_yolodv2tiny_pt",
    "st_yololcv1",
    "st_yoloxn",
    "yolov8n",
    "yolov11n",
]

# Template types
SSD_FAMILIES = {
    "ssdlite_mobilenetv1_pt",
    "ssdlite_mobilenetv2_pt",
    "ssdlite_mobilenetv3large_pt",
    "ssdlite_mobilenetv3small_pt",
}
YOLOD_FAMILIES = {"st_yolodv2milli_pt", "st_yolodv2tiny_pt"}
TF_FAMILIES = {"st_yololcv1", "st_yoloxn", "yolov8n", "yolov11n"}

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
]

# Hardcoded remote models
REMOTE_MODELS = {
    "yolov8n": {
        "model_path": "https://github.com/stm32-hotspot/ultralytics/raw/refs/heads/main/examples/YOLOv8-STEdgeAI/stedgeai_models/object_detection/yolov8n_256_quant_pc_uf_od_coco-person-st.tflite",
        "model_type": "yolov8n",
        "resolution": 256,
        "dataset": "COCO-Person",
        "num_classes": 1,
    },
    "yolov11n": {
        "model_path": "https://github.com/stm32-hotspot/ultralytics/blob/main/examples/YOLOv8-STEdgeAI/stedgeai_models/object_detection/yolo11/yolo11n_256_quant_pc_uf_od_coco-person-st.tflite",
        "model_type": "yolov11n",
        "resolution": 256,
        "dataset": "COCO-Person",
        "num_classes": 1,
    },
}


def ensure_dirs():
    """Create output directories. Call once from __main__."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(N6_WORKDIR, exist_ok=True)
