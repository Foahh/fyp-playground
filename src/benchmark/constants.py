"""Constants: class lists, CSV columns, and configuration values."""

# ── COCO 80 class names ──

COCO_80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
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
    "host_time_iso",
    "stedgeai_version",
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
    "pm_avg_inf_mW",
    "pm_avg_idle_mW",
    "pm_avg_delta_mW",
    "pm_avg_inf_ms",
    "pm_avg_idle_ms",
    "pm_avg_inf_mJ",
    "pm_avg_idle_mJ",
]

CSV_COLUMNS_NO_POWER = [
    c for c in CSV_COLUMNS
    if c not in (
        "pm_avg_inf_mW", "pm_avg_idle_mW", "pm_avg_delta_mW",
        "pm_avg_inf_ms", "pm_avg_idle_ms", "pm_avg_inf_mJ", "pm_avg_idle_mJ",
    )
]
