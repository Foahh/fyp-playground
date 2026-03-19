"""
Benchmark all object detection models on STM32N6570-DK.

Walks stm32ai-modelzoo, generates configs, runs evaluation via
stm32ai-modelzoo-services, and collects results into benchmark_results.csv.

Usage:
    python benchmark_od.py
    python benchmark_od.py --filter st_yoloxn_d033_w025_192
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path("D:/FYP/fyp-playground")
MODELZOO_DIR = BASE_DIR / "stm32ai-modelzoo" / "object_detection"
SERVICES_DIR = BASE_DIR / "stm32ai-modelzoo-services" / "object_detection"
OUTPUT_DIR = BASE_DIR / "results"
CSV_PATH = OUTPUT_DIR / "benchmark_results.csv"
ERROR_LOG = OUTPUT_DIR / "benchmark_errors.log"

STEDGEAI_PATH = os.environ.get("STEDGEAI_CORE_DIR", "") + "/Utilities/windows/stedgeai.exe"

# Dataset paths
COCO_PERSON_TFS_TEST = "./datasets/coco_2017_person/test"
COCO_PERSON_ANNOTATIONS = "./datasets/coco/annotations/instances_val2017_person.json"
COCO_80_ANNOTATIONS = "./datasets/coco/annotations/instances_val2017.json"
COCO_IMAGES = "./datasets/coco/val2017"

# COCO 80 class names
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
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# In-scope families
IN_SCOPE_FAMILIES = [
    "ssdlite_mobilenetv1_pt", "ssdlite_mobilenetv2_pt",
    "ssdlite_mobilenetv3large_pt", "ssdlite_mobilenetv3small_pt",
    "st_yolodv2milli_pt", "st_yolodv2tiny_pt",
    "st_yololcv1", "st_yoloxn", "yolov8n", "yolov11n",
]

# Template types
SSD_FAMILIES = {
    "ssdlite_mobilenetv1_pt", "ssdlite_mobilenetv2_pt",
    "ssdlite_mobilenetv3large_pt", "ssdlite_mobilenetv3small_pt",
}
YOLOD_FAMILIES = {"st_yolodv2milli_pt", "st_yolodv2tiny_pt"}
TF_FAMILIES = {"st_yololcv1", "st_yoloxn", "yolov8n", "yolov11n"}

CSV_COLUMNS = [
    "model_family", "model_variant", "hyperparameters", "dataset", "format",
    "resolution", "internal_ram_kib", "external_ram_kib", "weights_flash_kib",
    "inference_time_ms", "inf_per_sec", "ap_50",
]

# Hardcoded remote models (no local files in modelzoo)
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


@dataclass
class ModelEntry:
    family: str
    variant: str
    hyperparameters: str
    dataset: str          # "COCO-Person" or "COCO-80"
    num_classes: int
    fmt: str              # "Int8" or "W4A8"
    resolution: int
    model_path: str       # absolute or URL
    model_type: str       # ssd, st_yolod, st_yoloxn, st_yololcv1, yolov8n, yolov11n
    model_name: str = ""  # for PT models: e.g. ssdlite_mobilenetv1_pt


# ── Discovery ──────────────────────────────────────────────────────────────────

def _parse_resolution(variant_name: str) -> int:
    """Extract the last numeric segment from variant directory name."""
    m = re.findall(r'(\d+)', variant_name)
    return int(m[-1]) if m else 0


def _parse_hyperparams(family: str, variant_name: str) -> str:
    """Extract hyperparameters from variant name."""
    if family == "st_yoloxn":
        m = re.search(r'(d\d+_w\d+)', variant_name)
        return m.group(1) if m else ""
    if family in YOLOD_FAMILIES:
        if "actrelu" in variant_name:
            return "actrelu"
    return ""


def _detect_dataset(path_parts: list[str]) -> tuple[str, int]:
    """Detect dataset from directory path components."""
    path_str = "/".join(path_parts).lower()
    if "coco_2017_person" in path_str or "coco_person" in path_str:
        return "COCO-Person", 1
    if "coco_2017_80_classes" in path_str or "coco" in path_str:
        return "COCO-80", 80
    return "", 0


def _pick_w4a8(files: list[Path]) -> Optional[Path]:
    """Pick the W4A8 ONNX with highest mAP in filename."""
    w4_files = [f for f in files if re.search(r'_qdq_w4_', f.name) and f.suffix == '.onnx']
    if not w4_files:
        return None
    if len(w4_files) == 1:
        return w4_files[0]
    # pick highest map
    best, best_map = w4_files[0], 0.0
    for f in w4_files:
        m = re.search(r'_map_(\d+(?:\.\d+)?)', f.name)
        if m and float(m.group(1)) > best_map:
            best_map = float(m.group(1))
            best = f
    return best


def _model_type_for_family(family: str) -> str:
    if family in SSD_FAMILIES:
        return "ssd"
    if family in YOLOD_FAMILIES:
        return "st_yolod"
    if family == "st_yoloxn":
        return "st_yoloxn"
    if family == "st_yololcv1":
        return "st_yololcv1"
    if family == "yolov8n":
        return "yolov8n"
    if family == "yolov11n":
        return "yolov11n"
    return family


def _model_name_for_ssd(family: str) -> str:
    """Return the model_name string expected by the SSD framework."""
    return family  # e.g. "ssdlite_mobilenetv1_pt"


def _model_name_for_yolod(variant_name: str) -> str:
    """e.g. st_yolodv2milli_actrelu_pt from st_yolodv2milli_actrelu_pt_coco_person_192."""
    # Pattern: st_yolodv2{size}_actrelu_pt
    m = re.match(r'(st_yolodv2\w+_actrelu_pt)', variant_name)
    return m.group(1) if m else variant_name


def discover_models() -> list[ModelEntry]:
    """Walk modelzoo tree and return list of ModelEntry."""
    entries = []

    for family in IN_SCOPE_FAMILIES:
        family_dir = MODELZOO_DIR / family

        # Handle remote-only models (yolov8n, yolov11n)
        if family in REMOTE_MODELS:
            info = REMOTE_MODELS[family]
            entries.append(ModelEntry(
                family=family,
                variant=f"{family}_{info['resolution']}",
                hyperparameters="",
                dataset=info["dataset"],
                num_classes=info["num_classes"],
                fmt="Int8",
                resolution=info["resolution"],
                model_path=info["model_path"],
                model_type=info["model_type"],
            ))
            continue

        if not family_dir.is_dir():
            print(f"[WARN] Family dir not found: {family_dir}")
            continue

        # Walk to find variant directories (leaf dirs containing model files)
        for root, dirs, files in os.walk(family_dir):
            root_path = Path(root)
            path_parts = root_path.relative_to(MODELZOO_DIR).parts

            # Skip VOC variants
            if any(p.lower() == "voc" for p in path_parts):
                continue

            # Skip custom dataset variants (st_person)
            if any("custom_dataset" in p.lower() for p in path_parts):
                continue

            # Check if this directory has model files
            all_files = [root_path / f for f in files]
            onnx_files = [f for f in all_files if f.suffix == '.onnx']
            tflite_files = [f for f in all_files if f.suffix == '.tflite']

            if not onnx_files and not tflite_files:
                continue

            variant_name = root_path.name
            dataset, num_classes = _detect_dataset(list(path_parts))
            if not dataset:
                continue

            resolution = _parse_resolution(variant_name)
            hyperparams = _parse_hyperparams(family, variant_name)
            model_type = _model_type_for_family(family)

            # Determine model_name for PT models
            model_name = ""
            if family in SSD_FAMILIES:
                model_name = _model_name_for_ssd(family)
            elif family in YOLOD_FAMILIES:
                model_name = _model_name_for_yolod(variant_name)

            # W4A8 entry
            w4_file = _pick_w4a8(all_files)
            if w4_file:
                entries.append(ModelEntry(
                    family=family, variant=variant_name,
                    hyperparameters=hyperparams, dataset=dataset,
                    num_classes=num_classes, fmt="W4A8",
                    resolution=resolution,
                    model_path=str(w4_file),
                    model_type=model_type, model_name=model_name,
                ))

            # Int8 entry: prefer tflite, else qdq_int8.onnx
            int8_tflite = [f for f in tflite_files if "_int8" in f.stem]
            int8_onnx = [f for f in onnx_files if "_qdq_int8" in f.name]

            if int8_tflite:
                entries.append(ModelEntry(
                    family=family, variant=variant_name,
                    hyperparameters=hyperparams, dataset=dataset,
                    num_classes=num_classes, fmt="Int8",
                    resolution=resolution,
                    model_path=str(int8_tflite[0]),
                    model_type=model_type, model_name=model_name,
                ))
            elif int8_onnx:
                entries.append(ModelEntry(
                    family=family, variant=variant_name,
                    hyperparameters=hyperparams, dataset=dataset,
                    num_classes=num_classes, fmt="Int8",
                    resolution=resolution,
                    model_path=str(int8_onnx[0]),
                    model_type=model_type, model_name=model_name,
                ))

    return entries


# ── Config Generation ──────────────────────────────────────────────────────────

def _class_names_yaml(num_classes: int) -> str:
    if num_classes == 1:
        return "[person]"
    return "[" + ", ".join(f'"{c}"' for c in COCO_80_CLASSES) + "]"


def _generate_tf_config(entry: ModelEntry) -> str:
    """Template A: TF models (st_yoloxn, st_yololcv1, yolov8n, yolov11n)."""
    if entry.dataset == "COCO-Person":
        test_path = COCO_PERSON_TFS_TEST
    else:
        test_path = "./datasets/coco_2017_80_classes/test"

    return f"""\
operation_mode: evaluation

model:
  model_type: {entry.model_type}
  model_path: {entry.model_path}

evaluation:
  profile: profile_O3
  input_type: uint8
  output_type: int8
  input_chpos: chlast
  output_chpos: chlast
  target: stedgeai_n6

dataset:
  format: tfs
  dataset_name: coco
  class_names: {_class_names_yaml(entry.num_classes)}
  test_path: {test_path}

preprocessing:
  rescaling: {{scale: 1/255, offset: 0}}
  resizing:
    aspect_ratio: fit
    interpolation: nearest
  color_mode: rgb

postprocessing:
  confidence_thresh: 0.001
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  plot_metrics: false
  max_detection_boxes: 100

tools:
  stedgeai:
    path_to_stedgeai: {STEDGEAI_PATH}

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${{now:%Y_%m_%d_%H_%M_%S}}
"""


def _generate_ssd_config(entry: ModelEntry) -> str:
    """Template B: PT SSD models (ssdlite_*)."""
    if entry.dataset == "COCO-Person":
        annotations = COCO_PERSON_ANNOTATIONS
        pretrained_dataset = "coco_person"
    else:
        annotations = COCO_80_ANNOTATIONS
        pretrained_dataset = "coco"

    return f"""\
operation_mode: evaluation

model:
  framework: 'torch'
  model_type: ssd
  model_path: {entry.model_path}
  model_name: {entry.model_name}
  width_mult: 1.0
  pretrained: true
  pretrained_dataset: {pretrained_dataset}
  input_shape: [3, 300, 300]
  num_classes: {entry.num_classes}

evaluation:
  profile: profile_O3
  input_type: uint8
  output_type: int8
  input_chpos: chlast
  output_chpos: chfirst
  target: stedgeai_n6

dataset:
  format: coco
  dataset_name: coco
  class_names: {_class_names_yaml(entry.num_classes)}
  num_workers: 0
  training_path: ""
  test_annotations_path: {annotations}
  test_images_path: {COCO_IMAGES}

preprocessing:
  mean: [127, 127, 127]
  std: 128.0
  rescaling: {{scale: 1, offset: 0}}
  resizing:
    aspect_ratio: fit
    interpolation: nearest
  color_mode: rgb

postprocessing:
  confidence_thresh: 0.001
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  plot_metrics: false
  max_detection_boxes: 100

tools:
  stedgeai:
    path_to_stedgeai: {STEDGEAI_PATH}

mlflow:
  uri: ./pt/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./pt/src/experiments_outputs/${{now:%Y_%m_%d_%H_%M_%S}}
"""


def _generate_yolod_config(entry: ModelEntry) -> str:
    """Template C: PT YOLOD models (st_yolodv2milli_pt, st_yolodv2tiny_pt)."""
    if entry.dataset == "COCO-Person":
        annotations = COCO_PERSON_ANNOTATIONS
        pretrained_dataset = "coco_person"
    else:
        annotations = COCO_80_ANNOTATIONS
        pretrained_dataset = "coco"

    h = w = entry.resolution

    return f"""\
operation_mode: evaluation

model:
  framework: 'torch'
  model_type: st_yolod
  model_path: {entry.model_path}
  model_name: {entry.model_name}
  input_shape: [3, {h}, {w}]
  pretrained_input_shape: [3, {h}, {w}]
  pretrained: true
  pretrained_dataset: {pretrained_dataset}
  num_classes: {entry.num_classes}

evaluation:
  profile: profile_O3
  input_type: uint8
  output_type: int8
  input_chpos: chlast
  output_chpos: chlast
  target: stedgeai_n6

dataset:
  format: coco
  dataset_name: coco
  class_names: {_class_names_yaml(entry.num_classes)}
  num_workers: 1
  training_path: ""
  test_annotations_path: {annotations}
  test_images_path: {COCO_IMAGES}

preprocessing:
  rescaling: {{scale: 1, offset: 0}}
  resizing:
    aspect_ratio: fit
    interpolation: nearest
  color_mode: rgb

postprocessing:
  confidence_thresh: 0.001
  NMS_thresh: 0.5
  IoU_eval_thresh: 0.5
  plot_metrics: false
  max_detection_boxes: 100

tools:
  stedgeai:
    path_to_stedgeai: {STEDGEAI_PATH}

mlflow:
  uri: ./pt/src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./pt/src/experiments_outputs/${{now:%Y_%m_%d_%H_%M_%S}}
"""


def generate_config(entry: ModelEntry) -> str:
    if entry.family in SSD_FAMILIES:
        return _generate_ssd_config(entry)
    if entry.family in YOLOD_FAMILIES:
        return _generate_yolod_config(entry)
    return _generate_tf_config(entry)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def run_evaluation(entry: ModelEntry) -> tuple[str, str, int]:
    """Write temp config, invoke stm32ai_main.py, return (stdout, stderr, returncode)."""
    config_path = SERVICES_DIR / "_benchmark_temp_config.yaml"
    config_content = generate_config(entry)
    config_path.write_text(config_content, encoding="utf-8")

    cmd = [
        sys.executable, "stm32ai_main.py",
        "--config-path", ".",
        "--config-name", "_benchmark_temp_config",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(SERVICES_DIR),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT: evaluation exceeded 1 hour", -1


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_metrics(stdout: str, stderr: str) -> dict:
    """Extract ap_50 and memory/inference metrics from output."""
    metrics = {
        "ap_50": "",
        "internal_ram_kib": "",
        "external_ram_kib": "",
        "weights_flash_kib": "",
        "inference_time_ms": "",
        "inf_per_sec": "",
    }

    combined = stdout + "\n" + stderr

    # mAP from evaluator stdout
    m = re.search(r'Mean AP \(mAP\):\s+([\d.]+)', combined)
    if m:
        metrics["ap_50"] = m.group(1)

    # Memory from common_benchmark _analyze_footprints output
    m = re.search(r'\[INFO\]\s*:\s*RAM Activations\s*:\s*([\d.]+)\s*\(KiB\)', combined)
    if m:
        metrics["internal_ram_kib"] = m.group(1)

    m = re.search(r'\[INFO\]\s*:\s*Flash weights\s*:\s*([\d.]+)\s*\(KiB\)', combined)
    if m:
        metrics["weights_flash_kib"] = m.group(1)

    # Also try generic patterns
    if not metrics["internal_ram_kib"]:
        m = re.search(r'Internal RAM[:\s]+([\d.]+)\s*KiB', combined)
        if m:
            metrics["internal_ram_kib"] = m.group(1)

    if not metrics["external_ram_kib"]:
        m = re.search(r'External RAM[:\s]+([\d.]+)\s*KiB', combined)
        if m:
            metrics["external_ram_kib"] = m.group(1)

    if not metrics["weights_flash_kib"]:
        m = re.search(r'(?:Flash|Weights)[:\s]+([\d.]+)\s*KiB', combined)
        if m:
            metrics["weights_flash_kib"] = m.group(1)

    # Inference time
    m = re.search(r'(?:duration|Inference)[:\s]+([\d.]+)\s*ms', combined)
    if m:
        metrics["inference_time_ms"] = m.group(1)
        try:
            ms = float(m.group(1))
            if ms > 0:
                metrics["inf_per_sec"] = f"{1000.0 / ms:.2f}"
        except ValueError:
            pass

    # Fallback: try network_c_info.json
    if not metrics["internal_ram_kib"] or not metrics["weights_flash_kib"]:
        _try_parse_network_c_info(metrics)

    return metrics


def _try_parse_network_c_info(metrics: dict):
    """Fallback: parse network_c_info.json from stedgeai output."""
    stedgeai_dir = os.environ.get("STEDGEAI_CORE_DIR", "")
    if not stedgeai_dir:
        return

    json_path = Path(stedgeai_dir) / "scripts" / "N6_scripts" / "st_ai_output" / "network_c_info.json"
    if not json_path.exists():
        return

    try:
        with open(json_path, 'r') as f:
            cinfo = json.load(f)
        mem = cinfo.get("memory_footprint", {})
        if not metrics["internal_ram_kib"]:
            act = mem.get("activations", 0)
            if act:
                metrics["internal_ram_kib"] = f"{act / 1024:.2f}"
        if not metrics["weights_flash_kib"]:
            w = mem.get("weights", 0)
            if w:
                metrics["weights_flash_kib"] = f"{w / 1024:.2f}"
    except Exception:
        pass


# ── CSV ────────────────────────────────────────────────────────────────────────

def load_completed() -> set[tuple[str, str]]:
    """Read CSV and return set of (variant, format) keys already done."""
    completed = set()
    if not CSV_PATH.exists():
        return completed
    with open(CSV_PATH, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("model_variant", ""), row.get("format", ""))
            completed.add(key)
    return completed


def append_result(row: dict):
    """Append one row to CSV, creating file + header if needed."""
    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def log_error(msg: str):
    """Append error to log file."""
    with open(ERROR_LOG, 'a', encoding='utf-8') as f:
        f.write(msg + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark OD models on STM32N6570-DK")
    parser.add_argument("--filter", type=str, default="",
                        help="Only run variants whose name contains this string")
    args = parser.parse_args()

    entries = discover_models()

    # Sort for deterministic order: family, variant, format
    entries.sort(key=lambda e: (e.family, e.variant, e.fmt))

    if args.filter:
        entries = [e for e in entries if args.filter in e.variant]

    completed = load_completed()
    total = len(entries)

    print(f"Discovered {total} model entries to benchmark.")
    print(f"Already completed: {len(completed)} entries in CSV.")
    print()

    for i, entry in enumerate(entries, 1):
        key = (entry.variant, entry.fmt)
        tag = f"[{i}/{total}]"

        if key in completed:
            print(f"{tag} SKIPPED: {entry.variant} ({entry.fmt}) — already in CSV")
            continue

        print(f"{tag} Running: {entry.variant} ({entry.fmt}) ...")

        try:
            stdout, stderr, rc = run_evaluation(entry)

            if rc != 0:
                err_msg = f"{tag} FAILED (rc={rc}): {entry.variant} ({entry.fmt})\n"
                err_msg += f"  stderr (last 500 chars): {stderr[-500:]}\n"
                print(f"{tag} FAILED: {entry.variant} ({entry.fmt}) — rc={rc}")
                log_error(err_msg)
                # Still try to parse partial results
                metrics = parse_metrics(stdout, stderr)
            else:
                metrics = parse_metrics(stdout, stderr)

            row = {
                "model_family": entry.family,
                "model_variant": entry.variant,
                "hyperparameters": entry.hyperparameters,
                "dataset": entry.dataset,
                "format": entry.fmt,
                "resolution": entry.resolution,
                "internal_ram_kib": metrics.get("internal_ram_kib", ""),
                "external_ram_kib": metrics.get("external_ram_kib", ""),
                "weights_flash_kib": metrics.get("weights_flash_kib", ""),
                "inference_time_ms": metrics.get("inference_time_ms", ""),
                "inf_per_sec": metrics.get("inf_per_sec", ""),
                "ap_50": metrics.get("ap_50", ""),
            }

            append_result(row)

            ap = metrics.get("ap_50", "N/A")
            inf = metrics.get("inference_time_ms", "N/A")
            print(f"{tag} DONE: ap_50={ap}, inference={inf}ms")

        except Exception as exc:
            err_msg = f"{tag} EXCEPTION: {entry.variant} ({entry.fmt}): {exc}\n"
            print(f"{tag} ERROR: {entry.variant} ({entry.fmt}) — {exc}")
            log_error(err_msg)

    print("\nBenchmark complete. Results in:", CSV_PATH)


if __name__ == "__main__":
    main()
