"""YAML config generation for the three model template types."""

from pathlib import Path

from .constants import (
    COCO_80_CLASSES,
    COCO_80_ANNOTATIONS,
    COCO_IMAGES,
    COCO_80_TFS_TEST,
    COCO_PERSON_ANNOTATIONS,
    COCO_PERSON_TFS_TEST,
    SSD_FAMILIES,
    STEDGEAI_PATH,
    YOLOD_FAMILIES,
)
from .models import ModelEntry


def _class_names_yaml(num_classes: int) -> str:
    if num_classes == 1:
        return "[person]"
    return "[" + ", ".join(f'"{c}"' for c in COCO_80_CLASSES) + "]"


def _generate_tf_config(entry: ModelEntry) -> str:
    """Template A: TF models (st_yoloxn, st_yololcv1, yolov8n, yolov11n)."""
    if entry.dataset == "COCO-Person":
        test_path = COCO_PERSON_TFS_TEST
    else:
        test_path = COCO_80_TFS_TEST

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
  target: host

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
  target: host

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
  target: host

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
    if entry.model_path and not entry.model_path.startswith(("http://", "https://")):
        entry.model_path = str(Path(entry.model_path).resolve())
    if entry.family in SSD_FAMILIES:
        return _generate_ssd_config(entry)
    if entry.family in YOLOD_FAMILIES:
        return _generate_yolod_config(entry)
    return _generate_tf_config(entry)
