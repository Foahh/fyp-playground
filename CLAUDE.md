# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project Overview

FYP research workspace for TinyissimoYOLO training, INT8 TFLite quantization, STM32N6570-DK benchmarking, and STM32 Model Zoo finetuning. Uses STEdgeAI 4.0.

## Command Entry Point

All commands run through `project.py` from the repository root:

```bash
python project.py <command> [args]
```

## Conda Environments

Two environments are required:

- **fyp-ml** (`ST_YOLO_ENV`): Dataset prep, training, quantization
- **fyp-bhmk** (`ST_STZOO_ENV`): Benchmarking, Model Zoo finetuning

Setup:
```bash
python project.py setup-conda-ml
python project.py setup-conda-bhmk
```

## Common Commands

### Training (fyp-ml)
```bash
conda activate fyp-ml
python project.py train --size 192              # Train TinyissimoYOLO
python project.py train --size 192 --profile paper
python project.py quantize --size 192           # Export to INT8 TFLite
```

### Benchmarking (fyp-bhmk)
```bash
conda activate fyp-bhmk
python project.py benchmark                     # Run both underdrive and overdrive
python project.py benchmark --mode underdrive   # Underdrive (400 MHz: NO_OVD_CLK400 on, USE_OVERDRIVE 0)
python project.py benchmark --mode nominal      # Nominal (600 MHz: NO_OVD_CLK400 commented, USE_OVERDRIVE 0)
python project.py benchmark --filter st_yoloxn_d033_w025_192  # Single model
python project.py compare-runs                  # Compare against README metrics
```

### Dataset Preparation (fyp-ml)
```bash
conda activate fyp-ml
python project.py download-coco
python project.py download-finetune
```

### Model Zoo Finetuning (fyp-bhmk)
```bash
conda activate fyp-bhmk
python project.py prepare-finetune-dataset -- --config configs/finetune_dataset.yaml
python project.py finetune -- --config configs/finetune.yaml --mode training
```

## Architecture

### Source Structure

- `src/dataset/` — COCO and finetune dataset download/preparation
- `src/ml/` — Training and quantization runners
- `src/benchmark/` — STM32 benchmarking workflow
  - `core/` — Model registry, config loading
  - `execution/` — Benchmark workflow, power measurement
  - `io/` — CSV parsing, results handling
- `src/conda/` — Conda environment setup scripts
- `src/common/` — Shared utilities and path helpers

### Key Files

- `project.py` — Unified CLI dispatcher mapping commands to Python modules
- `configs/model_registry.yaml` — Benchmark model definitions
- `configs/tinyissimoyolo_v8_192_config.yaml` — Export config for TinyissimoYOLO
- `configs/finetune.yaml` — Model Zoo training config
- `configs/finetune_dataset.yaml` — Dataset conversion config

### External Dependencies

- `external/TinyissimoYOLO/` — Modified Ultralytics fork for training
- `external/stm32ai-modelzoo/` — STM32 Model Zoo for finetuning
- `external/stm32ai-modelzoo-services/` — Benchmarking services
- `external/fyp-power-measure/` — Optional INA228 power logging

### Output Locations

- Training: `results/model/tinyissimoyolo_v8_<size>/weights/best.pt`
- Quantized models: `results/model/tinyissimoyolo_v8_<size>/weights/best_saved_model/best_int8.tflite`
- Benchmark results: `results/benchmark_underdrive/`, `results/benchmark_nominal/`, or `results/benchmark_overdrive/` (each contains `benchmark_results.csv`)
- Power logs: `results/benchmark_<mode>/power_measure.csv` (same subdir as the active mode)

## Environment Variables

- `STEDGEAI_CORE_DIR` — Path to STEdgeAI installation (required for benchmarking)
- `DATASETS_DIR` — Override default `./datasets` location

## Hardware Requirements

Benchmarking requires:
- STM32N6570-DK board connected via USB
- STM32CubeCLT installed with `arm-none-eabi-*` in PATH
- User in `dialout` group (Linux)
- `$STEDGEAI_CORE_DIR/scripts/N6_scripts/config.json` set to `{"compiler_type": "gcc"}`

Optional power measurement:
- ESP32 + INA228 hardware (see `external/fyp-power-measure/README.md`)

## Workflow Notes

- Always activate the correct Conda environment before running commands
- Training outputs checkpoints that quantization consumes
- Benchmark mode patches `app_config.h`: `USE_OVERDRIVE`, and for underdrive vs nominal the `NO_OVD_CLK400` define (see `main.c` clock selection)
- Model registry in `configs/model_registry.yaml` defines which models to benchmark
- Hydra configs default output to `./configs/outputs/`
