# FYP Playground

Workspace for FYP research, model training, export, quantization, and on-device benchmarking.

**STEdgeAI version:** `4.0`

---

## Overview

This repository provides a unified workflow for:

- dataset setup
- model training
- model export and quantization
- benchmarking on STM32 hardware

For most workflows, use `project.py` as the main entry point.

---

## Getting Started

Before running any project components, initialize the Git submodules:

```sh
git submodule update --init --recursive
```

Use **Conda** as the default environment for training/export/quantization/benchmark workflows.

### Conda environments by workflow

There are **three** conda envs (defaults shown). Override names with env vars if needed.

| Purpose | Default env name | One-time setup |
|--------|------------------|----------------|
| Dataset download/prep (`load_coco.py`, `load_finetune_data.py`, Dataset Ninja) | `dataset` (`ST_DATASET_ENV`) | `python project.py conda-dataset` |
| TinyissimoYOLO training / export (`run_train_tinyissimo_coco_person.py`) | `yolo` (`ST_YOLO_ENV`) | `python project.py conda-yolo` |
| Quantization, on-device benchmark, Model Zoo finetune helpers | `stzoo` (`ST_STZOO_ENV`) | `python project.py conda-benchmark` |

Before each step, run `conda activate <env>` for the row that matches the command.

---

## Unified Command Runner

Use `project.py` as the single entry point for common workflows.

**Required conda env** for each command (after `conda activate …`):

| Command | Conda env |
|---------|-----------|
| `dataset-coco`, `dataset-finetune` | `dataset` |
| `train` | `yolo` |
| `quant`, `benchmark`, `finetune-dataset`, `finetune` | `stzoo` |
| `conda-dataset`, `conda-yolo`, `conda-benchmark` | none (run from base or any env with `conda` available) |

### Local commands

**Env: `dataset`** — COCO download:

```sh
conda activate dataset
python project.py dataset-coco
```

**Env: `stzoo`** — hardware benchmark:

```sh
conda activate stzoo
python project.py benchmark --filter st_yoloxn_d033_w025_192
```

### Workflow commands

**Env: `yolo`** — training:

```sh
conda activate yolo
python project.py train --size 192
```

**Env: `stzoo`** — quantization:

```sh
conda activate stzoo
python project.py quant --img_size 192
```

---

## Linux Setup

On Linux, install **STEdgeAI** in a user-owned location such as `~/ST/STEdgeAI` to avoid permission issues.

After installation, add the following to your `~/.bashrc`:

```sh
export STEDGEAI_CORE_DIR="$HOME/ST/STEdgeAI/4.0"
```

Then add your user to the `dialout` group:

```sh
sudo usermod -aG dialout $USER
```

Log out and back in for the group change to take effect.

---

## Dataset Setup

**Conda environment:** `dataset` — create it with `python project.py conda-dataset`, then `conda activate dataset`.

Prepare datasets (default: `./datasets` under project root):

```sh
conda activate dataset
mkdir -p ./datasets
python ./scripts/load_coco.py
```

To store datasets elsewhere, set `DATASETS_DIR`:

```sh
conda activate dataset
DATASETS_DIR=~/datasets python ./scripts/load_coco.py
```

---

## Train TinyissimoYOLO

### Requirements

Training requires:

- **Python 3.12** (provided by the `yolo` conda env from `conda-yolo`)
- additional Python packages

For complete setup instructions, see:

- [external/TinyissimoYOLO/tinyissimoYOLO_README.md](external/TinyissimoYOLO/tinyissimoYOLO_README.md)

### Conda setup

**Conda environment:** `yolo` — set it up before training:

```sh
python project.py conda-yolo
conda activate yolo
```

`conda-yolo` already installs PyTorch from the CUDA 12.8 wheel index (RTX 50-series). To reinstall or upgrade manually:

```sh
conda activate yolo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Run training

**Conda environment:** `yolo`.

From the repository root, outputs are written to:

```text
results/model/
```

Run training for different input sizes (training auto-exports SavedModel after completion):

```sh
conda activate yolo
python project.py train --size 192
python project.py train --size 256
python project.py train --size 288
python project.py train --size 320
```

Force export-only (skip training) from the latest checkpoint:

```sh
conda activate yolo
python project.py train --size 192 --export
```

SavedModel export uses:

```text
results/model/tinyissimoyolo_v8_<size>/weights/best.pt
```

---

## Quantize to TFLite INT8

**Conda environment:** `stzoo` — `python project.py conda-benchmark`, then `conda activate stzoo`.

### Quantize SavedModel to TFLite INT8

```sh
conda activate stzoo
python project.py quant \
  --img_size 192
```

`project.py quant` now runs quantization through `stm32ai-modelzoo-services` and then
evaluates the generated TFLite model by default.

Useful options (still with `stzoo` activated):

```sh
conda activate stzoo
python project.py quant --img_size 192 --no-eval
python project.py quant --img_size 192 --saved-model /path/to/saved_model
python project.py quant --img_size 192 --out /path/to/output.tflite
```

---

## Benchmark on STM32N6570-DK

**Conda environment:** `stzoo` — same as quantization (`python project.py conda-benchmark`).

This benchmark performs on-device evaluation for all supported model variants and saves results to:

```text
results/benchmark_nominal/benchmark_results.csv
```

### Additional reading

For more details, see:

- [docs/stm32n6_getting_started.md](docs/stm32n6_getting_started.md)

Before benchmarking, ensure:

- **STM32CubeCLT** is installed and `arm-none-eabi-*` is in PATH.
- `$STEDGEAI_CORE_DIR/scripts/N6_scripts/config.json` is configured correctly

```json
{
	"compiler_type": "gcc",
}
```

### Requirements

Benchmarking requires:

- **Python 3.12.9**
- additional Python packages

For full setup instructions, see:

- [external/stm32ai-modelzoo-services/README.md](external/stm32ai-modelzoo-services/README.md#before-you-start)

### Conda environment setup

```sh
python project.py conda-benchmark
conda activate stzoo
```

To use a different environment name instead of the default (`stzoo`):

```sh
ST_STZOO_ENV=my-stzoo-env python project.py conda-benchmark
conda activate my-stzoo-env
```

### Prerequisites

- STM32N6570-DK board connected via USB
- `STEDGEAI_CORE_DIR` environment variable set
- ESPS3-C3 connected with INA228
- Arduino IDE available to flash `external/fyp-power-measure/fyp-power-measure.ino`

### Optional power measurement

To enable inference-window power logging (INA228 + ESP32-C6), follow:

- [external/fyp-power-measure/README.md](external/fyp-power-measure/README.md)

This guide covers wiring, Arduino sketch flashing, required ST Edge AI patching, CLI flags (`--power-serial`, `--power-baud`, `--validation-count`), and troubleshooting.

In `results/benchmark_nominal/benchmark_results.csv`, power-measure averages use the `pm_avg_*` columns (e.g. `pm_avg_inf_mW`, `pm_avg_idle_mW`, `pm_avg_delta_mW`, `pm_avg_inf_ms`, `pm_avg_idle_ms`, `pm_avg_inf_mJ`, `pm_avg_idle_mJ`).

When power serial is enabled, the benchmark also appends a continuous log to:

```text
results/benchmark_nominal/power_measure.csv
```

This log contains host timestamps and INA228 fields. The Arduino sketch waits for `START` on the serial line, which the benchmark sends when opening the port.

### Run benchmark

**Conda environment:** `stzoo`.

Test a single model first:

```sh
conda activate stzoo
python scripts/run_benchmark.py --filter st_yoloxn_d033_w025_192
```

Run the full benchmark suite for all variants:

```sh
conda activate stzoo
python scripts/run_benchmark.py
```

Run nominal first, pause 5 seconds, then overdrive before going to bed:

```sh
conda activate stzoo
python scripts/run_benchmark_nominal_overdrive.py
```