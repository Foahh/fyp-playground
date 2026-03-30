# FYP Playground

Workspace for FYP research, model training, INT8 TFLite quantization, and on-device benchmarking.

**STEdgeAI version:** `4.0`

---

## Overview

This repository provides a unified workflow for:

- dataset setup
- model training
- model training and INT8 TFLite quantization (Ultralytics)
- benchmarking on STM32 hardware

For most workflows, use `project.py` as the main entry point.

---

## Getting Started

Before running any project components, initialize the Git submodules:

```sh
git submodule update --init --recursive
```

Use **Conda** as the default environment for training, quantization, and benchmark workflows.

### Conda environments by workflow

There are **two** conda envs (defaults shown). Override names with env vars if needed.

| Purpose | Default env name | One-time setup |
|--------|------------------|----------------|
| Dataset download/prep (`load_coco.py`, `load_finetune_data.py`), TinyissimoYOLO training (`run_train_tinyissimo_coco_person.py`), INT8 TFLite quantization (`run_quantize.py`) | `yolo` (`ST_YOLO_ENV`) | `python project.py conda-yolo` |
| On-device benchmark, Model Zoo finetune helpers | `stzoo` (`ST_STZOO_ENV`) | `python project.py conda-benchmark` |

Before each step, run `conda activate <env>` for the row that matches the command.

---

## Unified Command Runner

Use `project.py` as the single entry point for common workflows.

**Required conda env** for each command (after `conda activate …`):

| Command | Conda env |
|---------|-----------|
| `dataset-coco`, `dataset-finetune`, `train`, `quant` | `yolo` |
| `benchmark`, `compare`, `finetune-dataset`, `finetune` | `stzoo` |
| `conda-yolo`, `conda-benchmark` | none (run from base or any env with `conda` available) |

### Local commands

**Env: `yolo`** — COCO download:

```sh
conda activate yolo
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

**Env: `yolo`** — INT8 TFLite quantization (after training):

```sh
conda activate yolo
python project.py quant --size 192
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

**Conda environment:** `yolo` — same as training (`python project.py conda-yolo`, then `conda activate yolo`).

Prepare datasets (default: `./datasets` under project root):

```sh
conda activate yolo
mkdir -p ./datasets
python ./scripts/load_coco.py
```

To store datasets elsewhere, set `DATASETS_DIR`:

```sh
conda activate yolo
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

Run training for different input sizes (checkpoints only; quantize to TFLite separately):

```sh
conda activate yolo
python project.py train --size 192
python project.py train --size 256
python project.py train --size 288
python project.py train --size 320
```

Weights are written under:

```text
results/model/tinyissimoyolo_v8_<size>/weights/best.pt
```

---

## Quantize to TFLite INT8

**Conda environment:** `yolo` (same as training; Ultralytics `model.export(format="tflite", int8=True, …)` via ONNX → onnx2tf).

### Export INT8 TFLite from `best.pt`

```sh
conda activate yolo
python project.py quant --size 192
```

Ultralytics writes the INT8 TFLite next to the checkpoint (no separate `quantized/` copy). With the default `best.pt` layout that is:

```text
results/model/tinyissimoyolo_v8_<size>/weights/best_saved_model/best_int8.tflite
```

The script prints this path at the end as the artifact. By default it runs Ultralytics validation on the test split after export; metrics and plots go under:

```text
results/model/tinyissimoyolo_v8_<size>/weights/best_saved_model/val_int8/
```

Optional flags:

```sh
conda activate yolo
python project.py quant --size 192 --no-eval
python project.py quant --size 192 --checkpoint /path/to/best.pt
```

To evaluate with the STM32 model zoo host pipeline instead, use `configs/tinyissimoyolo_v8_192_config.yaml` (update `model.model_path` if your checkpoint stem or export location differs) and run `stm32ai_main.py` from the `stzoo` environment as documented in `external/stm32ai-modelzoo-services`.

---

## Benchmark on STM32N6570-DK

**Conda environment:** `stzoo` — create with `python project.py conda-benchmark` (quantization uses the `yolo` env above).

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
PYTHONPATH=scripts python -m benchmark --filter st_yoloxn_d033_w025_192
```

Run the full benchmark suite for all variants:

```sh
conda activate stzoo
PYTHONPATH=scripts python -m benchmark
```

Run nominal first, pause 5 seconds, then overdrive before going to bed:

```sh
conda activate stzoo
python project.py benchmark
```

---

## Compare README metrics to benchmark CSVs

**Conda environment:** `stzoo` — same as on-device benchmarking (`conda-benchmark` installs `markdown`, `beautifulsoup4`, and other deps used by the parser).

`python project.py compare` runs [`scripts/run_compare.py`](scripts/run_compare.py), which either parses STM32 model zoo README tables into a CSV, prints a delta report against `benchmark_results.csv`, or both.

### README source (`external/stm32ai-modelzoo/object_detection`)

The **reference** side of a readme-vs-benchmark comparison is not a single file. Metrics are scraped from HTML tables inside each model family’s `README.md` under the STM32 model zoo object-detection tree:

```text
external/stm32ai-modelzoo/object_detection/
```

Examples of those README paths include `st_yoloxn/README.md`, `yolov8n/README.md`, and `ssdlite_mobilenetv2_pt/README.md`. Which `README.md` applies to each onboard variant (and how rows are matched) is defined in [`scripts/benchmark/constants.py`](scripts/benchmark/constants.py) via `MODEL_REGISTRY` and `MODELZOO_DIR`.

**Official benchmark conditions (README tables):** ST publishes those NPU latency and footprint numbers for **overdrive mode**, under the **default STM32Cube.AI configuration** (including enabled input/output allocation), as stated in each family README under *Performances → Metrics*.

That directory comes from the **`stm32ai-modelzoo`** Git submodule. If it is missing or empty, run `git submodule update --init --recursive` from the repository root (see [Getting Started](#getting-started)).

### Subcommands

| Subcommand | Purpose |
|------------|---------|
| *(none)* | Same as `all` — parse README metrics, then compare parsed CSV to overdrive results. |
| `parse` | Write `results/benchmark_parsed.csv` from model zoo README tables only (`--out` to override path). |
| `compare` | Delta tables only — requires CSVs already on disk (see flags below). |
| `all` | Parse then compare README vs **overdrive** `benchmark_results.csv` (delta = measured − readme). |

If the first argument looks like a `compare`-only flag (`--mode`, `--benchmark`, `--nominal`, `--min-abs-delta-pct`), the runner inserts the `compare` subcommand automatically (e.g. `python project.py compare --mode readme-nominal`).

### Default files

| Artifact | Default path |
|----------|----------------|
| Parsed README metrics | `results/benchmark_parsed.csv` |
| Nominal device results | `results/benchmark_nominal/benchmark_results.csv` |
| Overdrive device results | `results/benchmark_overdrive/benchmark_results.csv` |

### Examples

Refresh parsed CSV and print README vs overdrive (default workflow):

```sh
conda activate stzoo
python project.py compare
```

Parse only:

```sh
conda activate stzoo
python project.py compare parse
```

Compare existing CSVs — modes:

- `readme-overdrive` — parsed README vs overdrive measured (default).
- `readme-nominal` — parsed README vs nominal measured.
- `nominal-overdrive` — nominal vs overdrive `benchmark_results.csv` only (Δ = overdrive − nominal).

```sh
conda activate stzoo
python project.py compare compare --mode readme-nominal
python project.py compare compare --mode nominal-overdrive
```

Optional: `--parsed`, `--nominal`, `--overdrive`, `--benchmark` (alias for the measured file in readme-* modes), and `--min-abs-delta-pct PCT` to hide metric rows with \|Δ%\| below `PCT`. Full flag list: `python project.py compare compare --help`.

Output is plain-text tables grouped by `model_variant` (no pass/fail); see the module docstring in [`scripts/benchmark/compare.py`](scripts/benchmark/compare.py) for column semantics and edge cases.