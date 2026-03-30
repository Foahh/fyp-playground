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

### Repository layout

- [`src/dataset`](src/dataset) — COCO and finetune dataset prep
- [`src/conda`](src/conda) — Conda environment setup
- [`src/ml`](src/ml) — TinyissimoYOLO training, INT8 TFLite quantization, Model Zoo finetune runners
- [`src/benchmark`](src/benchmark) — on-device benchmark and README compare tooling
- [`configs/`](configs) — YAML configs (Model Zoo finetuning Hydra files, Tinyissimo export, etc.)
- [`requirements-ml.txt`](requirements-ml.txt), [`requirements-bhmk.txt`](requirements-bhmk.txt) — extra pip constraints at the repo root

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
| Dataset download/prep (`src/dataset/load_coco.py`, `src/dataset/load_finetune_data.py`), TinyissimoYOLO training (`src/ml/run_train_tinyissimo_coco_person.py`), INT8 TFLite quantization (`src/ml/run_quantize.py`) | `fyp-ml` (`ST_YOLO_ENV`) | `python project.py conda-ml` |
| On-device benchmark, Model Zoo finetune helpers | `fyp-bhmk` (`ST_STZOO_ENV`) | `python project.py conda-bhmk` |

Before each step, run `conda activate <env>` for the row that matches the command.

---

## Unified Command Runner

Use `project.py` as the single entry point for common workflows.

**Required conda env** for each command (after `conda activate …`):

| Command | Conda env |
|---------|-----------|
| `dataset-coco`, `dataset-finetune`, `train`, `quant` | `fyp-ml` |
| `benchmark`, `compare`, `finetune-dataset`, `finetune` | `fyp-bhmk` |
| `conda-ml`, `conda-bhmk` | none (run from base or any env with `conda` available) |

### Local commands

**Env: `fyp-ml`** — COCO download:

```sh
conda activate fyp-ml
python project.py dataset-coco
```

**Env: `fyp-bhmk`** — hardware benchmark:

```sh
conda activate fyp-bhmk
python project.py benchmark --filter st_yoloxn_d033_w025_192
```

### Workflow commands

**Env: `fyp-ml`** — training:

```sh
conda activate fyp-ml
python project.py train --size 192
```

**Env: `fyp-ml`** — INT8 TFLite quantization (after training):

```sh
conda activate fyp-ml
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

**Conda environment:** `fyp-ml` — same as training (`python project.py conda-ml`, then `conda activate fyp-ml`).

Prepare datasets (default: `./datasets` under project root):

```sh
conda activate fyp-ml
mkdir -p ./datasets
python project.py dataset-coco
```

To store datasets elsewhere, set `DATASETS_DIR`:

```sh
conda activate fyp-ml
DATASETS_DIR=~/datasets python project.py dataset-coco
```

---

## Train TinyissimoYOLO

### Requirements

Training requires:

- **Python 3.12** (provided by the `fyp-ml` conda env from `conda-ml`)
- additional Python packages

For complete setup instructions, see:

- [external/TinyissimoYOLO/tinyissimoYOLO_README.md](external/TinyissimoYOLO/tinyissimoYOLO_README.md)

### Conda setup

**Conda environment:** `fyp-ml` — set it up before training:

```sh
python project.py conda-ml
conda activate fyp-ml
```

`conda-ml` already installs PyTorch from the CUDA 12.8 wheel index (RTX 50-series). To reinstall or upgrade manually:

```sh
conda activate fyp-ml
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Run training

**Conda environment:** `fyp-ml`.

From the repository root, outputs are written to:

```text
results/model/
```

Run training for different input sizes (checkpoints only; quantize to TFLite separately):

```sh
conda activate fyp-ml
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

**Conda environment:** `fyp-ml` (same as training; Ultralytics `model.export(format="tflite", int8=True, …)` via ONNX → onnx2tf).

### Export INT8 TFLite from `best.pt`

```sh
conda activate fyp-ml
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
conda activate fyp-ml
python project.py quant --size 192 --no-eval
python project.py quant --size 192 --checkpoint /path/to/best.pt
```

To evaluate with the STM32 model zoo host pipeline instead, use `configs/tinyissimoyolo_v8_192_config.yaml` (update `model.model_path` if your checkpoint stem or export location differs) and run `stm32ai_main.py` from the `fyp-bhmk` environment as documented in `external/stm32ai-modelzoo-services`.

---

## STM32 Model Zoo finetune

Starter configs for ST Model Zoo object-detection finetuning live under [`configs/`](configs):

- `configs/finetune_dataset.yaml` — dataset conversion and TFS preparation (Hydra)
- `configs/finetune.yaml` — training / chained quantization+export modes for `stm32ai_main.py`

### 1) Prepare dataset

Run converter + TFS creation:

```bash
python project.py finetune-dataset -- --config configs/finetune_dataset.yaml
```

Skip format conversion and only regenerate `.tfs` files:

```bash
python project.py finetune-dataset -- --config configs/finetune_dataset.yaml --skip-convert
```

Run optional analysis after prep:

```bash
python project.py finetune-dataset -- --config configs/finetune_dataset.yaml --analyze
```

Pass Hydra overrides:

```bash
python project.py finetune-dataset -- --config configs/finetune_dataset.yaml --override hydra.run.dir=./configs/outputs/dataset/debug
```

### 2) Finetune model

Use mode from YAML:

```bash
python project.py finetune -- --config configs/finetune.yaml
```

Override operation mode from CLI:

```bash
python project.py finetune -- --config configs/finetune.yaml --mode training
python project.py finetune -- --config configs/finetune.yaml --mode chain_tqe
python project.py finetune -- --config configs/finetune.yaml --mode chain_tqeb
```

Add extra Hydra overrides:

```bash
python project.py finetune -- --config configs/finetune.yaml --override training.epochs=80 --override training.batch_size=16
```

### Notes

- Prepare data to TFS before finetuning (`dataset.format: tfs` in `configs/finetune.yaml`).
- Paths in YAML are relative to the repository root when launched via `project.py`.
- Edit class taxonomy and source paths first (`class_names`, train/val/test directories). Hydra run and artifact dirs default under `./configs/outputs/…`.

---

## Benchmark on STM32N6570-DK

**Conda environment:** `fyp-bhmk` — create with `python project.py conda-bhmk` (quantization uses the `fyp-ml` env above).

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
python project.py conda-bhmk
conda activate fyp-bhmk
```

To use a different environment name instead of the default (`fyp-bhmk`):

```sh
ST_STZOO_ENV=my-benchmark-env python project.py conda-bhmk
conda activate my-benchmark-env
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

**Conda environment:** `fyp-bhmk`.

Test a single model first:

```sh
conda activate fyp-bhmk
PYTHONPATH=src python -m benchmark --filter st_yoloxn_d033_w025_192
```

Run the full benchmark suite for all variants:

```sh
conda activate fyp-bhmk
PYTHONPATH=src python -m benchmark
```

Run nominal first, pause 5 seconds, then overdrive before going to bed:

```sh
conda activate fyp-bhmk
python project.py benchmark
```

---

## Compare README metrics to benchmark CSVs

**Conda environment:** `fyp-bhmk` — same as on-device benchmarking (`conda-bhmk` installs `markdown`, `beautifulsoup4`, and other deps used by the parser).

`python project.py compare` runs [`src/benchmark/run_compare.py`](src/benchmark/run_compare.py), which either parses STM32 model zoo README tables into a CSV, prints a delta report against `benchmark_results.csv`, or both.

### README source (`external/stm32ai-modelzoo/object_detection`)

The **reference** side of a readme-vs-benchmark comparison is not a single file. Metrics are scraped from HTML tables inside each model family’s `README.md` under the STM32 model zoo object-detection tree:

```text
external/stm32ai-modelzoo/object_detection/
```

Examples of those README paths include `st_yoloxn/README.md`, `yolov8n/README.md`, and `ssdlite_mobilenetv2_pt/README.md`. Which `README.md` applies to each onboard variant (and how rows are matched) is defined in [`src/benchmark/constants.py`](src/benchmark/constants.py) via `MODEL_REGISTRY` and `MODELZOO_DIR`.

**Official benchmark conditions (README tables):** ST publishes those NPU latency and footprint numbers for **overdrive mode**, under the **default STM32Cube.AI configuration** (including enabled input/output allocation), as stated in each family README under *Performances → Metrics*.

That directory comes from the **`stm32ai-modelzoo`** Git submodule. If it is missing or empty, run `git submodule update --init --recursive` from the repository root (see [Getting Started](#getting-started)).

### Subcommands

| Subcommand | Purpose |
|------------|---------|
| *(none)* | Same as `all` — parse README metrics, then compare parsed CSV to overdrive results. |
| `parse` | Write `results/benchmark_parsed.csv` from model zoo README tables only (`--out` to override path). |
| `compare` | Delta tables only — requires CSVs already on disk (see flags below). |
| `all` | Parse then compare README vs **overdrive** `benchmark_results.csv` (delta = measured − readme). |

If the first argument looks like a `compare`-only flag (`--mode`, `--benchmark`, `--nominal`, `--delta-pct`), the runner inserts the `compare` subcommand automatically (e.g. `python project.py compare --mode readme-nominal`).

### Default files

| Artifact | Default path |
|----------|----------------|
| Parsed README metrics | `results/benchmark_parsed.csv` |
| Nominal device results | `results/benchmark_nominal/benchmark_results.csv` |
| Overdrive device results | `results/benchmark_overdrive/benchmark_results.csv` |

### Examples

Refresh parsed CSV and print README vs overdrive (default workflow):

```sh
conda activate fyp-bhmk
python project.py compare
```

Parse only:

```sh
conda activate fyp-bhmk
python project.py compare parse
```

Compare existing CSVs — modes:

- `readme-overdrive` — parsed README vs overdrive measured (default).
- `readme-nominal` — parsed README vs nominal measured.
- `nominal-overdrive` — nominal vs overdrive `benchmark_results.csv` only (Δ = overdrive − nominal).

```sh
conda activate fyp-bhmk
python project.py compare compare --mode readme-nominal
python project.py compare compare --mode nominal-overdrive
```

Optional: `--parsed`, `--nominal`, `--overdrive`, `--benchmark` (alias for the measured file in readme-* modes), and `--delta-pct PCT` to hide metric rows with \|Δ%\| below `PCT` (matching `stedgeai_version` rows are omitted). Full flag list: `python project.py compare compare --help`.

Output is plain-text tables grouped by `model_variant` (no pass/fail); see the module docstring in [`src/benchmark/compare.py`](src/benchmark/compare.py) for column semantics and edge cases.