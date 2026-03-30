# FYP Playground

Workspace for FYP research, TinyissimoYOLO training, INT8 TFLite quantization, STM32 benchmarking, and STM32 Model Zoo finetuning.

**STEdgeAI version:** `4.0`

---

## Overview

This repository provides a unified workflow to:

1. initialize the workspace
2. prepare datasets
3. train TinyissimoYOLO
4. quantize to INT8 TFLite
5. benchmark on STM32 hardware
6. finetune STM32 Model Zoo models

Use `project.py` as the main entry point for most tasks.

### Repository layout

- [`src/dataset`](src/dataset) — COCO and finetune dataset prep
- [`src/conda`](src/conda) — Conda environment setup
- [`src/ml`](src/ml) — training, INT8 TFLite quantization, Model Zoo finetune runners
- [`src/benchmark`](src/benchmark) — on-device benchmark and README compare tooling
- [`configs/`](configs) — YAML configs for export and finetuning
- [`requirements-ml.txt`](requirements-ml.txt), [`requirements-bhmk.txt`](requirements-bhmk.txt) — extra pip constraints

---

## Quick start

```sh
git submodule update --init --recursive

python project.py setup-conda-ml
python project.py setup-conda-bhmk

conda activate fyp-ml
python project.py download-coco
python project.py train --size 192
python project.py quantize --size 192

conda activate fyp-bhmk
python project.py benchmark
```

---

## Workflow

- [1. Initialize](#1-initialize)
- [2. Train](#2-train)
- [3. Quantize](#3-quantize)
- [4. Benchmark](#4-benchmark)
- [5. Finetune](#5-finetune)

---

## 1. Initialize

### Clone submodules

```sh
git submodule update --init --recursive
```

### Linux setup

Install **STEdgeAI** in a user-owned location such as `~/ST/STEdgeAI`, then add this to `~/.bashrc`:

```sh
export STEDGEAI_CORE_DIR="$HOME/ST/STEdgeAI/4.0"
```

Add your user to `dialout`:

```sh
sudo usermod -aG dialout $USER
```

Log out and back in after changing groups.

### Conda environments

Two Conda environments are used by default:

| Purpose | Default env | Setup |
|---|---|---|
| Dataset prep, TinyissimoYOLO training, INT8 quantization | `fyp-ml` (`ST_YOLO_ENV`) | `python project.py setup-conda-ml` |
| Benchmarking, README comparison, Model Zoo finetuning | `fyp-bhmk` (`ST_STZOO_ENV`) | `python project.py setup-conda-bhmk` |

Command mapping:

| Command | Env |
|---|---|
| `download-coco`, `download-finetune`, `train`, `quantize` | `fyp-ml` |
| `benchmark`, `compare-runs`, `prepare-finetune-dataset`, `finetune`, `verify-model-dtypes`, `parse-modelzoo-readme` | `fyp-bhmk` |
| `setup-conda-ml`, `setup-conda-bhmk` | base or any env with `conda` |

Create both environments:

```sh
python project.py setup-conda-ml
python project.py setup-conda-bhmk
```

### Prepare datasets

Activate `fyp-ml`:

```sh
conda activate fyp-ml
```

Default dataset location is `./datasets`:

```sh
mkdir -p ./datasets
python project.py download-coco
```

To use a different dataset directory:

```sh
DATASETS_DIR=~/datasets python project.py download-coco
```

---

## 2. Train

Train TinyissimoYOLO in `fyp-ml`.

Activate `fyp-ml`:

```sh
conda activate fyp-ml
```

`setup-conda-ml` already installs PyTorch from the CUDA 12.8 wheel index. To reinstall manually:

```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For full training details, see:

- [external/TinyissimoYOLO/tinyissimoYOLO_README.md](external/TinyissimoYOLO/tinyissimoYOLO_README.md)

Outputs are written under:

```text
results/model/
```

Training examples:

- `python project.py train --size 192`
- `python project.py train --size 256`
- `python project.py train --size 288`
- `python project.py train --size 320`

Checkpoint path:

```text
results/model/tinyissimoyolo_v8_<size>/weights/best.pt
```

---

## 3. Quantize

Quantize a trained checkpoint to **INT8 TFLite** in `fyp-ml`.

Activate `fyp-ml`:

```sh
conda activate fyp-ml
```

Main command:

- `python project.py quantize --size 192`

Optional examples:

- `python project.py quantize --size 192 --no-eval`
- `python project.py quantize --size 192 --checkpoint /path/to/best.pt`

Default output:

```text
results/model/tinyissimoyolo_v8_<size>/weights/best_saved_model/best_int8.tflite
```

Validation output:

```text
results/model/tinyissimoyolo_v8_<size>/weights/best_saved_model/val_int8/
```

### Notes

- Export uses Ultralytics INT8 TFLite flow.
- The script prints the final artifact path.
- For STM32 Model Zoo host-side evaluation, use `configs/tinyissimoyolo_v8_192_config.yaml` and run `stm32ai_main.py` from `fyp-bhmk`.

---

## 4. Benchmark

Benchmark on **STM32N6570-DK** in `fyp-bhmk`.

Results are saved to:

```text
results/benchmark_nominal/benchmark_results.csv
```

### Requirements

Before benchmarking, ensure:

- STM32CubeCLT is installed and `arm-none-eabi-*` is in `PATH`
- `STEDGEAI_CORE_DIR` is set
- `$STEDGEAI_CORE_DIR/scripts/N6_scripts/config.json` contains:

```json
{
  "compiler_type": "gcc"
}
```

Hardware:

- STM32N6570-DK board connected via USB
- optional: ESP32 + INA228 for power logging

For setup details, see:

- [docs/stm32n6_getting_started.md](docs/stm32n6_getting_started.md)
- [external/stm32ai-modelzoo-services/README.md](external/stm32ai-modelzoo-services/README.md#before-you-start)

Activate `fyp-bhmk`:

```sh
conda activate fyp-bhmk
```

Benchmark examples:

- Single model: `python project.py benchmark --filter st_yoloxn_d033_w025_192`
- Nominal mode only: `python project.py benchmark --mode nominal`
- Overdrive mode only: `python project.py benchmark --mode overdrive`
- Both modes (default): `python project.py benchmark`

### Optional power measurement

For INA228-based logging, see:

- [external/fyp-power-measure/README.md](external/fyp-power-measure/README.md)

Benchmark averages are written to `pm_avg_*` columns in:

```text
results/benchmark_nominal/benchmark_results.csv
```

Continuous logs are appended to:

```text
results/benchmark_nominal/power_measure.csv
```

### Compare README metrics

Use `project.py compare-runs` in `fyp-bhmk` to parse STM32 Model Zoo README tables and compare them against measured results.

Reference README files are scraped from:

```text
external/stm32ai-modelzoo/object_detection/
```

Default files:

| Artifact | Path |
|---|---|
| Parsed README metrics | `results/benchmark_parsed.csv` |
| Nominal results | `results/benchmark_nominal/benchmark_results.csv` |
| Overdrive results | `results/benchmark_overdrive/benchmark_results.csv` |

Examples:

- Refresh parsed CSV and compare to overdrive: `python project.py compare-runs`
- Parse only: `python project.py compare-runs parse`
- Compare README vs nominal: `python project.py compare-runs compare --mode readme-nominal`
- Compare nominal vs overdrive: `python project.py compare-runs compare --mode nominal-overdrive`

Useful flags:

- `--parsed`
- `--nominal`
- `--overdrive`
- `--benchmark`
- `--delta-pct PCT`

Full help:

```sh
python project.py compare-runs compare --help
```

---

## 5. Finetune

Use the STM32 Model Zoo finetune pipeline in `fyp-bhmk`.

Configs:

- `configs/finetune_dataset.yaml` — dataset conversion and TFS preparation
- `configs/finetune.yaml` — training and chained quantization/export modes

Activate `fyp-bhmk`:

```sh
conda activate fyp-bhmk
```

### Prepare finetune dataset

Examples:

- Default: `python project.py prepare-finetune-dataset -- --config configs/finetune_dataset.yaml`
- Skip format conversion: `python project.py prepare-finetune-dataset -- --config configs/finetune_dataset.yaml --skip-convert`
- Run analysis: `python project.py prepare-finetune-dataset -- --config configs/finetune_dataset.yaml --analyze`
- Pass Hydra override: `python project.py prepare-finetune-dataset -- --config configs/finetune_dataset.yaml --override hydra.run.dir=./configs/outputs/dataset/debug`

### Run finetuning

Examples:

- Use mode from YAML: `python project.py finetune -- --config configs/finetune.yaml`
- Training mode: `python project.py finetune -- --config configs/finetune.yaml --mode training`
- Chain TQE mode: `python project.py finetune -- --config configs/finetune.yaml --mode chain_tqe`
- Chain TQEB mode: `python project.py finetune -- --config configs/finetune.yaml --mode chain_tqeb`
- With Hydra overrides: `python project.py finetune -- --config configs/finetune.yaml --override training.epochs=80 --override training.batch_size=16`

### Notes

- Prepare TFS data before finetuning.
- YAML paths are relative to the repository root when launched via `project.py`.
- Update class names and dataset paths before training.
- Hydra outputs default under `./configs/outputs/`.