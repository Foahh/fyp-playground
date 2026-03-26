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

Using **Docker** and **Conda** is recommended for dependency and environment management.

---

## Unified Command Runner

Use `project.py` as the single entry point for common workflows.

### Local commands

```sh
python project.py coco
python project.py benchmark --filter st_yoloxn_d033_w025_192
```

### Docker-based commands

```sh
python project.py train --img_size 192
python project.py export --img_size 192
python project.py quantize --img_size 192
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

Create a dataset directory and symlink it into the project:

```sh
mkdir -p ~/datasets
ln -s ~/datasets/ <project>
python ./scripts/load_coco.py
```

> Replace `<project>` with the appropriate path inside this repository.

---

## Train TinyissimoYOLO

### Requirements

Training requires:

- **Python 3.10**
- additional Python packages

For complete setup instructions, see:

- [external/TinyissimoYOLO/tinyissimoYOLO_README.md](external/TinyissimoYOLO/tinyissimoYOLO_README.md)

### Build Docker image

```sh
docker compose build train
```

### Run training

From the repository root, outputs are written to:

```text
external/TinyissimoYOLO/results/
```

Run training for different image sizes:

```sh
python project.py train --img_size 192
python project.py train --img_size 256
python project.py train --img_size 288
python project.py train --img_size 320
```

---

## Export TinyissimoYOLO to TFLite INT8

### Build Docker images

```sh
docker compose build export
docker compose build quantize
```

### Export model

```sh
python project.py export --img_size 192
python project.py export --img_size 256
python project.py export --img_size 288
python project.py export --img_size 320
```

By default, export reads checkpoints from:

```text
results/model/tinyissimoyolo_v8_<img_size>/weights/best.pt
```

To export a specific checkpoint:

```sh
python project.py export \
  --img_size 192 \
  --weights results/model/tinyissimoyolo_v8_192/weights/best.pt
```

### Quantize SavedModel to TFLite INT8

```sh
python project.py quantize \
  --img_size 192
```

---

## Benchmark on STM32N6570-DK

This benchmark performs on-device evaluation for all supported model variants and saves results to:

```text
results/benchmark/benchmark_results.csv
```

### Additional reading

For more details, see:

- [docs/stm32n6_getting_started.md](docs/stm32n6_getting_started.md)

Before benchmarking, ensure:

- **STM32CubeIDE** is installed
- `$STEDGEAI_CORE_DIR/scripts/N6_scripts/config.json` is configured correctly

### Requirements

Benchmarking requires:

- **Python 3.12.9**
- additional Python packages

For full setup instructions, see:

- [external/stm32ai-modelzoo-services/README.md](external/stm32ai-modelzoo-services/README.md#before-you-start)

### Conda environment setup

```sh
python3 conda_setup_benchmark.py
```

To use a different environment name instead of the default (`fyp`):

```sh
ST_BENCHMARK_ENV=my-benchmark-env python3 conda_setup_benchmark.py
```

### Prerequisites

- STM32N6570-DK board connected via USB
- `STEDGEAI_CORE_DIR` environment variable set
- ESPS3-C3 connected with INA228
- Arduino IDE available to flash `external/fyp-power-measure/fyp-power-measure.ino`

### Optional power measurement (`avg_power_mW`)

To enable inference-window `avg_power_mW` logging (INA228 + ESP32-C6), follow:

- [external/fyp-power-measure/README.md](external/fyp-power-measure/README.md)

This guide covers wiring, Arduino sketch flashing, required ST Edge AI patching, CLI flags (`--power-serial`, `--power-baud`, `--validation-count`), and troubleshooting.

When power serial is enabled, the benchmark also appends a continuous log to:

```text
results/benchmark/power-measure.csv
```

This log contains host timestamps and INA228 fields. The Arduino sketch waits for `START` on the serial line, which the benchmark sends when opening the port.

### Run benchmark

Test a single model first:

```sh
python scripts/run_benchmark.py --filter st_yoloxn_d033_w025_192
```

Run the full benchmark suite for all variants:

```sh
python scripts/run_benchmark.py
```