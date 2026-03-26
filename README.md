# FYP Playground

A workspace for FYP research and experimentation.

**STEdgeAI version:** 4.0

## Getting Started

Before running any project components, initialize the Git submodules:

```sh
git submodule update --init --recursive
```

It is recommended to use **Docker** for environment management.

## Linux Setup

On Linux, install **STEdgeAI** in a user-owned location such as `~/ST/STEdgeAI` to avoid permission issues.

After installing STEdgeAI, add the following to your `~/.bashrc`:

```sh
export STEDGEAI_CORE_DIR="$HOME/ST/STEdgeAI/4.0"
```

Also add your user to the `dialout` group:

```sh
sudo usermod -aG dialout $USER
```

You may need to log out and log back in for this change to take effect.

## Dataset Setup

Create a dataset directory and link it into the project:

```sh
mkdir -p ~/datasets
ln -s ~/datasets/ <project>
python ./load_coco.py
```

## Train TinyissimoYOLO

### Requirements

Training requires **Python 3.10** and several Python packages.

For full setup instructions, see [external/TinyissimoYOLO/tinyissimoYOLO_README.md](external/TinyissimoYOLO/tinyissimoYOLO_README.md).

### Docker Setup

```sh
docker compose build train
```

### Training

From the repository root (outputs go to `external/TinyissimoYOLO/results/`):

```sh
docker compose run --rm train train_coco_person.py --img_size 192
docker compose run --rm train train_coco_person.py --img_size 256
docker compose run --rm train train_coco_person.py --img_size 288
docker compose run --rm train train_coco_person.py --img_size 320
```

---

## Export TinyissimoYOLO to TFLite INT8

### Docker Setup

```sh
docker compose build export
```

### Export

```sh
docker compose run --rm export run_export.py --img_size 192
docker compose run --rm export run_export.py --img_size 256
docker compose run --rm export run_export.py --img_size 288
docker compose run --rm export run_export.py --img_size 320
```

Quantize SavedModel to TFLite INT8:

```sh
docker compose run --rm export run_quantize.py \
  --img_size 192 \
  --saved-model-dir results/model/tinyissimoyolo_v8_192/weights/best_saved_model
```

By default, export reads checkpoints from `results/model/tinyissimoyolo_v8_<img_size>/weights/best.pt`.

You can also export a specific checkpoint:

```sh
docker compose run --rm export run_export.py --img_size 192 --weights results/model/tinyissimoyolo_v8_192/weights/best.pt
```

---

## Benchmark on STM32N6570-DK

This benchmark performs on-device evaluation for all supported model variants and saves the results to `results/benchmark/benchmark_results.csv`.

### Reading

For more details, see [docs/stm32n6_getting_started.md](docs/stm32n6_getting_started.md).

Before benchmarking, make sure:

- **STM32CubeIDE** is installed
- `$STEDGEAI_CORE_DIR/scripts/N6_scripts/config.json` is properly configured

### Requirements

Benchmarking requires **Python 3.12.9** and several additional packages.

For full setup instructions, see [external/stm32ai-modelzoo-services/README.md](external/stm32ai-modelzoo-services/README.md#before-you-start).

### Conda Environment Setup

```sh
python3 conda_setup_benchmark.py
```

Optional: use a different env name (default is `fyp`):

```sh
ST_BENCHMARK_ENV=my-benchmark-env python3 conda_setup_benchmark.py
```

### Prerequisites

- STM32N6570-DK board connected via USB
- `STEDGEAI_CORE_DIR` environment variable set
- ESPS3-C3 connected with INA228
- Arduino IDE (flash `external/fyp-power-measure/fyp-power-measure.ino`)

#### Power measurement (`avg_power_mW`, optional)

For inference-window **`avg_power_mW`** in the benchmark CSV (INA228 + `external/fyp-power-measure/fyp-power-measure.ino`), apply a **single-file** patch to ST Edge AI:

1. Open `Middlewares/ST/AI/Validation/Src/aiValidation_ATON.c` in your ST install (`$STEDGEAI_CORE_DIR`).
2. Add `#include "stm32n6xx_hal.h"` once with the other includes if it is not there yet.
3. Paste the code block from [`external/fyp-power-measure/patch/aiValidation_ATON_power_sync.inc.c`](external/fyp-power-measure/patch/aiValidation_ATON_power_sync.inc.c) (not the comment header) **after** the `_dumpable_tensor_name[]` array and **before** `_APP_VERSION_MAJOR_`.
4. Add the **call sites** if your vendor file does not already include them (see [external/fyp-power-measure/patch/power-measure-patch-stedge-ai.md](external/fyp-power-measure/patch/power-measure-patch-stedge-ai.md)).

Full wiring, command-line flags (`--power-serial`, `--power-baud`, `--validation-count`), and troubleshooting: **[external/fyp-power-measure/patch/power-measure-patch-stedge-ai.md](external/fyp-power-measure/patch/power-measure-patch-stedge-ai.md)**. With power serial enabled, the run also appends a continuous log to **`results/benchmark/power-measure.csv`** (host timestamp + INA228 fields). The sketch waits for **`START`** on the serial line; the benchmark sends it when opening the port.

### Run Benchmark

Test with a single model first:

```sh
python run_benchmark.py --filter st_yoloxn_d033_w025_192
```

Run the full benchmark suite for all variants:

```sh
python run_benchmark.py
```
