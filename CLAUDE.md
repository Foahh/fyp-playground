# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FYP Playground** is a research workspace for STM32 AI benchmarking and model training, specifically focused on TinyissimoYOLO object detection for embedded devices. The workflow consists of three main phases:

1. **Training**: Train TinyissimoYOLO v8 on COCO person dataset at multiple image sizes (192, 256, 288, 320)
2. **Export**: Convert trained PyTorch models to TensorFlow Lite INT8 format
3. **Benchmark**: Evaluate models on STM32N6570-DK boards and collect performance metrics (latency, memory, power consumption)

**STEdgeAI version**: 4.0

## Architecture & Dependencies

### External Submodules
The repo uses Git submodules for external dependencies:
- `external/TinyissimoYOLO` - Modified YOLOv8 fork for tiny models; training logic uses Ultralytics YOLO API
- `external/ultralytics` - Used by training scripts
- `external/stm32ai-modelzoo` - STM32 model zoo (not currently active in this config)
- `external/stm32ai-modelzoo-services` - Contains benchmarking dependencies and validation logic

Initialize with: `git submodule update --init --recursive`

### Python Environments
The project requires **three separate conda environments** due to version conflicts:

1. **Training** (`tinyissimo-train` or custom name)
   - Python 3.10
   - PyTorch + CUDA
   - TinyissimoYOLO requirements

2. **Export** (`yolo-export` or custom name)
   - Python 3.10
   - TFLite export tools
   - Patched Ultralytics (INT8 per-channel quantization)

3. **Benchmark** (`st_zoo` or custom name)
   - Python 3.12.9
   - CUDA 11.8 + cuDNN
   - stm32ai-modelzoo-services requirements

Environment setup scripts auto-create these if missing.

### Scripts Organization

```
scripts/
├── benchmark/          # Benchmarking logic (target: STM32N6570-DK)
│   ├── __main__.py     # Entry point for run_benchmark.py
│   ├── models.py       # Model discovery and loading
│   ├── parsing.py      # Metrics parsing from ST Edge AI output
│   ├── workflow.py     # Device communication & inference execution
│   ├── power_serial.py # INA228 power measurement via serial
│   ├── results.py      # CSV results handling
│   └── parse_modelzoo_readme.py  # README metrics extraction
└── conda/              # Environment setup utilities
    ├── conda_setup_common.py      # Shared setup logic
    └── patch_ultralytics_tflite_quant.py  # TFLite INT8 patch
```

## Common Commands

### Environment Setup
```bash
# From repo root; creates Python 3.10 training env
python3 conda_setup_train.py

# Export environment (Python 3.10)
python3 conda_setup_export.py

# Benchmark environment (Python 3.12.9); requires STEDGEAI_CORE_DIR
python3 conda_setup_benchmark.py

# Custom env names via environment variables:
TINYISSIMO_TRAIN_ENV=custom-train python3 conda_setup_train.py
ST_BENCHMARK_ENV=custom-bench python3 conda_setup_benchmark.py
```

### Training
```bash
# Activate training env and train single image size
conda activate tinyissimo-train  # or your custom env name
python train_coco_person.py --img_size 192

# Resume training from last checkpoint
python train_coco_person.py --img_size 192 --resume

# Train all supported sizes (run sequentially or in parallel)
for size in 192 256 288 320; do
  python train_coco_person.py --img_size $size
done
```

Results are saved to `external/TinyissimoYOLO/results/tinyissimoyolo_v8_<size>/`

### Export to TFLite INT8
```bash
# Activate export env
conda activate yolo-export  # or your custom env name

# Export single model (reads from external/TinyissimoYOLO/results/)
python export_tflite.py --img_size 192

# Export with custom checkpoint
python export_tflite.py --img_size 192 --weights external/TinyissimoYOLO/results/tinyissimoyolo_v8_192/weights/best.pt

# Export all variants
for size in 192 256 288 320; do
  python export_tflite.py --img_size $size
done
```

Exported models are copied to `results/model/tinyissimoyolo_v8_<size>/`

### Benchmarking
```bash
# Activate benchmark env (requires STEDGEAI_CORE_DIR)
conda activate st_zoo  # or your custom env name

# Test with single model
python run_benchmark.py --filter st_yoloxn_d033_w025_192

# Run full benchmark (all models)
python run_benchmark.py

# With power measurement (optional; requires power_serial enabled)
# Results append to results/benchmark/power-measure.csv
python run_benchmark.py
```

Results CSV: `results/benchmark/benchmark_results.csv`

## Key Files & Paths

| Path | Purpose |
|------|---------|
| `train_coco_person.py` | Training entry point (wrapper around Ultralytics YOLO) |
| `export_tflite.py` | TFLite export with INT8 quantization |
| `run_benchmark.py` | Benchmark entry point (thin shim to scripts/benchmark/__main__.py) |
| `conda_setup_*.py` | Environment creation scripts |
| `load_coco.py` | COCO dataset loader |
| `configs/` | Training/export configuration files |
| `external/TinyissimoYOLO/results/` | Training outputs (checkpoints, logs) |
| `results/model/` | Exported TFLite models |
| `results/benchmark/` | Benchmark CSV and logs |
| `power-measure/` | Power measurement sketch & ST Edge AI patch |
| `docs/` | Detailed setup guides (STM32N6, power measurement) |

## System Requirements

### Linux Setup
- Install **STEdgeAI 4.0** in user-owned location (e.g., `~/ST/STEdgeAI/4.0`)
- Add to `~/.bashrc`: `export STEDGEAI_CORE_DIR="$HOME/ST/STEdgeAI/4.0"`
- Add user to dialout group: `sudo usermod -aG dialout $USER` (requires logout/login)

### Benchmarking Prerequisites
- **STM32N6570-DK board** connected via USB
- **STM32CubeIDE** installed (C code compilation)
- `$STEDGEAI_CORE_DIR/scripts/N6_scripts/config.json` configured for your board setup

### Power Measurement
- Arduino IDE for flashing `external/fyp-power-measure/fyp-power-measure.ino`
- **ESP32-C6** with INA228 module connected to power supply rail
- Uses interrupt-driven edge detection + INA228 energy accumulator for accurate measurements
- Sends binary protobuf messages (PowerSample) at 921600 baud
- Apply one-file patch to ST Edge AI: `external/fyp-power-measure/patch/aiValidation_ATON_power_sync.inc.c`
- Auto-detects ESP32-C6 power monitor if `--power-serial` not specified (looks for Espressif VID 0x303A)
- Command-line flags: `--power-serial` (optional, auto-detects if omitted), `--power-baud` (default 921600), `--validation-count` (default 10)
- Requires: `pip install pyserial protobuf`
- See `external/fyp-power-measure/patch/power-measure-patch-stedge-ai.md` for full wiring & troubleshooting

## Data Setup

### COCO Dataset
```bash
mkdir -p ~/datasets
ln -s ~/datasets/ <project>/datasets
python load_coco.py  # Downloads COCO 2017 person annotations + splits
```

Expects dataset structure recognized by TinyissimoYOLO/Ultralytics (auto-handled by `load_coco.py`).

## Important Notes

- **Submodules are critical**: `TinyissimoYOLO` provides modified YOLO code; `ultralytics` is used for exports. Always init submodules.
- **Separate conda envs are required** due to Python version and dependency conflicts between training (3.10), export (3.10), and benchmarking (3.12).
- **Image sizes are model variants**: Training produces separate checkpoints for each size (192, 256, 288, 320); export and benchmark treat these as distinct model families.
- **Results organization**: Training outputs go to `external/TinyissimoYOLO/results/`; export copies them to `results/model/` before generating TFLite files; benchmark reads from `results/model/`.
- **Power measurement is single-file patch**: Only `aiValidation_ATON.c` needs modification in ST Edge AI; patch file is provided.
- **CSV output is append-only**: Benchmark can resume; completed entries are skipped using `variant` + `format` tuple as key.

## Debugging & Troubleshooting

- **Training fails**: Check Python 3.10 version, CUDA availability, dataset path in YAML
- **Export fails**: Verify checkpoint exists in `results/model/<name>/weights/best.pt`; check Ultralytics patch was applied
- **Benchmark connection fails**: Verify USB connection, dialout group membership, STEdgeAI config, board detection via STM32CubeIDE
- **Power measurement unreliable**: Check serial port permissions, Arduino sketch flashing, INA228 wiring; see `external/fyp-power-measure/patch/power-measure-patch-stedge-ai.md`

## Related Documentation

- `README.md` - High-level workflow & quick start
- `docs/stm32n6_getting_started.md` - STM32N6570-DK board setup & troubleshooting
- `external/fyp-power-measure/patch/power-measure-patch-stedge-ai.md` - Power measurement wiring, config, debugging
- `docs/ai_runner.md` - ST Edge AI runner API (reference for benchmarking backend)
- `external/TinyissimoYOLO/tinyissimoYOLO_README.md` - TinyissimoYOLO training details
- `external/stm32ai-modelzoo-services/README.md` - Benchmarking service setup
