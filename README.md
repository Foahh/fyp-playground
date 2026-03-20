# FYP Playground

A workspace for FYP research and experimentation.

**STEdgeAI version:** 4.0

## Getting Started

Before running any project components, initialize the Git submodules:

```sh
git submodule update --init --recursive
```

It is recommended to use **Conda** for Python environment management.

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

For full setup instructions, see [TinyissimoYOLO/tinyissimoYOLO_README.md](TinyissimoYOLO/tinyissimoYOLO_README.md).

### Conda Environment Setup

```sh
cd TinyissimoYOLO

conda create -n yolo python=3.10
conda activate yolo

# For CUDA 11.8:
# pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.6:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# For CUDA 13.0:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt
```

### Training

Run training with different input sizes:

```sh
python train_coco_person.py --img_size 192
python train_coco_person.py --img_size 256
python train_coco_person.py --img_size 288
python train_coco_person.py --img_size 320
```

---

## Benchmark on STM32N6570-DK

This benchmark performs on-device evaluation for all supported model variants and saves the results to `benchmark_results.csv`.

### Reading

For more details, see [docs/stm32n6_getting_started.md](docs/stm32n6_getting_started.md).

Before benchmarking, make sure:

- **STM32CubeIDE** is installed
- `$STEDGEAI_CORE_DIR/scripts/N6_scripts/config.json` is properly configured

### Requirements

Benchmarking requires **Python 3.12.9** and several additional packages.

For full setup instructions, see [stm32ai-modelzoo-services/README.md](stm32ai-modelzoo-services/README.md#before-you-start).

### Conda Environment Setup

```sh
conda create -n st_zoo python=3.12.9
conda activate st_zoo
```

If you want to use **NVIDIA GPU support**, run:

```sh
conda install -c conda-forge cudatoolkit=11.8 cudnn
```

### Prerequisites

- STM32N6570-DK board connected via USB
- `STEDGEAI_CORE_DIR` environment variable set

### Run Benchmark

Test with a single model first:

```sh
python run_benchmark.py --filter st_yoloxn_d033_w025_192
```

Run the full benchmark suite for all variants:

```sh
python run_benchmark.py
```
