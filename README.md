# FYP Playground

The space for researching FYP.

## Quick Initialize

```sh
git submodule update --init --depth 1
```

## Train TinyissimoYOLO

### Install Dependencies

Requires Python 3.10 and several packages. See [TinyissimoYOLO/tinyissimoYOLO_README.md](TinyissimoYOLO/tinyissimoYOLO_README.md) for full setup details.

```sh
python train_coco_person.py --img_size 192
python train_coco_person.py --img_size 256
python train_coco_person.py --img_size 288
python train_coco_person.py --img_size 320
```

## Benchmark Object Detection Models on STM32N6570-DK

Runs on-device evaluation for all in-scope model variants and collects results into `benchmark_results.csv`.

### Install Dependencies

Requires Python 3.12.9 and several packages. See [stm32ai-modelzoo-services/README.md](stm32ai-modelzoo-services/README.md) for full setup details.

### Prerequisites

- STM32N6570-DK board connected via USB
- `STEDGEAI_CORE_DIR` environment variable set

### Run

```sh
# Test with a single model first
python benchmark_od.py --filter st_yoloxn_d033_w025_192

# Run the full benchmark (all 41 variants)
python benchmark_od.py
```
