# Finetune Workflow

This folder contains starter configs for ST Model Zoo object-detection finetuning.

## Files

- `finetune_dataset.yaml`: dataset conversion/TFS preparation config
- `finetune.yaml`: training/chain mode config for `stm32ai_main.py`

## 1) Prepare Dataset

Run converter + TFS creation:

```bash
python project.py finetune-dataset -- --config finetune/finetune_dataset.yaml
```

Skip format conversion and only regenerate `.tfs` files:

```bash
python project.py finetune-dataset -- --config finetune/finetune_dataset.yaml --skip-convert
```

Run optional analysis after prep:

```bash
python project.py finetune-dataset -- --config finetune/finetune_dataset.yaml --analyze
```

Pass Hydra overrides:

```bash
python project.py finetune-dataset -- --config finetune/finetune_dataset.yaml --override hydra.run.dir=./finetune/outputs/dataset/debug
```

## 2) Finetune Model

Use mode from YAML:

```bash
python project.py finetune -- --config finetune/finetune.yaml
```

Override operation mode from CLI:

```bash
python project.py finetune -- --config finetune/finetune.yaml --mode training
python project.py finetune -- --config finetune/finetune.yaml --mode chain_tqe
python project.py finetune -- --config finetune/finetune.yaml --mode chain_tqeb
```

Add extra Hydra overrides:

```bash
python project.py finetune -- --config finetune/finetune.yaml --override training.epochs=80 --override training.batch_size=16
```

## Notes

- Dataset should be prepared to TFS before finetuning (`dataset.format: tfs` in `finetune.yaml`).
- Paths in YAML are relative to the repository root when launched via `project.py`.
- Edit class taxonomy and source paths first (`class_names`, train/val/test directories).
