# Torch Pipeline: Faster R-CNN (CLI)

A simple, reproducible CLI training pipeline for Faster R-CNN on COCO-style annotations using PyTorch and torchvision.

## Contents

- `train.py`: CLI to train and validate
- `dataset.py`: COCO dataset wrapper compatible with torchvision detection models
- `utils.py`: Helpers (metrics, checkpoints, seed, optim/scheduler)
- `config.example.yaml`: Example configuration file to customize paths, hyperparameters, and outputs
- `requirements.txt`: Python dependencies

## Quick start

1) Create a virtual environment (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r torch-pipeline/requirements.txt
```

3) Copy and edit the config:

```bash
cp torch-pipeline/config.example.yaml torch-pipeline/config.yaml
# edit paths if needed; by default it points to new_whales_rcnn
```

4) Run training:

```bash
python torch-pipeline/train.py --config torch-pipeline/config.yaml
```

Optional flags override config values:
- `--epochs` `--batch-size` `--lr` `--num-workers` `--seed` `--project-root` `--resume`

Checkpoints are saved under `outputs/<experiment>/epoch_*.pth`.

## Config notes

- `project_root`: Base path to resolve all relative paths.
- `data.images.train_dir` / `data.images.val_dir`: Folders with images.
- `data.annotations.train_json` / `data.annotations.val_json`: COCO jsons.
- `model.num_classes`: Include background (e.g., 2 for background + whale).

## Resume training

Pass `--resume /path/to/checkpoint.pth` or set `output.resume` in config.

## Inference

This repo focuses on training; add an `infer.py` later if needed to export detections.
