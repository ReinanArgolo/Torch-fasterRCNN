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

### Redimensionamento (com ajuste de bbox)

No `config.yaml` você pode habilitar redimensionamento mantendo aspecto e ajustando as bounding boxes:

```
training:
	transforms:
		resize:
			min_size: 800   # lado menor
			max_size: 1333  # limite do lado maior
```

O mesmo transform é aplicado em treino e validação.

## Early stopping (anti-overfitting)

Configure in `config.yaml`:

```
early_stopping:
	enabled: true
	monitor: val_loss   # or "map"
	mode: min           # "min" for loss, "max" for mAP
	min_delta: 0.0
	patience: 3
```

If you monitor `map`, set `training.eval_map_every` to how often to compute COCO mAP during training (default 0 = off).

## Evaluate COCO mAP

```
python torch-pipeline/eval_coco.py --config torch-pipeline/config.yaml \
	--checkpoint outputs/<experiment>/epoch_10.pth --score-thresh 0.05
```

## Inference (predições)

```
python torch-pipeline/infer.py --images new_whales_rcnn/images/Test/test \
	--checkpoint outputs/<experiment>/epoch_10.pth \
	--num-classes 2 --score-thresh 0.5 --model fasterrcnn_resnet50_fpn_v2 \
	--out outputs/preds_test.json
```

Para redimensionar na inferência (com ajuste de bbox):

```
python torch-pipeline/infer.py --images new_whales_rcnn/images/Test/test \
	--checkpoint outputs/<experiment>/epoch_10.pth --num-classes 2 \
	--resize-min 800 --resize-max 1333 --out outputs/preds_resized.json
```

## Config notes

- `project_root`: Base path to resolve all relative paths.
- `data.images.train_dir` / `data.images.val_dir`: Folders with images.
- `data.annotations.train_json` / `data.annotations.val_json`: COCO jsons.
- `model.num_classes`: Include background (e.g., 2 for background + whale).

## Resume training

Pass `--resume /path/to/checkpoint.pth` or set `output.resume` in config.

## Inference

This repo focuses on training; add an `infer.py` later if needed to export detections.
Now included: `infer.py` and `eval_coco.py`.
