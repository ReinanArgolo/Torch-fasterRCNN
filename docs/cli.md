# Linha de Comando e `config.yaml`

Argumentos principais de `train.py`:

- `--config` (obrigatório): caminho para o arquivo YAML de configuração, ex.: `config.yaml`.
- `--project-root`: sobrescreve `project_root` do YAML (útil para runs locais/relativos).
- `--resume`: caminho para checkpoint `.pth` para retomar treino.
- `--epochs`, `--batch-size`, `--lr`, `--num-workers`, `--seed`: sobrescrevem valores do YAML.
- `--eval-map-every`: avalia COCO mAP a cada N épocas (0 para desabilitar).

Exemplo de execução:

```bash
python train.py --config config.yaml --project-root . --epochs 6 --batch-size 2
```

Estrutura do `config.yaml` (exemplo resumido):

- `project_root`: pasta base para resolver caminhos.
- `data.images.train_dir`, `data.images.val_dir`: diretórios de imagens.
- `data.annotations.train_json`, `data.annotations.val_json`: JSONs COCO.
- `training`: parâmetros (`epochs`, `batch_size`, `lr`, `num_workers`, `amp`, `eval_map_every`, `transforms`).
- `model`: `name`, `pretrained`, `num_classes`.
- `output`: `dir`, `experiment`, `save_every`, `resume`.
- `early_stopping`: configurações descritas em `training.md`.
- `cross_validation`: `enabled`, `folds`.

Dica: use valores pequenos para debug (poucas épocas e `num_workers=0`).
