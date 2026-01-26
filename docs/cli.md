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

## Visualização do ensemble (folds + WBF)

Para comparar rapidamente os `best.pth` de cada `fold_*` de um run (ex.: CV com 5 folds) e também ver um ensemble com *Weighted Boxes Fusion (WBF)*, use:

```bash
python scripts/ensemble_grid_test.py \
	--run-dir outputs/fasterrcnn_test/run_20260126_000613 \
	--images datasets/images/Validation/val \
	--out-dir outputs/ensemble_vis/run_20260126_000613 \
	--score-thresh 0.4 \
	--limit 6
```

Você também pode passar o `config.yaml` do projeto (para ler `model.name`, `model.num_classes` e `model.params`):

```bash
python scripts/ensemble_grid_test.py \
	--config config.yaml \
	--run-dir outputs/fasterrcnn_test/run_20260126_000613 \
	--images datasets/images/Validation/val \
	--out-dir outputs/ensemble_vis/run_20260126_000613 \
	--score-thresh 0.4 \
	--limit 6
```

O script gera:

- `grid.png`: um grid (colunas = folds + `ensemble_wbf`, linhas = imagens)
- `report.html`: página simples com o grid + visualização por imagem

### Ver na sua máquina local (você está em SSH)

Opção A (mais simples): copie para o seu computador com `scp`.

```bash
scp -r usuario@SERVIDOR:/CAMINHO/outputs/ensemble_vis/run_20260126_000613 ./
```

Opção B (sem copiar tudo): sirva via HTTP no servidor e faça *port-forward*.

No servidor (dentro do `out-dir`):

```bash
python -m http.server 8000
```

No seu computador local (em outro terminal):

```bash
ssh -L 8000:localhost:8000 usuario@SERVIDOR
```

Depois abra no browser local: `http://localhost:8000/report.html`.

## Ensemble em vídeo (MP4)

Se você quer ver o *vídeo em si* com as caixas desenhadas (colunas = folds + `ensemble_wbf`), gere um MP4:

```bash
python scripts/ensemble_video_grid.py \
	--run-dir outputs/fasterrcnn_test/run_20260126_000613 \
	--video /caminho/do/video.mp4 \
	--out outputs/ensemble_vis/video_grid.mp4 \
	--every 5 \
	--max-frames 300 \
	--out-fps 10
```

Rodando com `config.yaml`:

```bash
python scripts/ensemble_video_grid.py \
	--config config.yaml \
	--run-dir outputs/fasterrcnn_test/run_20260126_000613 \
	--video /caminho/do/video.mp4 \
	--out outputs/ensemble_vis/video_grid.mp4
```

### Modo GUI (rodar no seu PC)

Para abrir uma janela e ver o grid em tempo real (sem gerar MP4), rode localmente:

```bash
python scripts/ensemble_video_grid.py \
	--config config.yaml \
	--run-dir outputs/fasterrcnn_test/run_20260126_000613 \
	--video /caminho/do/video.mp4 \
	--gui \
	--no-mp4 \
	--score-thresh 0.85
```

Dica: no modo `--gui`, pressione `q` ou `Esc` para parar.

Para assistir no seu computador local:

- Copiando: `scp usuario@SERVIDOR:/CAMINHO/outputs/ensemble_vis/video_grid.mp4 ./`
- Ou via HTTP + port-forward (como acima) e abrir/baixar o `.mp4` pelo navegador.
