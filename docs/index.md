# Documentação do Pipeline de Treinamento - Faster R-CNN

Bem-vindo à documentação do pipeline de treinamento Faster R-CNN presente neste repositório.

Objetivo:
- Explicar a arquitetura do código, como configurar o ambiente, executar treinamento e avaliação.
- Fornecer instruções para exportar a documentação para outros formatos (HTML / PDF).

Conteúdo desta pasta:
- `setup.md` — requisitos e instalação
- `code_structure.md` — visão geral dos arquivos e responsabilidades
- `training.md` — fluxo de treino, checkpoints e early stopping
- `cli.md` — argumentos de linha de comando e `config.yaml`
- `exporting.md` — como exportar os arquivos Markdown para HTML/PDF
 - `methodology.md` — metodologia geral e boas práticas
 - `augmentation_cv_report.md` — relatório científico sobre Data Augmentation e K-fold CV

Comandos rápidos

Executar um treino básico:

```bash
python train.py --config config.yaml
```

Exemplo com override de parâmetros:

```bash
python train.py --config config.yaml --epochs 10 --batch-size 4 --lr 0.002
```

Para instruções detalhadas, abra os tópicos abaixo.
