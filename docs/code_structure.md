# Estrutura do Código

Arquivos principais e responsabilidades:

- `train.py`: CLI principal que carrega `config.yaml`, prepara datasets, model, otimizador e instancia `Trainer`.
- `dataset.py`: implementação de `COCODetectionDataset` compatível com o formato COCO e `collate_fn`.
- `transforms.py`: constrói as transformações aplicadas a cada amostra (a abstração usada por `train.py`).
- `modules/model_factory.py`: função `get_model()` que cria modelos Faster R-CNN (vários nomes suportados) e substitui a cabeça quando necessário.
- `modules/trainer.py`: classe `Trainer` que implementa `fit()` com loop de treino/validação, checkpointing, early stopping e exportação de métricas.
- `modules/metrics.py`: utilitários para salvar/exportar métricas do treinamento.
- `coco_eval.py` e `eval_coco.py`: avaliação compatível COCO (mAP) usada pelo `Trainer` quando `eval_map_every > 0`.
- `utils.py`: funções utilitárias (checkpointing, scheduler/optimizer builders, AverageMeter, formatação de tempo, etc.).
- `infer.py` / `scripts/quick_infer.py`: utilitários para inferência e geração de predições.

Padrões e observações:
- O dataset mapeia arquivos de `file_name` no JSON COCO para um diretório de imagens, com heurísticas de resolução de nomes.
- `Trainer` salva checkpoints em `outputs/<experiment>/<run_stamp>/epoch_XXX/checkpoint_epoch_XXX.pth` e mantém um `best.pth` quando early-stopping estiver habilitado.
- Amostras visualizadas com bounding boxes são salvas em diretórios `epoch_XXX/samples/`.
