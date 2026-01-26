# Fluxo de Treinamento (Resumo)

Esse documento fornece um resumo prático do fluxo de treinamento presente no repositório. Para uma discussão metodológica aprofundada — motivação, escolhas de design, dicas para estabilizar e otimizar o treino — veja `methodology.md` nesta mesma pasta.

Principais etapas do pipeline:

1) Preparação
- O script de entrada lê um arquivo YAML de configuração e permite overrides por CLI (`--epochs`, `--batch-size`, etc.).

2) Transformações e Augmentations
- As transformações de amostra (redimensionamento, flips, jitter de cor, crop baseado em IoU etc.) são aplicadas de modo a alterar também as caixas delimitadoras corretamente.

3) Dataset e DataLoader
- O dataset COCO resolve nomes de arquivo e retorna pares (imagem, target) prontos para os modelos do `torchvision`.
- O DataLoader usa um `collate_fn` que empacota lotes variáveis de anotações.

4) Modelo e Cabeça
- O backbone e as componentes do Faster R-CNN (RPN, FPN, RoI heads) são instanciados a partir de funções utilitárias; a cabeça de predição é substituída quando `num_classes` difere do padrão.

5) Otimização e Precisão Mista
- Otimizador, scheduler e (opcional) AMP são configurados pela `config.yaml`; o treinamento suporta `torch.amp` quando CUDA está disponível.

6) Loop de Treino e Validação
- A cada época ocorre treino por batches, update do scheduler, avaliação de perda em validação e (opcional) cálculo de COCO mAP.

7) Checkpointing e Early Stopping
- Checkpoints periódicos e um `best.pth` são salvos; early stopping pode monitorar `val_loss` ou `val_mAP` e restaurar pesos.

8) Cross-validation
- Há suporte a K-fold CV gerando subconjuntos COCO a partir do JSON de treino e executando experimentos separados por fold.

9) Artefatos
- As saídas principais incluem checkpoints, visualizações de amostra por época, e métricas exportadas (CSV/JSON).

Para uma visão metodológica detalhada (motivação por trás de cada escolha, boas práticas, armadilhas e estratégias de debug), abra `docs/methodology.md`.
