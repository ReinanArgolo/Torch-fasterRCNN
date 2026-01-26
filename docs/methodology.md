# Metodologia de Treinamento — Detalhes e Boas Práticas

Este texto explica em profundidade as decisões e práticas recomendadas para treinar um detector Faster R-CNN em datasets no formato COCO. O objetivo é transmitir a "intuição" e os trade-offs por trás das escolhas de pipeline, sem entrar em trechos de código.

**1. Objetivo do Modelo e Métricas**
- Objetivo: detectar instâncias de objetos (caixas + classes) em imagens. Para detecção, a métrica padrão é o COCO mAP (mean Average Precision) avaliado em múltiplas IoU thresholds (AP@[.5:.95]) — ela reflete a qualidade das localizações e classificações.
- Use AP50 e AP75 para avaliar sensibilidade a localização: AP50 (IoU≥0.5) é permissiva; AP75 exige boxes mais precisas.

**2. Preparação do Dataset**
- Qualidade das anotações: caixas incorretas ou rótulos inconsistentes prejudicam fortemente o treino. Verifique outliers (boxes muito pequenas/grandes ou áreas negativas) e remova/ajuste.
- Balanceamento: classes raras podem demandar oversampling, augmentation direcionada ou loss re-weighting.
- Divisão treino/val: reserve um conjunto de validação representativo. Em imagens geograficamente estratificadas (ex.: câmeras, locais), mantenha divisão por grupo para evitar vazamento.

**3. Transformações e Data Augmentation**
- Propósito: aumentar variedade, reduzir overfitting, tornar o modelo robusto a diferentes condições.
- Tipos úteis:
  - Redimensionamento com conservação de aspecto (min_size / max_size) para manter escalas de objeto previsíveis.
  - Flips horizontais aleatórios para aumentar simetria.
  - Jitter de cor (brightness/contrast/saturation/hue) para invariância a iluminação.
  - Random crops e IoU-aware cropping: útil para forçar o detector aprender a localizar em partes menores; porém, cuidado para não cortar quase todas as anotações (desbalanceia).
  - Escalonamento aleatório e multi-scale training: treinar em múltiplas resoluções ajuda a detectar objetos em escalas variadas.
- Boas práticas: aplicar augmentations que preservem a semântica (não rotacionar em 90° se os objetos têm orientação importante) e garantir transformação consistente entre imagem e caixas.

**4. Arquitetura (Rápido Overview)**
- Faster R-CNN: backbone (ex.: ResNet) extrai features → FPN (Feature Pyramid Network) combina escalas → RPN (Region Proposal Network) gera propostas → RoI Pooling/Align + cabeças classificadora/regressora refinam e classificam boxes.
- Escolhas de backbone (ResNet50 vs ResNet101) implicam trade-off entre precisão e custo computacional.

**5. Losses e Estabilidade do Treino**
- Loss total combina múltiplos termos: loss do RPN (objectness + bbox regression) e loss das cabeças (classificação + bbox regression). O balanceamento desses termos afeta prioridades do modelo.
- Gradientes instáveis podem surgir com learning rates altos, batch-size pequeno ou exemplos com grande perda — use clipping de gradiente se necessário.

**6. Otimizador e Scheduler**
- Otimizadores comuns: SGD com momentum para estabilidade e boa generalização; AdamW pode convergir mais rápido, mas ocasionalmente generaliza pior em detecção.
- Weight decay ajuda a regularizar pesos convolucionais e fully-connected.
- Scheduler: StepLR/ReduceLROnPlateau/Cosine annealing. Uma estratégia prática é reduzir learning rate em plateaus de validação ou usar milestones fixos após observar quando a curva de validação estagna.

**7. Batch Size, Normalização e BatchNorm**
- Detecção de objetos frequentemente usa batch-sizes pequenos por GPU (1–4) devido ao custo de memória. Isso afeta camadas BatchNorm — considere usar SyncBatchNorm ou congelar BatchNorm durante fine-tuning (usar estatísticas pré-treinadas) se o batch for muito pequeno.

**8. Treinamento em Precisão Mista (AMP)**
- Benefícios: reduz uso de VRAM e acelera FP16-capable GPUs. Atenção a operações que não são estáveis em FP16 (algumas versões de ROIAlign/ops customizados). Usar `GradScaler` para escala automática do gradiente.

**9. Checkpointing e Recuperação**
- Salve checkpoints periódicos e mantenha um `best` (por exemplo, melhor mAP ou menor val_loss). Salvar também o estado do otimizador e scheduler permite retomar com comportamento idêntico.

**10. Early Stopping — Estratégia**
- Early stopping reduz overfitting e economiza GPU time. Configure `patience` e `min_delta` baseados na variabilidade da métrica: datasets pequenos exigem mais cuidado com ruído.
- Monitorar `val_mAP` é preferível quando mAP estável; `val_loss` é mais ruidoso e pode não refletir qualidade de bounding boxes.

**11. Cross-Validation**
- K-fold CV oferece estimativa robusta de variância do modelo e ajuda quando os dados são limitados. Ao gerar folds em COCO, mantenha distribuição de classes e verifique que não exista sobreposição de imagens (ou grupos correlacionados) entre folds.

**12. Avaliação (COCO mAP)**
- AP@[.5:.95] é a média de APs em 10 IoU thresholds; fornece visão global.
- Para problemas onde localização precisa é crítica, reporte AP75; para detecções gerais, AP50 costuma ser mais estável.

**13. Visualização e Debugging**
- Salvar imagens com boxes e scores ajuda a identificar:
  - problemas de escala (boxes muito grandes/pequenas),
  - classes confundidas,
  - problemas de anotação (caixas fora do objeto).
- Verifique histogramas de perdas por lote para detectar batches com perda muito alta (outliers).

**14. Heurísticas de Hyperparameter Tuning**
- Learning rate: experimente escalonar por regra linear com batch size (se aumentar batch, aumente lr proporcionalmente até certo limite).
- Weight decay: valores entre 1e-3 e 1e-5; filtre por validação.
- Number of proposals (RPN): reduzir para acelerar inferência, mas pode reduzir recall.

**15. Inferência e Pós-processamento**
- Non-maximum suppression (NMS) e thresholds de score governam número de boxes finais. Ajuste `score_thresh` para equilibrar precisão/recall.
- Para deployment em edge devices, quantize ou exporte para ONNX/TorchScript e reduza backbone.

**16. Considerações Computacionais**
- Monitore uso de VRAM e tempo por batch; ajuste `num_workers`, `pin_memory` e `prefetch_factor` no DataLoader para throughput.

**17. Reprodutibilidade**
- Defina sementes globais (PyTorch, NumPy, Python random) e anote versões de dependências. Determinismo completo pode degradar desempenho — documente as decisões.

**18. Checklist de Debug Rápido**
- Verifique anotações COCO (formatos e ids).  
- Visualize amostras com bboxes.  
- Teste forward pass com um batch pequeno para garantir shapes.  
- Tente reduzir lr se perdas explode.  

**19. Experimentos Recomendados**
- Baseline: backbone ResNet50, lr=0.005, batch=2, epochs=12, StepLR (milestone=8,11).  
- Experimento AMP vs FP32 para medir ganho de throughput.  
- Curva de learning rate (1–5 rodadas) para encontrar ponto de quebra.

Referências para aprofundar:
- Ren et al., "Faster R-CNN" (2015) — arquitetura base.
- Lin et al., "Feature Pyramid Networks" (2016) — multicamada de features para detecção em escalas.
- COCO evaluation toolkit — entender métricas padrão.

---

Se quiser, posso transformar seções específicas em checklists operacionais (por exemplo: "Antes de rodar um treino grande, verifique X/Y/Z") ou gerar um documento de experiments tracking (CSV/JSON) integrado ao `outputs/` para registrar HParams e resultados.
