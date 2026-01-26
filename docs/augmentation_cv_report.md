# Relatório Metodológico: Data Augmentation e Validação Cruzada em Detecção com Faster R-CNN

Resumo
- Este relatório descreve, em formato metodológico, os princípios, decisões e procedimentos recomendados para o uso de Data Augmentation e Validação Cruzada (K-fold) ao treinar detectores do tipo Faster R-CNN em datasets COCO-like.  
- Fornecemos justificativas teóricas, protocolos experimentais, fórmulas para sumarização de resultados entre folds e um checklist prático para reprodutibilidade.

1. Introdução
- Objetivo: maximizar a capacidade de generalização do detector equilibrando viés e variância sob restrição de dados rotulados.  
- Problema: modelos de detecção são sensíveis a variações de escala, iluminação, oclusões e distribuição de classes. Data Augmentation mitiga overfitting; Validação Cruzada provê estimativas robustas de desempenho e variância.

2. Conjunto de Dados e Pré-processamento
- Formato: COCO (imagens + anotações com bboxes e categorias).  
- Qualidade de anotação é crítica: remover caixas degeneradas (w≤0 ou h≤0), padronizar índices de classes e consistência entre arquivos e imagens.  
- Normalização/Conversão: imagens em [0,1], preservando aspecto durante resize; bboxes transformadas de forma consistente.

3. Data Augmentation
3.1. Objetivos e Princípios
- Aumentar a diversidade de exemplos sem alterar a semântica das anotações.  
- Induzir invariâncias relevantes (translação, variações leves de escala e cor), mantendo a tarefa (detectar e localizar) inalterada.

3.2. Operadores Recomendados (com diretrizes práticas)
- Redimensionamento multi-escala (short_sizes ∈ {640,…,800}, max_size≈1333): melhora robustez a escala de objetos.  
- Flip horizontal (p≈0.5): útil para objetos sem orientação dominante; evite se a orientação for semanticamente relevante.  
- Jitter de cor (p≈0.5; brilho/contraste/saturação modestos): promove invariância a iluminação; evite exageros que alterem textura-chave.  
- Crop baseado em IoU mínimo (min_iou≈0.3–0.5): força o modelo a aprender com partes de objetos; monitore para não eliminar sistematicamente as anotações.  
- Random translate/scale leves: ampliar variação sem distorcer aspecto. Evitar rotações rígidas de 90° se a orientação importar.


3.3. Protocolo de Avaliação da Augmentation
- Ablações: compare baseline (mínimo de augmentations) vs. incrementos (ex.: +flip, +multi-escala, +jitter, +IoU-crop).  
- Métricas: mAP (AP@[.5:.95]), AP50 e AP75 por época; inspecione curvas de aprendizado e estabilidade (desvio padrão entre folds).  
- Critérios: aceite operadores que melhoram mAP médio e reduzem variância sem degradar AP75 (precisão de localização).


4. Validação Cruzada (K-fold)
4.1. Motivação
- Estimar desempenho esperado e sua variabilidade quando os dados rotulados são limitados.  
- Reduzir dependência de uma única divisão treino/val.

4.2. Formação dos Folds
- Estratificação por classes (quando possível) para manter proporções aproximadas de categorias em cada fold.  
- Em detecção, estratificar por “presença por classe na imagem” e por grupos naturais para mitigar vazamento (leakage).  
- Garantir que imagens correlacionadas não apareçam simultaneamente em treino e validação do mesmo fold.

4.3. Procedimento K-fold
- Particione IDs de imagem em K subconjuntos disjuntos.  
- Para cada fold i∈{1,…,K}: valide naquele fold e treine nos demais K-1 folds.  
- Hiperparâmetros fixos ao longo dos folds; early stopping com monitor consistente (val_mAP ou val_loss).  
- Semente aleatória controlada para reprodutibilidade (Python/NumPy/PyTorch).

4.4. Agregação de Métricas e Intervalos de Confiança
- Para uma métrica M (ex.: mAP) por fold M_i, calcule média e desvio padrão:
  $$\bar{M} = \frac{1}{K} \sum_{i=1}^{K} M_i, \quad s = \sqrt{\frac{1}{K-1} \sum_{i=1}^{K} (M_i - \bar{M})^2}$$
- Intervalo de confiança (IC) 95% para a média (aproximação t-Student):
  $$IC_{95\%} = \bar{M} \pm t_{0.975,\,K-1} \cdot \frac{s}{\sqrt{K}}$$
- Reporte também AP50 e AP75 agregados; inclua tabela por fold e o IC para análise de robustez.

Referências
- Ren et al., 2015. Faster R-CNN.  
- Lin et al., 2017. Feature Pyramid Networks.  
- COCO Evaluation Toolkit e métricas AP@[.5:.95].
