# Configuração do Ambiente

Requisitos principais (Python e pacotes):

- Python 3.10+ recomendado
- CUDA compatível (se usar GPU)
- Dependências Python listadas em `requirements.txt`

Instalação rápida (virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dependências de sistema (opcionais para exportar docs):
- `pandoc` — para converter Markdown em HTML/PDF (instale via apt/conda/homebrew)
- LaTeX (por exemplo `texlive`) — necessário para conversão para PDF via pandoc

Ubuntu (exemplo):

```bash
sudo apt update
sudo apt install pandoc texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended
```

Observações:
- As versões exatas do PyTorch devem corresponder ao seu CUDA; o `requirements.txt` inclui um índice PyTorch com uma build CUDA como exemplo.
- Para usar AMP (treinamento com precisão mista) habilite CUDA e `amp: true` em `config.yaml`.
