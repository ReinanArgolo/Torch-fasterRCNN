#!/usr/bin/env bash
set -euo pipefail

# Script simples para exportar a documentação Markdown para HTML e PDF usando pandoc
# Saída: docs/output/docs.html e docs/output/docs.pdf

OUT_DIR="$(dirname "$0")/output"
mkdir -p "$OUT_DIR"

FILES=(
  "$(dirname "$0")/index.md"
  "$(dirname "$0")/setup.md"
  "$(dirname "$0")/code_structure.md"
  "$(dirname "$0")/training.md"
  "$(dirname "$0")/cli.md"
  "$(dirname "$0")/exporting.md"
  "$(dirname "$0")/methodology.md"
  "$(dirname "$0")/augmentation_cv_report.md"
)

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc não encontrado. Instale pandoc (apt/conda/brew) para usar este script." >&2
  exit 2
fi

echo "Gerando HTML..."
pandoc -s -o "$OUT_DIR/docs.html" "${FILES[@]}" --metadata title="Documentação Faster R-CNN"
echo "HTML gerado em: $OUT_DIR/docs.html"

echo "Gerando PDF (pode exigir LaTeX)..."
PDF_ENGINE_ARGS=()
if command -v xelatex >/dev/null 2>&1; then
  PDF_ENGINE_ARGS=("--pdf-engine=xelatex")
else
  echo "Aviso: xelatex não encontrado; tentando engine padrão do pandoc (pode falhar com caracteres Unicode)." >&2
fi
pandoc -s -o "$OUT_DIR/docs.pdf" "${FILES[@]}" --metadata title="Documentação Faster R-CNN" "${PDF_ENGINE_ARGS[@]}"
echo "PDF gerado em: $OUT_DIR/docs.pdf"

echo "Concluído."
