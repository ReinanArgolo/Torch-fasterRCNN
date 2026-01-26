#!/usr/bin/env bash
set -euo pipefail

# Exporta somente o relatório de Data Augmentation e Cross-Validation
# Saídas: docs/output/augmentation_cv_report.html e .pdf

REPORT_FILE="$(dirname "$0")/augmentation_cv_report.md"
OUT_DIR="$(dirname "$0")/output"
mkdir -p "$OUT_DIR"

if [ ! -f "$REPORT_FILE" ]; then
  echo "Relatório não encontrado: $REPORT_FILE" >&2
  exit 1
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc não encontrado. Instale pandoc para continuar." >&2
  exit 2
fi

TITLE="Relatório Augmentation & Cross-Validation"

echo "Gerando HTML do relatório..."
pandoc -s -o "$OUT_DIR/augmentation_cv_report.html" "$REPORT_FILE" --metadata title="$TITLE"
echo "HTML: $OUT_DIR/augmentation_cv_report.html"

PDF_ENGINE_ARGS=()
if command -v xelatex >/dev/null 2>&1; then
  # Tenta fontes amplas; se indisponível cai para padrão sem especificar mainfont
  if fc-list | grep -qi "DejaVu Serif"; then
    PDF_ENGINE_ARGS=("--pdf-engine=xelatex" -V mainfont="DejaVu Serif")
  elif fc-list | grep -qi "Liberation Serif"; then
    PDF_ENGINE_ARGS=("--pdf-engine=xelatex" -V mainfont="Liberation Serif")
  else
    echo "Fonte serif ampla não encontrada; usando xelatex com fonte padrão." >&2
    PDF_ENGINE_ARGS=("--pdf-engine=xelatex")
  fi
else
  echo "Aviso: xelatex não encontrado; tentando engine padrão (pode falhar com alguns caracteres Unicode)." >&2
fi

echo "Gerando PDF do relatório..."
pandoc -s -o "$OUT_DIR/augmentation_cv_report.pdf" "$REPORT_FILE" --metadata title="$TITLE" "${PDF_ENGINE_ARGS[@]}"
echo "PDF: $OUT_DIR/augmentation_cv_report.pdf"

echo "Concluído."
