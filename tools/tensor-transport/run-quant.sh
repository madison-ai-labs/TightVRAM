#!/usr/bin/env bash
# Quality matrix: 2 layers x gate/up/down compared against BF16, quantized with the unsloth imatrix
# (same way llama-quantize does it). CPU only (10 threads), ~5.5 GB RAM.
# Inputs (env, with defaults relative to this script's directory):
#   DATA_DIR  default ./data   containing bf16/Qwen3.6-35B-A3B-BF16-00001.gguf (see fetch-bf16.py)
#                              and imatrix/imatrix_unsloth.gguf
#   MODEL     the shipped GGUF used as comparison (e.g. Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf), required
#   TT        default ./tt     (see build.sh)
# Outputs: ./results/quant.tsv (raw rows), ./results/quant.log (stderr), ./q/ (quantized tensors)
# Run:     MODEL=/path/to/model.gguf ./run-quant.sh
set -euo pipefail
cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-./data}"
TT="${TT:-./tt}"
VGL="${MODEL:?set MODEL to the comparison GGUF}"
mkdir -p q results
REF="$DATA_DIR/bf16/Qwen3.6-35B-A3B-BF16-00001.gguf"
export TT_IMATRIX="$DATA_DIR/imatrix/imatrix_unsloth.gguf" TT_IMATRIX_QUANT=1
# Format lists: single formats, progressive (base+residual), outlier channel (+outP = P per mille), shared base
EINZEL="F16 Q8_0 Q6_K Q5_K Q4_K IQ4_XS IQ4_NL MXFP4 Q3_K IQ3_S IQ3_XXS Q2_K IQ2_S IQ2_XS IQ2_XXS IQ1_M"
PROGRESSIV="Q2_K+Q2_K Q2_K+Q3_K Q2_K+Q4_K Q3_K+Q2_K Q3_K+Q3_K Q4_K+Q2_K Q4_K+Q4_K Q2_K+Q2_K+Q2_K IQ2_XS+IQ2_XS IQ2_XS+Q2_K IQ3_XXS+IQ2_XXS IQ4_XS+IQ2_XXS"
AUSREISSER="Q2_K+out5 Q2_K+out20 Q3_K+out5 IQ2_XS+out5"
GETEILT="shared:Q4_K+Q2_K"
: > results/quant.tsv
for L in 5 20; do for M in gate up down; do
  T=blk.$L.ffn_${M}_exps.weight
  $TT quant $REF $T q $VGL -- $EINZEL $PROGRESSIV $AUSREISSER $GETEILT 2>>results/quant.log | tee -a results/quant.tsv
done; done
echo done
