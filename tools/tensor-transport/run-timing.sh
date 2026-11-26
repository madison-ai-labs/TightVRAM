#!/usr/bin/env bash
# Timing per format on layer 20 (gate: K=2048, down: K=512), using the real quantized data from run-quant.sh.
#   run-timing.sh cpu   tg path: MUL_MAT_ID on the CPU, 1 token, 8 random of 256 experts, 10 threads
#   run-timing.sh ssd   O_DIRECT read from an NVMe (needs a GGUF file that is not otherwise in use)
#   run-timing.sh gpu   the GPU must be idle: MUL_MAT_ID in VRAM (1 / 512 / 4096 tokens) and host->VRAM
#                       copies of whole tensors
# Inputs (env): TT (default ./tt), Q_DIR (default ./q, quantized tensors from run-quant.sh),
#               SSD_FILE (a large GGUF on the drive to test, required for "ssd")
# Outputs: ./results/{cpu,ssd,gpu,h2d}.tsv (appended)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
TT="${TT:-./tt}"
Q_DIR="${Q_DIR:-q}"
EINZEL="F16 Q8_0 Q6_K Q5_K Q4_K IQ4_XS IQ4_NL MXFP4 Q3_K IQ3_S IQ3_XXS Q2_K IQ2_S IQ2_XS IQ2_XXS IQ1_M"
PROGRESSIV="Q2_K+Q2_K Q2_K+Q3_K Q2_K+Q4_K Q3_K+Q2_K Q3_K+Q3_K Q4_K+Q2_K Q4_K+Q4_K Q2_K+Q2_K+Q2_K IQ2_XS+IQ2_XS IQ2_XS+Q2_K IQ3_XXS+IQ2_XXS IQ4_XS+IQ2_XXS"
TENSOREN="blk.20.ffn_gate_exps.weight blk.20.ffn_down_exps.weight"
case "${1:?usage: run-timing.sh cpu|ssd|gpu}" in
cpu)
  for T in $TENSOREN; do
    for wdh in 1 2; do  # two passes to see drift between formats ("durchgang" = pass, kept as column key)
      $TT cpu $Q_DIR $T 1 300 $EINZEL $PROGRESSIV | sed "s/^/durchgang=$wdh\t/" | tee -a results/cpu.tsv
    done
  done ;;
ssd)
  $TT ssd "${SSD_FILE:?set SSD_FILE to a large GGUF on the drive under test}" 1024 | tee -a results/ssd.tsv ;;
gpu)
  for T in $TENSOREN; do
    for N in 1 512 4096; do
      R=200; [ $N = 4096 ] && R=40
      $TT gpu $Q_DIR $T $N $R $EINZEL $PROGRESSIV | tee -a results/gpu.tsv
    done
    $TT h2d $Q_DIR $T 10 $EINZEL | tee -a results/h2d.tsv
  done ;;
esac
