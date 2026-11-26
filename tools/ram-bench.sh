#!/bin/sh
# RAM bandwidth on the test machine. Output: one line per measurement, raw logs in $OUT.
# usage: ram-bench.sh <label>   (e.g. xmp-off / xmp-on)
# Input:  env BENCH_DIR (output root, default ~/bench); needs mbw, dmidecode (sudo -n) and the compiled
#         stream binary (see stream.c) at $STREAM_BIN (default $BENCH_DIR/scripts/stream).
#         mbw and dmidecode are optional (skipped if missing / no passwordless sudo).
#         env THREADS_LIST: STREAM thread counts (default "1 <physical cores> <logical cores>").
# Output: $BENCH_DIR/<label>/{ram-meta,mbw-memcpy,mbw-mcblock,stream}.txt, summary on stdout.
set -eu
L=${1:?label}
BENCH_DIR=${BENCH_DIR:-$HOME/bench}
STREAM_BIN=${STREAM_BIN:-$BENCH_DIR/scripts/stream}
OUT=$BENCH_DIR/$L; mkdir -p "$OUT"
{
  echo "# $(date -Is) $L"
  sudo -n dmidecode -t memory 2>/dev/null | grep -E "Configured Memory Speed|Configured Voltage" | sort | uniq -c \
    || echo "dmidecode unavailable (needs passwordless sudo)"
} > "$OUT/ram-meta.txt"
# mbw: 4 GiB buffer, 5 runs, MEMCPY + MCBLOCK
if command -v mbw >/dev/null; then
  mbw -q -n 5 -t0 4096 > "$OUT/mbw-memcpy.txt"
  mbw -q -n 5 -t2 -b 1048576 4096 > "$OUT/mbw-mcblock.txt"
fi
# STREAM-like (stream.c, OpenMP, 3x1 GiB). sysbench memory is useless here:
# with 1M blocks it measures L2/L3 and even with 256M blocks it reports >100 GB/s.
LOG=$(nproc)
PHYS=$(lscpu -p=core 2>/dev/null | grep -v '^#' | sort -u | wc -l); [ "$PHYS" -gt 0 ] || PHYS=$LOG
for t in ${THREADS_LIST:-1 $PHYS $LOG}; do OMP_NUM_THREADS=$t OMP_PROC_BIND=spread "$STREAM_BIN"; done > "$OUT/stream.txt"
grep AVG "$OUT"/mbw-*.txt 2>/dev/null || true
cat "$OUT/stream.txt"
