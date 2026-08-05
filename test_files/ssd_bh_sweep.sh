#!/bin/bash
# BH_TILES sweep across the verified T envelope, one cold compile per process
# (backend one-Spyre-compile-per-process limit). Clears inductor cache between
# every variant (stale-cache trap — see memory). Appends parseable BENCH lines.
# Usage: bash ssd_bh_sweep.sh   (writes /tmp/bh_sweep_results.txt)
set -u
export LD_PRELOAD=/opt/ibm/spyre/deeptools/lib/libutil.so
cd /home/sanjitaballapur/torch-spyre
OUT=/tmp/bh_sweep_results.txt
: > "$OUT"

# (T, L) pairs at the device-verified C=64 envelope; sweep bh per pair.
# bh must divide B*nheads=64. Use L=64 legal set {2,4} + broader {8,16,32}.
run() {  # T L bh
  local T=$1 L=$2 bh=$3
  rm -rf inductor-logs /tmp/torchinductor_* 2>/dev/null
  local log=/tmp/bh_${T}_${L}_${bh}.log
  timeout 360 .venv/bin/python ssd_bench.py "$T" "$L" "$bh" > "$log" 2>&1
  local line
  line=$(grep -E "BENCH" "$log")
  if [ -z "$line" ]; then
    line="FAIL T=$T L=$L bh=$bh :: $(grep -iE 'span [0-9]|exceeds|SIGABRT|Immediate|OutOfMemory|Error' "$log" | head -1)"
  fi
  echo "$line" | tee -a "$OUT"
}

for TL in "4096 64" "8192 128" "16384 256" "32768 512"; do
  set -- $TL; T=$1; L=$2
  for bh in 2 4 8 16 32; do
    run "$T" "$L" "$bh"
  done
done
echo "SWEEP DONE" | tee -a "$OUT"
