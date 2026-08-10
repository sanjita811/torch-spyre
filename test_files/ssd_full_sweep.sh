#!/bin/bash
# Full (T, L, bh) sweep for the C-block SSD kernel. One cold compile per process
# (backend one-Spyre-compile-per-process limit); clears the inductor cache between every
# run (stale-cache trap). Serial (single Spyre holder). Appends parseable SWEEP lines
# with peak HBM merged in from the HBM_POOL_PLANNING log.
# Usage: bash test_files/ssd_full_sweep.sh   (writes /tmp/ssd_full_sweep.txt)
set -u
export LD_PRELOAD=/opt/ibm/spyre/deeptools/lib/libutil.so
cd /home/sanjitaballapur/torch-spyre
OUT=/tmp/ssd_full_sweep.txt
: > "$OUT"

run() {  # T L bh
  local T=$1 L=$2 bh=$3
  rm -rf inductor-logs /tmp/torchinductor_* 2>/dev/null
  local log=/tmp/sweep_${T}_${L}_${bh}.log
  timeout 600 env HBM_POOL_PLANNING=1 SPYRE_INDUCTOR_LOG=1 \
    .venv/bin/python test_files/ssd_bench.py "$T" "$L" "$bh" > "$log" 2>&1
  local line peak
  line=$(grep -E "^SWEEP" "$log")
  peak=$(grep -oE "peak concurrent usage [0-9.]+ GB" "$log" | tail -1 | grep -oE "[0-9.]+" | head -1)
  if [ -z "$line" ]; then
    line="FAIL T=$T L=$L bh=$bh :: $(grep -iE 'span [0-9]|exceeds|SIGABRT|Immediate|OutOfMemory|invalid for input|Error|StopIteration' "$log" | head -1)"
  else
    line="$line peak=${peak:-n/a}GB"
  fi
  echo "$line" | tee -a "$OUT"
}

# (T L) candidates: L in {64,128,256,512}, C=T/L<=512, C%64==0 when C>64. bh in {8,16}.
for bh in 8 16; do
  for TL in \
    "4096 64" "4096 128" "4096 256" "4096 512" \
    "8192 64" "8192 128" "8192 256" "8192 512" \
    "16384 64" "16384 128" "16384 256" "16384 512" \
    "32768 64" "32768 128" "32768 256" "32768 512" \
    "65536 128" "65536 256" "65536 512"; do
    set -- $TL
    run "$1" "$2" "$bh"
  done
done
echo "SWEEP DONE" | tee -a "$OUT"
