#!/bin/bash
# Fill the bh endpoints {4,32} the main sweep skipped, at each T's winning L.
# Confirms whether bh=32 (2 chunk-heads/tile) halves peak again vs bh=16, or bh=4 wins.
set -u
export LD_PRELOAD=/opt/ibm/spyre/deeptools/lib/libutil.so
cd /home/sanjitaballapur/torch-spyre
OUT=/tmp/ssd_bh_endpoints.txt
: > "$OUT"

run() {  # T L bh
  local T=$1 L=$2 bh=$3
  rm -rf inductor-logs /tmp/torchinductor_* 2>/dev/null
  local log=/tmp/bhend_${T}_${L}_${bh}.log
  timeout 600 env HBM_POOL_PLANNING=1 SPYRE_INDUCTOR_LOG=1 \
    .venv/bin/python test_files/ssd_bench.py "$T" "$L" "$bh" > "$log" 2>&1
  local line peak
  line=$(grep -E "^SWEEP" "$log")
  peak=$(grep -oE "peak concurrent usage [0-9.]+ GB" "$log" | grep -oE "[0-9.]+" | sort -g | tail -1)
  if [ -z "$line" ]; then
    line="FAIL T=$T L=$L bh=$bh :: $(grep -iE 'span [0-9]|exceeds|SIGABRT|OutOfMemory|Error' "$log" | head -1)"
  else
    line="$line peak=${peak:-n/a}GB"
  fi
  echo "$line" | tee -a "$OUT"
}

# T -> winning L from the main sweep
for bh in 4 32; do
  run 4096 64 "$bh"
  run 8192 64 "$bh"
  run 16384 128 "$bh"
  run 32768 128 "$bh"
  run 65536 128 "$bh"
done
echo "BHEND DONE" | tee -a "$OUT"
