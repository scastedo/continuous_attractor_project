#!/usr/bin/env bash
set -euo pipefail  # safer: exit on error, undefined var, or pipeline error

# Common idir values
IDIR_VALUES="0.5 0.52 0.6" # 0.55 0.6 0.7 0.8" # 0.575 0.6 0.7 0.9"
# IDIR_VALUES="0.5 0.50025 0.5005 0.501 0.5025 0.505 0.51 0.52 0.55 0.6"
NEURONS="250"
GENERATIONS="1000"
# OUTPUT_DIR="/data/scastedo/runs_feb_14_test"
OUTPUT_DIR="/DATA/scastedo/runs_april_29_test_3"
# OUTPUT_DIR="/mnt/data/scastedo/runs_differential_2"
TRIALS="1"
SIGMA_ETA_VALUES="0  " #0.005
SIGMA_TEMP_VALUES="0.05"
SIGMA_THETA_VALUES="0" #0.01
BLOCK_SIZE="100"
I_STR="0.00"
CANN_WIDTH_VALUES="0.1"
ACTIVE_FRACTIONS="0.1"
SYN_FAIL="0.0"
SPON_REL="0.0"
AMPAR_RIN_PAIRS="1:1 0.71:1.22"
SAVE_ENERGY_METRICS="0"  # 1 = save energy_metrics.npy, 0 = disable

if [[ "$SAVE_ENERGY_METRICS" == "1" ]]; then
  ENERGY_METRICS_FLAG="--save-energy-metrics"
else
  ENERGY_METRICS_FLAG="--no-save-energy-metrics"
fi

# Single command (pairwise AMPAR/RIN; no cross terms)
python main_gain.py \
  --ampar-rin-pairs $AMPAR_RIN_PAIRS \
  --sigma-eta $SIGMA_ETA_VALUES \
  --sigma-temp $SIGMA_TEMP_VALUES \
  --idir $IDIR_VALUES \
  --N "$NEURONS" \
  --gens "$GENERATIONS" \
  --outdir "$OUTPUT_DIR" \
  --trials "$TRIALS" \
  --sigma-theta $SIGMA_THETA_VALUES \
  --block-size "$BLOCK_SIZE" \
  --i-str "$I_STR" \
  --threshold $CANN_WIDTH_VALUES \
  --active-fraction $ACTIVE_FRACTIONS \
  --syn-fail "$SYN_FAIL" \
  --spon-rel "$SPON_REL" \
  "$ENERGY_METRICS_FLAG" \
  > output.log 2>&1
