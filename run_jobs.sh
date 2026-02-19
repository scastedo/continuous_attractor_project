#!/usr/bin/env bash
set -euo pipefail  # safer: exit on error, undefined var, or pipeline error

# Common idir values
IDIR_VALUES="0.5 0.52 0.55 0.6 0.7" # 0.575 0.6 0.7 0.9"
# IDIR_VALUES="0.5 0.50025 0.5005 0.501 0.5025 0.505 0.51 0.52 0.55 0.6"
NEURONS="200"
GENERATIONS="500000"
# OUTPUT_DIR="/data/scastedo/runs_feb_14_test"
OUTPUT_DIR="/DATA/scastedo/runs_feb_18_long"
# OUTPUT_DIR="/mnt/data/scastedo/runs_differential_2"
TRIALS="1"
SIGMA_ETA_VALUES="0.00 0.005" #0.002
SIGMA_TEMP_VALUES="0.03"
SIGMA_THETA_VALUES="0 0.02" #0.01
BLOCK_SIZE="100"
I_STR="0.025"
THRESHOLD="0.1"
SYN_FAIL="0.0"
SPON_REL="0.0"
AMPAR_RIN_PAIRS="1:1 0.64:1.27"

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
  --threshold "$THRESHOLD" \
  --syn-fail "$SYN_FAIL" \
  --spon-rel "$SPON_REL" \
  > output.log 2>&1