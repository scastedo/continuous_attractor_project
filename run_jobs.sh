#!/usr/bin/env bash
set -euo pipefail  # safer: exit on error, undefined var, or pipeline error

# Common idir values
IDIR_VALUES="0.5 0.51 0.52 0.55 0.6 0.7" # 0.575 0.6 0.7 0.9"
# IDIR_VALUES="0.5 0.50025 0.5005 0.501 0.5025 0.505 0.51 0.52 0.55 0.6"
NEURONS="250"
GENERATIONS="2000000"
OUTPUT_DIR="/data/scastedo/runs_feb_11_equilibrium_test"
# OUTPUT_DIR="/mnt/data/scastedo/runs_differential_2"
TRIALS="1"
SIGMA_ETA_VALUES="0.00 0.002"
SIGMA_TEMP_VALUES="0.01"
SIGMA_THETA_VALUES="0 0.01"
BLOCK_SIZE="250"

# 1st command
python main_gain.py \
  --ampar 1 \
  --rin 1 \
  --sigma-eta $SIGMA_ETA_VALUES \
  --sigma-temp $SIGMA_TEMP_VALUES \
  --idir $IDIR_VALUES \
  --N $NEURONS \
  --gens $GENERATIONS \
  --outdir $OUTPUT_DIR \
  --trials $TRIALS \
  --sigma-theta $SIGMA_THETA_VALUES \
  --block-size $BLOCK_SIZE \
  > output1.log 2>&1 &

# 2nd command
python main_gain.py \
  --ampar 0.64 \
  --rin 1.27 \
  --sigma-eta $SIGMA_ETA_VALUES \
  --sigma-temp $SIGMA_TEMP_VALUES \
  --idir $IDIR_VALUES \
  --N $NEURONS \
  --gens $GENERATIONS \
  --outdir $OUTPUT_DIR \
  --trials $TRIALS \
  --sigma-theta $SIGMA_THETA_VALUES \
  --block-size $BLOCK_SIZE \
  > output2.log 2>&1 &

  wait