#!/usr/bin/env bash
set -euo pipefail  # safer: exit on error, undefined var, or pipeline error

# Common idir values
IDIR_VALUES="0.5 0.504 0.505 0.506 0.508 0.51 0.515 "
NEURONS="250"
GENERATIONS="5000"
# OUTPUT_DIR="/data/scastedo/runs_latest"
OUTPUT_DIR="/mnt/data/scastedo/runs_differential_2"
TRIALS="1"
SIGMA_ETA_VALUES="0 0.02"
SIGMA_TEMP_VALUES="0.02"
SIGMA_THETA_VALUES="0 0.01 0.5"
BLOCK_SIZE="200"

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
  > output1.log 2>&1

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
  > output2.log 2>&1
