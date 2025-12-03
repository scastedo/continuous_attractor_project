#!/usr/bin/env bash
set -euo pipefail  # safer: exit on error, undefined var, or pipeline error

# Common idir values
IDIR_VALUES="0.5 0.503 0.505 0.506 0.5075 0.509 0.51 0.515 0.52 0.53"
NEURONS="300"
GENERATIONS="30000"
# OUTPUT_DIR="/data/scastedo/runs_latest"
OUTPUT_DIR="/mnt/data/scastedo/runs_latest_long"
TRIALS="10"
SIGMA_ETA_VALUES="0 0.02 0.04"
SIGMA_TEMP_VALUES="0.02 0.04"

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
  > output2.log 2>&1
