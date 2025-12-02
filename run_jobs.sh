#!/usr/bin/env bash
set -e  # exit if any command fails

# 1st command
python main_gain.py \
  --ampar 1 \
  --rin 1 \
  --sigma-eta 0 0.02 0.04\
  --sigma-temp 0.02 0.03\
  --idir 0.5 0.503 0.505 0.5075 0.51 0.515 0.52 0.53\
  > output1.log 2>&1

# 2nd command – change this to whatever your second run is
python main_gain.py \
  --ampar 0.63 \
  --rin 1.22 \
  --sigma-eta 0 0.02 0.04 \
  --sigma-temp 0.02 0.04\
  --idir 0.5 0.503 0.505 0.5075 0.51 0.515 0.52 0.53 \
  > output2.log 2>&1