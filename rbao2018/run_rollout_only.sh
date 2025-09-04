#!/bin/bash

set -x

python train.py \
  --debug-rollout-only \
  --hf-checkpoint /path/to/hf/ckpt \
  --rollout-num-gpus 1 \
  --rollout-num-gpus-per-node 1 \
  --rollout-num-gpus-per-engine 1 \
  --rollout-batch-size 8 \
  --n-samples-per-prompt 1 \
  --num-rollout 10 \
  --save-debug-rollout-data "/tmp/slime_debug/rollout_data_{rollout_id}.pt"