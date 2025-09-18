# Copyright 2025 rbao2018. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -x

# 获取格式 YYYYMMDD_HHMMSS
time_str=$(date +"%Y%m%d_%H%M%S")

source /root/slime/scripts/models/qwen2.5-7B.sh

CKPT_ARGS=(
   --hf-checkpoint /input/models/Qwen2.5-7B-Instruct
   --ref-load /root/GLM-Z1-9B-0414_torch_dist
   --load /root/GLM-Z1-9B-0414_slime/
   --save /root/GLM-Z1-9B-0414_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /input/baorong.bao/datasets/AM-Thinking-v1-Distilled/math_all_prompt.jsonl
   --rollout-function-path slime_plugins.rbao2018.sglang_rollout_pebble.generate_rollout
   --dynamic-sampling-filter-path slime_plugins.rbao2018.dynamic_sampling_filters.more_than_half_correct
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --num-rollout 3000
   --rollout-batch-size 1024
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --global-batch-size 8192
   --balance-data
   --debug-rollout-only
   --save-debug-rollout-data "/input/baorong.bao/tmp/slime_outputs/${time_str}/qwen25_7B_passrate_half_data_{rollout_id}.pt"
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

SGLANG_ARGS=(
   --rollout-num-gpus 16
   --rollout-num-gpus-per-engine 2
   --sglang-server-concurrency 32
   --sglang-disable-radix-cache
)

python3 /root/slime/train.py \
  --actor-num-nodes 2 \
  --actor-num-gpus-per-node 8 \
  --colocate \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${DISTRIBUTED_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]}