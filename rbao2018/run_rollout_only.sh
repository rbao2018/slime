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

# ray stop
# pkill -9 -f '\bray\b'
# sleep 5
# pkill -9 -f '\bray\b'
# ray start --head --node-ip-address localhost --num-gpus 8 --disable-usage-stats

source /ossfs/workspace/slime/scripts/models/qwen3-8B.sh

CKPT_ARGS=(
   --hf-checkpoint /root/models/Qwen3-8B
   --ref-load /root/GLM-Z1-9B-0414_torch_dist
   --load /root/GLM-Z1-9B-0414_slime/
   --save /root/GLM-Z1-9B-0414_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/zhuzilin/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
)

export PYTHONPATH=/root/Megatron-LM:$PYTHONPATH

python /root/slime/train.py \
  --debug-rollout-only \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --rollout-num-gpus 8 \
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
