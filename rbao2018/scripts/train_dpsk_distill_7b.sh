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

echo "TENSORBOARD_DIR is "$TENSORBOARD_DIR
echo "TORCH_NCCL_AVOID_RECORD_STREAMS is "$TORCH_NCCL_AVOID_RECORD_STREAMS
echo "NCCL_ALGO is "$NCCL_ALGO
echo "NCCL_IB_CUDA_SUPPORT is "$NCCL_IB_CUDA_SUPPORT
echo "NCCL_NET_GDR_READ is" $NCCL_NET_GDR_READ
echo "VLLM_USE_V1 is "$VLLM_USE_V1
echo "NUM_OF_NODES is" $NUM_OF_NODES
echo "MOE_MLP_PREFIX is" $MOE_MLP_PREFIX
echo "CUDA_DEVICE_MAX_CONNECTIONS is" $CUDA_DEVICE_MAX_CONNECTIONS
echo "CUDA_LAUNCH_BLOCKING is" $CUDA_LAUNCH_BLOCKING
echo "NVTE_BATCH_MHA_P2P_COMM is" $NVTE_BATCH_MHA_P2P_COMM

# ------------------------------------- verl mcore sglang or vllm ---------------------------------- 


# 获取格式 YYYYMMDD_HHMMSS
time_str=$(date +"%Y%m%d_%H%M%S")
# 使用该时间字符串设置 AIS_MEMO（如果未设置）
if [[ -z "$AIS_MEMO" ]]; then
   AIS_CKPT_NAME="slime_save_${time_str}"
fi

echo "AIS_CKPT_NAME: $AIS_CKPT_NAME"

python /input/baorong.bao/prepare_infra/watchdog_ais_ckpt.py --model_path /input/baorong.bao/models/$AIS_CKPT_NAME >/root/watchdog_ais_ckpt.log 2>&1 &

# DeepSeek-R1-Distill-Qwen-7B
MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 3584
   --ffn-hidden-size 18944
   --num-attention-heads 28
   --group-query-attention
   --num-query-groups 4
   --max-position-embeddings 131072
   --seq-length 4096
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-06
   --rotary-base 10000
   --vocab-size 152064
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type alltoall
   --untie-embeddings-and-output-weights
   --attention-dropout 0.0
   --hidden-dropout 0.0
)

CKPT_ARGS=(
   --hf-checkpoint /input/models/DeepSeek-R1-Distill-Qwen-7B
   --ref-load /input/models/DeepSeek-R1-Distill-Qwen-7B_mcore13_dcp
   --save-interval 100
   --save /input/baorong.bao/models/$AIS_CKPT_NAME
)

ROLLOUT_ARGS=(
   --rollout-function-path slime_plugins.rbao2018.sglang_rollout_pebble.generate_rollout
   --prompt-data /input/baorong.bao/datasets/zhuzilin/dapo-math-17k/dapo-math-17k.jsonl
   --dynamic-sampling-filter-path slime_plugins.rbao2018.dynamic_sampling_filters.is_reward_zero_std
   --input-key prompt
   --label-key label
   --apply-chat-template
   --num-rollout 3000
   --rollout-batch-size 256
   --rollout-num-process 800
   --rollout-max-response-len 16000
   --rollout-temperature 1.0
   --rollout-shuffle
   --n-samples-per-prompt 16
   --global-batch-size 1024
   --micro-batch-size 8
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9000
   --balance-data
)


DISTRIBUTED_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --sequence-parallel
)

PERF_ARGS=(
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --use-tis
)

OPTIMIZER_ARGS=(
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-mode offline
   --wandb-group rbao2018_test
   --wandb-dir /input/baorong.bao/slime/outputs/
   --tensorboard-dir "/home/admin/logs/tfevent"
)


SGLANG_ARGS=(
   --rollout-num-gpus 8
   --rollout-num-gpus-per-engine 2
   --sglang-server-concurrency 32
   --sglang-max-running-requests 64
   --sglang-disable-radix-cache
)


python /root/slime/train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${SGLANG_ARGS[@]} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   --log-passrate
   