#!/bin/bash
set -x


ray stop --force
pkill -9 -x sglang
pkill -9 -x ray

sleep 3
pkill -9 -x sglang
pkill -9 -x ray

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"

source "${SCRIPT_DIR}/../../slime/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/pretrained_models/Qwen3-8B-Base/
   --ref-load /root/pretrained_models/Qwen3-8B-Base_torch_dist
   --load /root/pretrained_models/Qwen3-8B-Base_slime/
   --save /root/pretrained_models/Qwen3-8B-Base_slime/
   --save-interval 1000
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /root/data/openhermes2_5.parquet
   --input-key messages
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 128
   --global-batch-size 128

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-warmup-iters 128
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.9
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-8B-base-sft
   # --wandb-key ${WANDB_KEY}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8

export CUDA_DEVICE_MAX_CONNECTIONS='1'
export PYTHONPATH='/root/Megatron-LM/':$PYTHONPATH
# we always has the nvlink
export NCCL_NVLS_ENABLE='1'


ray job submit -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}
