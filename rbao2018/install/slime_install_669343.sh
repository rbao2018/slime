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

sed -i 's/export NCCL_DEBUG="INFO"/export NCCL_DEBUG="WARN"/g' /etc/profile.d/pouchenv.sh

sudo yum install numactl-libs numactl-devel -y

uv pip install nvitop sglang-router --force-reinstall -i https://pypi.antfin-inc.com/simple

# reinstall slime
pip uninstall slime -y && rm -rf /root/slime
cd /root && git clone -b "250901" https://github.com/rbao2018/slime.git && pip install /root/slime


# reinstall megatron-core with patch
pip uninstall megatron-core -y && rm -rf /root/Megatron-LM
cd /root && git clone -b "core_v0.13.1" https://github.com/NVIDIA/Megatron-LM.git
cp /root/slime/docker/patch/v0.4.10-cu126/megatron.patch /root/Megatron-LM/
cd /root/Megatron-LM && git apply --3way megatron.patch
pip install /root/Megatron-LM


# reinstall sglang with patch
pip uninstall sglang -y && rm -rf /root/sglang
cd /root && git clone https://github.com/sgl-project/sglang.git
cd /root/sglang && git fetch --all && git checkout 5c14515feca116ff31c665484d01fd416597341b
yes | cp -rf /root/slime/rbao2018/pyproject.toml /root/sglang/python 
cp /root/slime/docker/patch/v0.4.10-cu126/sglang.patch /root/sglang
cd /root/sglang && git apply --3way sglang.patch
# Check for conflicts
if grep -R -n '^<<<<<<< ' .; then
    echo "Patch failed to apply cleanly. Please resolve conflicts."
    exit 1
fi
rm -rf sglang.patch

uv pip install "/root/sglang/python[all]" -i https://pypi.antfin-inc.com/simple


# export PYTHONPATH="/root/Megatron-LM:$PYTHONPATH"
# python -c "from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler"
# python -c "from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint"
# python -c "from flash_attn import flash_attn_qkvpacked_func, flash_attn_func"
# python -c "import apex;import torch;import fused_weight_gradient_mlp_cuda"
# python -c "import torch; print(torch.backends.cudnn.version())"
# python -c "import sglang; print(sglang.__version__)"


cd /root && git clone https://github.com/deepseek-ai/DeepEP.git
cd /root/DeepEP && MAX_JOBS=32 python setup.py install

# start math-verify server in advance
bash /workspace/bin/prepare_infra/evaluation/serve_math_verify/serve_hf_math_verify.sh
