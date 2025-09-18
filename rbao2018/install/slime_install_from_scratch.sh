
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

# FROM reg.docker.alibaba-inc.com/aii/aistudio:aistudio-195898721-3620778887-1755620561162

pip uninstall deepspeed trl verl sglang unsloth unsloth-zoo flashinfer atorch atorch-addon flash_attn flash_infer ray mdatasets bitsandbytes -y

pip install uv -i https://pypi.antfin-inc.com/simple
uv pip install "flashinfer-python==0.2.9.rc2" -i https://pypi.antfin-inc.com/simple
uv pip install "sglang[all]==0.4.10.post2" -i https://pypi.antfin-inc.com/simple
uv pip install "ray[default,adag,cgraph]" "httpx[http2]" wandb pylatexenc blobfile accelerate "mcp[cli]" -i https://pypi.antfin-inc.com/simple

# cumem_allocator
cd /root && git clone https://github.com/zhuzilin/cumem_allocator.git && pip install -e /root/cumem_allocator -i https://pypi.antfin-inc.com/simple

# install repo for antgroup internal use
uv pip install pandas codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 math-verify mathruler -i https://pypi.antfin-inc.com/simple
uv pip install mbridge blobfile pebble word2number pytest py-spy pyext pre-commit ruff bitsandbytes -i https://pypi.antfin-inc.com/simple

# download slime
cd /root && git clone https://github.com/rbao2018/slime.git && pip install -e /root/slime --no-deps

# reinstall transformer engine
pip uninstall transformer_engine_cu12 transformer_engine_torch -y
# MAX_JOBS=64 NVTE_FRAMEWORK=pytorch pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
MAX_JOBS=64 pip install --no-build-isolation "transformer_engine[pytorch]>=2.5.0" -i https://pypi.antfin-inc.com/simple

# reinstall apex
pip uninstall apex -y && cd /root && git clone https://github.com/NVIDIA/apex.git && cd apex
NVCC_APPEND_FLAGS="--threads 4" pip -v install --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" ./

# # reinstall flash attention 2
# pip uninstall flash-attn -y
# MAX_JOBS=64 pip install --no-build-isolation flash-attn==2.7.4.post1 -i https://pypi.antfin-inc.com/simple


# export PYTHONPATH="/root/Megatron-LM:$PYTHONPATH"
# python -c "from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler"
# python -c "from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint"
# python -c "from flash_attn import flash_attn_qkvpacked_func, flash_attn_func"
# python -c "import apex;import torch;import fused_weight_gradient_mlp_cuda"
# python -c "import torch; print(torch.backends.cudnn.version())"


# tar -xf cudnn-linux-x86_64-...._cuda12-archive.tar.xz

# # 复制头文件
# sudo cp -P cudnn-linux-x86_64-9.12.0.46_cuda12-archive/include/cudnn*.h /usr/local/cuda-12.6.3/include/
# sudo cp -P cudnn-linux-x86_64-9.12.0.46_cuda12-archive/lib/libcudnn* /usr/local/cuda-12.6.3/lib64/
# sudo chmod a+r /usr/local/cuda-12.6.3/include/cudnn*.h /usr/local/cuda-12.6.3/lib64/libcudnn*

# add to .bashrc and delete .bashrc the mount /nas command
# export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
