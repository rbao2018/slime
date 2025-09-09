# Unified Logging System for Slime Framework

这个文档说明了在slime框架中的统一logging系统，支持同时写入wandb和TensorBoard。

## 功能特性

### 1. 与wandb并行运行
- TensorBoard和wandb可以同时启用，互不影响
- 支持相同的指标记录，但使用不同的存储后端
- 配置参数独立，可以分别控制

### 2. 智能自动启用
- **自动检测**: 当设置了 `TENSORBOARD_DIR` 环境变量时，自动启用TensorBoard（无需 `--use-tensorboard` 参数）
- **环境变量优先级**: `TENSORBOARD_DIR` 控制日志目录，`TENSORBOARD_MODE` 控制启用/禁用状态
- **显式控制**: 可以通过 `--use-tensorboard=false` 或 `--tensorboard-mode=disabled` 显式禁用

### 3. 配置复用
- 当TensorBoard特定配置未设置时，自动回退到wandb配置
- 保持一致的项目和组名称体系

## 配置参数

### 命令行参数

```bash
# 基本配置
--use-tensorboard                    # 启用TensorBoard
--tensorboard-mode {enabled,disabled}  # 模式控制
--tensorboard-dir DIR               # 本地存储目录

# 项目组织
--tensorboard-project PROJECT       # 项目名称
--tensorboard-group GROUP          # 组名称
--disable-tensorboard-random-suffix # 禁用随机后缀
```

### 环境变量

```bash
export TENSORBOARD_DIR="/path/to/tensorboard/logs"  # 日志目录（优先级最高）
export TENSORBOARD_MODE="disabled"                  # 禁用TensorBoard
```

## 使用方法

### 1. 基本使用

```bash
# 方式1: 仅设置环境变量（自动启用TensorBoard）
export TENSORBOARD_DIR="/path/to/tensorboard/logs"
python train.py

# 方式2: 显式启用TensorBoard
python train.py --use-tensorboard

# 方式3: 同时启用wandb和TensorBoard
python train.py --use-wandb --use-tensorboard \
    --wandb-project my_project \
    --tensorboard-project my_project
```

### 2. 环境变量配置

```bash
# 自动启用模式（推荐）
export TENSORBOARD_DIR="/shared/tensorboard_logs"
python train.py  # TensorBoard会自动启用

# 显式禁用（即使设置了环境变量）
export TENSORBOARD_DIR="/shared/tensorboard_logs"
python train.py --tensorboard-mode=disabled

# 查看日志
tensorboard --logdir /shared/tensorboard_logs
```

### 3. 配置复用示例

```bash
# TensorBoard会自动使用wandb的项目和组配置
python train.py --use-wandb --use-tensorboard \
    --wandb-project my_rl_experiment \
    --wandb-group baseline_run
```

## 自动启用逻辑

TensorBoard的启用遵循以下优先级逻辑：

1. **显式启用**: `--use-tensorboard` 参数最高优先级
2. **显式禁用**: `--use-tensorboard=false` 或 `--tensorboard-mode=disabled` 强制禁用
3. **自动检测**: 当设置了 `TENSORBOARD_DIR` 环境变量且未被显式禁用时，自动启用
4. **默认禁用**: 以上条件都不满足时，保持禁用状态

```bash
# 示例：各种组合的行为
export TENSORBOARD_DIR="/logs"

python train.py                           # ✓ 自动启用
python train.py --use-tensorboard         # ✓ 显式启用
python train.py --use-tensorboard=false   # ✗ 显式禁用（优先级更高）
python train.py --tensorboard-mode=disabled # ✗ 通过mode禁用

# 未设置环境变量
python train.py                           # ✗ 默认禁用
python train.py --use-tensorboard         # ✓ 显式启用
```

## 目录结构

默认情况下，TensorBoard日志按以下结构组织：

```
tensorboard_logs/
├── project_name/
│   ├── group_name_abc123/          # 带随机后缀
│   │   ├── events.out.tfevents.*   # TensorBoard事件文件
│   │   └── ...
│   └── another_group_def456/
└── another_project/
```

## 指标体系

TensorBoard记录与wandb相同的指标类别：

### 训练指标 (`train/*`)
- 损失函数 (loss, kl_loss, ppo_kl等)
- 梯度范数 (grad_norm)
- 学习率 (lr-pg_*)
- 步骤指标：训练步数

### Rollout指标 (`rollout/*`)
- 对数概率 (log_probs, ref_log_probs)
- 奖励统计 (rewards, values, advantages)
- 响应长度统计
- 步骤指标：rollout轮次

### 评估指标 (`eval/*`)
- 各种评估任务的奖励
- 截断比例
- 步骤指标：评估轮次

### 性能指标 (`perf/*`)
- 时间统计 (rollout_time, train_time等)
- 吞吐量 (tokens_per_gpu_per_sec)
- TFLOPS计算

### 专项指标
- 多轮对话 (`multi_turn/*`)
- 通过率 (`passrate/*`)

## 技术实现

### 1. 核心文件

- `slime/utils/logging_utils.py` - 统一logging接口（新增）
- `slime/utils/tensorboard_utils.py` - TensorBoard工具函数
- `slime/utils/wandb_utils.py` - wandb工具函数（现有）
- `slime/utils/arguments.py` - 命令行参数定义
- `train.py` / `train_async.py` - 主训练脚本集成

### 2. 统一初始化流程

```python
# 主进程统一初始化
from slime.utils.logging_utils import init_logging
wandb_run_id, tensorboard_run_id = init_logging(args)

# 次级进程统一初始化（分布式训练）
from slime.utils.logging_utils import init_logging_secondary
init_logging_secondary(args, wandb_run_id, tensorboard_run_id)
```

### 3. 统一日志记录

```python
# 统一记录接口，自动处理wandb和TensorBoard
from slime.utils.logging_utils import log_metrics, log_train_metrics

# 通用指标记录
log_metrics({"loss": 0.5, "accuracy": 0.85, "step": 100}, args)

# 训练指标记录（自动计算step）
log_train_metrics(rollout_id, step_id, {"loss": 0.5}, args)
```

## 与wandb的差异

| 特性 | wandb | TensorBoard |
|------|-------|-------------|
| 云端同步 | ✓ | ✗ |
| 离线模式 | ✓ | 默认离线 |
| 指标定义 | 显式 | 隐式 |
| 分布式支持 | 完整 | 简化 |
| 配置复杂度 | 高 | 低 |

## 测试

运行测试脚本验证功能：

```bash
cd /path/to/slime
python test_tensorboard.py
```

测试包括：
- 基本初始化和日志记录
- 环境变量覆盖
- 禁用模式
- 配置回退机制

## 故障排除

### 1. 日志目录问题

```bash
# 检查目录权限
ls -la $TENSORBOARD_DIR

# 手动创建目录
mkdir -p $TENSORBOARD_DIR
```

### 2. 导入错误

确保安装了TensorBoard：
```bash
pip install tensorboard
```

### 3. 分布式训练中的日志

TensorBoard采用简化的分布式策略，主要由主进程处理写入，避免文件冲突。

## 最佳实践

1. **目录管理**：使用共享存储作为TensorBoard日志目录
2. **命名规范**：保持项目和组名称的一致性
3. **监控资源**：TensorBoard日志会占用磁盘空间，定期清理
4. **网络访问**：在集群环境中，确保TensorBoard服务可以访问日志目录

## 示例配置

```bash
# 完整的训练命令示例
export TENSORBOARD_DIR="/shared/experiments/tensorboard"

python train.py \
    --use-wandb \
    --use-tensorboard \
    --wandb-project rlhf_experiments \
    --wandb-group qwen_baseline \
    --tensorboard-project rlhf_experiments \
    --tensorboard-group qwen_baseline \
    --num-rollout 100 \
    --global-batch-size 64

# 启动TensorBoard服务
tensorboard --logdir /shared/experiments/tensorboard --port 6006
```
