"""
Unified logging utility for both wandb and TensorBoard.
Provides a single interface for logging metrics to multiple backends.
"""

import os
from typing import Dict, Any, Optional


def _should_use_tensorboard(args) -> bool:
    """Check if TensorBoard should be used based on args or environment variables."""
    # Explicit setting takes priority
    if hasattr(args, 'use_tensorboard') and args.use_tensorboard:
        return True
    
    # Auto-enable if TENSORBOARD_DIR is set and not explicitly disabled
    if os.environ.get("TENSORBOARD_DIR"):
        # Check if explicitly disabled
        if hasattr(args, 'use_tensorboard') and args.use_tensorboard is False:
            return False
        # Check via tensorboard_mode
        if hasattr(args, 'tensorboard_mode') and args.tensorboard_mode == "disabled":
            return False
        # Auto-enable
        return True
    
    return False


def log_metrics(metrics: Dict[str, Any], args, step_key: str = "step") -> None:
    """
    Unified function to log metrics to both wandb and TensorBoard.
    
    Args:
        metrics: Dictionary of metrics to log
        args: Arguments object containing configuration
        step_key: Key in metrics dict that contains the step value
    """
    if not metrics:
        return
    
    # Log to wandb if enabled
    if getattr(args, 'use_wandb', False):
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics)
        except (ImportError, Exception) as e:
            print(f"Warning: Failed to log to wandb: {e}")
    
    # Log to TensorBoard if enabled (auto-detect or explicit)
    if _should_use_tensorboard(args):
        try:
            from slime.utils.tensorboard_utils import get_tensorboard_writer
            writer = get_tensorboard_writer()
            if writer is not None:
                # Extract step value
                step = metrics.get(step_key, 0)
                
                # Log each metric except the step key
                for key, value in metrics.items():
                    if key != step_key and isinstance(value, (int, float)):
                        writer.add_scalar(key, value, step)
        except (ImportError, Exception) as e:
            print(f"Warning: Failed to log to TensorBoard: {e}")


def log_scalar(tag: str, value: float, step: int, args) -> None:
    """
    Log a single scalar value to all enabled backends.
    
    Args:
        tag: Metric name/tag
        value: Metric value
        step: Step number
        args: Arguments object containing configuration
    """
    metrics = {tag: value, "step": step}
    log_metrics(metrics, args)


def init_logging(args) -> tuple:
    """
    Initialize all enabled logging backends.
    
    Args:
        args: Arguments object containing configuration
        
    Returns:
        Tuple of (wandb_run_id, tensorboard_run_id)
    """
    wandb_run_id = None
    tensorboard_run_id = None
    
    # Initialize wandb if enabled
    if getattr(args, 'use_wandb', False):
        try:
            from slime.utils.wandb_utils import init_wandb_primary
            wandb_run_id = init_wandb_primary(args)
        except (ImportError, Exception) as e:
            print(f"Warning: Failed to initialize wandb: {e}")
    
    # Initialize TensorBoard if enabled (auto-detect or explicit)
    if _should_use_tensorboard(args):
        try:
            from slime.utils.tensorboard_utils import init_tensorboard_primary
            tensorboard_run_id = init_tensorboard_primary(args)
        except (ImportError, Exception) as e:
            print(f"Warning: Failed to initialize TensorBoard: {e}")
    
    return wandb_run_id, tensorboard_run_id


def init_logging_secondary(args, wandb_run_id: Optional[str] = None, tensorboard_run_id: Optional[str] = None) -> None:
    """
    Initialize secondary logging for distributed training.
    
    Args:
        args: Arguments object containing configuration
        wandb_run_id: wandb run ID from primary process
        tensorboard_run_id: TensorBoard run ID from primary process
    """
    # Initialize wandb secondary if enabled
    if getattr(args, 'use_wandb', False) and wandb_run_id is not None:
        try:
            from slime.utils.wandb_utils import init_wandb_secondary
            init_wandb_secondary(args, wandb_run_id)
        except (ImportError, Exception) as e:
            print(f"Warning: Failed to initialize wandb secondary: {e}")
    
    # Initialize TensorBoard secondary if enabled (auto-detect or explicit)
    if _should_use_tensorboard(args) and tensorboard_run_id is not None:
        try:
            from slime.utils.tensorboard_utils import init_tensorboard_secondary
            init_tensorboard_secondary(args, tensorboard_run_id)
        except (ImportError, Exception) as e:
            print(f"Warning: Failed to initialize TensorBoard secondary: {e}")


def close_logging(args) -> None:
    """
    Close all logging backends.
    
    Args:
        args: Arguments object containing configuration
    """
    # Close TensorBoard if enabled (auto-detect or explicit)
    if _should_use_tensorboard(args):
        try:
            from slime.utils.tensorboard_utils import close_tensorboard
            close_tensorboard()
        except (ImportError, Exception) as e:
            print(f"Warning: Failed to close TensorBoard: {e}")
    
    # wandb doesn't need explicit closing, it handles this automatically


def log_train_metrics(rollout_id: int, step_id: int, log_dict: Dict[str, Any], args) -> None:
    """
    Log training metrics with appropriate step calculation.
    
    Args:
        rollout_id: Current rollout ID
        step_id: Current step ID within rollout
        log_dict: Dictionary of metrics to log
        args: Arguments object containing configuration
    """
    # Calculate the accumulated step ID
    num_steps_per_rollout = getattr(args, 'num_steps_per_rollout', 1)
    accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
    
    # Prepare metrics with step
    metrics = log_dict.copy()
    metrics["train/step"] = accumulated_step_id
    
    log_metrics(metrics, args, step_key="train/step")


def log_rollout_metrics(rollout_id: int, log_dict: Dict[str, Any], args) -> None:
    """
    Log rollout metrics with appropriate step calculation.
    
    Args:
        rollout_id: Current rollout ID
        log_dict: Dictionary of metrics to log
        args: Arguments object containing configuration
    """
    # Calculate step based on configuration
    if getattr(args, 'wandb_always_use_train_step', False):
        step = rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    else:
        step = rollout_id
    
    # Prepare metrics with step
    metrics = log_dict.copy()
    metrics["rollout/step"] = step
    
    log_metrics(metrics, args, step_key="rollout/step")


def log_eval_metrics(rollout_id: int, log_dict: Dict[str, Any], args) -> None:
    """
    Log evaluation metrics with appropriate step calculation.
    
    Args:
        rollout_id: Current rollout ID
        log_dict: Dictionary of metrics to log
        args: Arguments object containing configuration
    """
    # Calculate step based on configuration
    if getattr(args, 'wandb_always_use_train_step', False):
        step = rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    else:
        step = rollout_id
    
    # Prepare metrics with step
    metrics = log_dict.copy()
    metrics["eval/step"] = step
    
    log_metrics(metrics, args, step_key="eval/step")


def log_perf_metrics(rollout_id: int, log_dict: Dict[str, Any], args) -> None:
    """
    Log performance metrics with appropriate step calculation.
    
    Args:
        rollout_id: Current rollout ID
        log_dict: Dictionary of metrics to log
        args: Arguments object containing configuration
    """
    # Calculate step based on configuration
    if getattr(args, 'wandb_always_use_train_step', False):
        step = rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    else:
        step = rollout_id
    
    # Prepare metrics with step
    metrics = log_dict.copy()
    metrics["rollout/step"] = step  # Performance metrics use rollout/step
    
    log_metrics(metrics, args, step_key="rollout/step")


def log_multi_turn_metrics(rollout_id: int, log_dict: Dict[str, Any], args) -> None:
    """
    Log multi-turn dialogue metrics.
    
    Args:
        rollout_id: Current rollout ID
        log_dict: Dictionary of metrics to log
        args: Arguments object containing configuration
    """
    # Multi-turn metrics use rollout/step as step metric
    step = rollout_id
    
    # Prepare metrics with step
    metrics = log_dict.copy()
    metrics["rollout/step"] = step
    
    log_metrics(metrics, args, step_key="rollout/step")


def log_passrate_metrics(rollout_id: int, log_dict: Dict[str, Any], args) -> None:
    """
    Log pass rate metrics.
    
    Args:
        rollout_id: Current rollout ID
        log_dict: Dictionary of metrics to log
        args: Arguments object containing configuration
    """
    # Pass rate metrics use rollout/step as step metric
    step = rollout_id
    
    # Prepare metrics with step
    metrics = log_dict.copy()
    metrics["rollout/step"] = step
    
    log_metrics(metrics, args, step_key="rollout/step")
