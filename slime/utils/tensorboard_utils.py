import os
from torch.utils.tensorboard import SummaryWriter


def _is_disabled_mode(args) -> bool:
    """Detect whether TensorBoard should be disabled.

    Priority order:
    1) args.tensorboard_mode if provided
    2) TENSORBOARD_MODE environment variable
    """
    if hasattr(args, 'tensorboard_mode') and args.tensorboard_mode:
        return args.tensorboard_mode == "disabled"
    return os.environ.get("TENSORBOARD_MODE") == "disabled"


def init_tensorboard_primary(args):
    if not getattr(args, 'use_tensorboard', False):
        return None

    # Set TensorBoard mode if specified (overrides TENSORBOARD_MODE env var)
    if hasattr(args, 'tensorboard_mode') and args.tensorboard_mode:
        os.environ["TENSORBOARD_MODE"] = args.tensorboard_mode
        if args.tensorboard_mode == "disabled":
            print("TensorBoard disabled mode enabled. No data will be logged.")
            return None

    disabled = _is_disabled_mode(args)
    if disabled:
        return None

    # Get directory from environment variable or args
    tensorboard_dir = os.environ.get("TENSORBOARD_DIR")
    if tensorboard_dir is None and hasattr(args, 'tensorboard_dir') and args.tensorboard_dir:
        tensorboard_dir = args.tensorboard_dir
    
    if tensorboard_dir is None:
        # Default directory structure similar to wandb
        project_name = getattr(args, 'tensorboard_project', getattr(args, 'wandb_project', 'slime'))
        group_name = getattr(args, 'tensorboard_group', getattr(args, 'wandb_group', 'default'))
        
        # Add random suffix if enabled
        if getattr(args, 'tensorboard_random_suffix', getattr(args, 'wandb_random_suffix', True)):
            import uuid
            group_name = f"{group_name}_{str(uuid.uuid4())[:6]}"
        
        tensorboard_dir = f"tensorboard_logs/{project_name}/{group_name}"

    # Ensure directory exists
    os.makedirs(tensorboard_dir, exist_ok=True)
    print(f"TensorBoard logs will be stored in: {tensorboard_dir}")

    # Create the writer
    writer = SummaryWriter(tensorboard_dir)
    
    # Store the writer globally for access by secondary processes
    global _primary_writer
    _primary_writer = writer
    
    # Generate a run ID for distributed training (similar to wandb)
    run_id = tensorboard_dir.replace('/', '_').replace('\\', '_')
    
    _init_tensorboard_common()
    
    return run_id


def init_tensorboard_secondary(args, tensorboard_run_id):
    if tensorboard_run_id is None or not getattr(args, 'use_tensorboard', False):
        return

    disabled = _is_disabled_mode(args)
    if disabled:
        return

    # For secondary processes, we don't create a new writer
    # Instead, we rely on the primary process to handle logging
    # This is different from wandb's approach but simpler for TensorBoard
    global _secondary_run_id
    _secondary_run_id = tensorboard_run_id


def _init_tensorboard_common():
    """Initialize common TensorBoard settings.
    
    Note: TensorBoard doesn't have explicit metric definitions like wandb,
    but we can document the expected metric structure here.
    
    Expected metric structure:
    - train/* metrics with train/step as the step metric
    - rollout/* metrics with rollout/step as the step metric
    - eval/* metrics with eval/step as the step metric
    - perf/* metrics with rollout/step as the step metric
    - multi_turn/* metrics with rollout/step as the step metric
    - passrate/* metrics with rollout/step as the step metric
    """
    pass


def get_tensorboard_writer():
    """Get the TensorBoard writer for logging."""
    global _primary_writer
    return getattr(globals(), '_primary_writer', None)


def log_scalar(tag, value, step):
    """Log a scalar value to TensorBoard."""
    writer = get_tensorboard_writer()
    if writer is not None:
        writer.add_scalar(tag, value, step)


def log_scalars(main_tag, tag_scalar_dict, step):
    """Log multiple scalars under a main tag to TensorBoard."""
    writer = get_tensorboard_writer()
    if writer is not None:
        writer.add_scalars(main_tag, tag_scalar_dict, step)


def log_dict(log_dict, step_key="step"):
    """Log a dictionary of metrics to TensorBoard, similar to wandb.log()."""
    writer = get_tensorboard_writer()
    if writer is None:
        return
    
    # Extract step from log_dict
    step = log_dict.get(step_key, 0)
    
    # Log each metric
    for key, value in log_dict.items():
        if key != step_key:  # Skip the step key itself
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, step)
            # TensorBoard can handle other types like histograms, images, etc.
            # but for now we focus on scalars to match wandb usage


def close_tensorboard():
    """Close the TensorBoard writer."""
    writer = get_tensorboard_writer()
    if writer is not None:
        writer.close()


def get_tensorboard_log_dir(args):
    """Get the directory where TensorBoard logs are stored."""
    if _is_disabled_mode(args):
        return None
    
    tensorboard_dir = os.environ.get("TENSORBOARD_DIR")
    if tensorboard_dir is None and hasattr(args, 'tensorboard_dir') and args.tensorboard_dir:
        tensorboard_dir = args.tensorboard_dir
    
    if tensorboard_dir is None:
        project_name = getattr(args, 'tensorboard_project', getattr(args, 'wandb_project', 'slime'))
        group_name = getattr(args, 'tensorboard_group', getattr(args, 'wandb_group', 'default'))
        tensorboard_dir = f"tensorboard_logs/{project_name}/{group_name}"
    
    return tensorboard_dir


# Global variables to store writer and run_id
_primary_writer = None
_secondary_run_id = None
