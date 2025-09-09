import os
from typing import Dict, Any, Optional
import wandb


class UnifiedLogger:
    """Unified logging interface that supports both wandb and tensorboard."""
    
    def __init__(self, use_wandb: bool = False, use_tensorboard: bool = False, 
                 tensorboard_log_dir: Optional[str] = None):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        # Initialize tensorboard if requested
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
            except ImportError:
                print("Warning: torch.utils.tensorboard not available. Tensorboard logging disabled.")
                self.use_tensorboard = False
    
    def log_metric(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to both wandb and tensorboard if enabled.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for tensorboard
        """
        # Log to wandb (existing behavior)
        if self.use_wandb:
            wandb.log(metrics)
        
        # Log to tensorboard
        if self.use_tensorboard and self.writer is not None:
            for key, value in metrics.items():
                # Handle different metric types
                if isinstance(value, (int, float)):
                    # Extract step from metrics if not provided
                    if step is None:
                        # Try to find step in the metrics
                        if "train/step" in metrics:
                            step = metrics["train/step"]
                        elif "rollout/step" in metrics:
                            step = metrics["rollout/step"]
                        elif "eval/step" in metrics:
                            step = metrics["eval/step"]
                        else:
                            step = 0  # Default step
                    
                    self.writer.add_scalar(key, value, step)
    
    def close(self):
        """Close the tensorboard writer if it exists."""
        if self.writer is not None:
            self.writer.close()


# Global logger instance
_global_logger: Optional[UnifiedLogger] = None


def init_logger(use_wandb: bool = False, use_tensorboard: bool = False, 
                tensorboard_log_dir: Optional[str] = None):
    """Initialize the global logger instance."""
    global _global_logger
    _global_logger = UnifiedLogger(
        use_wandb=use_wandb, 
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir=tensorboard_log_dir
    )


def log_metric(metrics: Dict[str, Any], step: Optional[int] = None):
    """
    Log metrics using the global logger instance.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number for tensorboard
    """
    if _global_logger is not None:
        _global_logger.log_metric(metrics, step)
    else:
        # Fallback to wandb if global logger not initialized
        if wandb.run is not None:
            wandb.log(metrics)


def close_logger():
    """Close the global logger."""
    global _global_logger
    if _global_logger is not None:
        _global_logger.close()
        _global_logger = None
