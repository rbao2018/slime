import os
import threading
from typing import Dict, Any, Optional


class UnifiedLogger:
    """Unified logging interface that supports both wandb and tensorboard."""
    
    def __init__(self, args):
        self.use_tensorboard = False if getattr(args, "tensorboard_dir", None) is None else True
        print("UnifiedLogger: use_tensorboard =", self.use_tensorboard, flush=True)
        self.args = args
        self.writer = None
        
        # Initialize tensorboard if requested
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=args.tensorboard_dir)
            except ImportError:
                print("Warning: torch.utils.tensorboard not available. Tensorboard logging disabled.", flush=True)
                self.use_tensorboard = False
    
    def log_metric(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to both wandb and tensorboard if enabled.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for tensorboard
        """
        # Extract step once if not provided
        if step is None:
            if "train/step" in metrics:
                step = metrics["train/step"]
            elif "rollout/step" in metrics:
                step = metrics["rollout/step"]
            elif "eval/step" in metrics:
                step = metrics["eval/step"]
            else:
                step = 0  # Default step
        
        # Log to tensorboard
        if self.use_tensorboard and self.writer is not None:
            logged_metrics = []
            for key, value in metrics.items():
                # Skip step metrics to avoid duplicate logging
                if key.endswith('/step'):
                    continue
                
                logged_metrics.append(f"{key}={value}")    
                # Handle different metric types
                if isinstance(value, (int, float)):
                    try:
                        self.writer.add_scalar(key, value, step)
                    except Exception as e:
                        print(f"Warning: Failed to log {key} to tensorboard: {e}", flush=True)
            
            self.writer.flush()
            
            # 立即刷新并打印确认信息
            if getattr(self.args, "tensorboard_log_interval", 0) > 0:
                try:
                    print(f"TensorBoard: Logged {len(logged_metrics)} metrics at step {step}", flush=True)
                except Exception as e:
                    print(f"Warning: Failed to flush tensorboard writer: {e}", flush=True)
            
    
    def close(self):
        """Close the tensorboard writer if it exists."""
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as e:
                print(f"Warning: Failed to close tensorboard writer: {e}", flush=True)
    

# 简化版本，移除锁
_global_logger: Optional[UnifiedLogger] = None

def init_logger(args):
    """Initialize the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = UnifiedLogger(args=args)

def log_metric(metrics: Dict[str, Any], step: Optional[int] = None):
    """Log metrics using the global logger instance."""
    if _global_logger is not None:
        _global_logger.log_metric(metrics, step)
