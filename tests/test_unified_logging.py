#!/usr/bin/env python3
"""Test script for unified logging interface."""

import os
import tempfile
import shutil
from unittest.mock import Mock

# Mock wandb to avoid requiring actual wandb setup
import sys
wandb_mock = Mock()
wandb_mock.run = None
wandb_mock.log = Mock()
sys.modules['wandb'] = wandb_mock

from slime.utils.logger_utils import UnifiedLogger, init_logger, log_metric


def test_wandb_only():
    """Test logging with wandb only."""
    print("Testing wandb-only logging...")
    logger = UnifiedLogger(use_wandb=True, use_tensorboard=False)
    
    test_metrics = {
        "train/loss": 0.5,
        "train/step": 100,
        "eval/accuracy": 0.9
    }
    
    logger.log_metric(test_metrics)
    wandb_mock.log.assert_called_with(test_metrics)
    print("✓ Wandb logging works")


def test_tensorboard_only():
    """Test logging with tensorboard only."""
    print("Testing tensorboard-only logging...")
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = UnifiedLogger(use_wandb=False, use_tensorboard=True, 
                              tensorboard_log_dir=temp_dir)
        
        test_metrics = {
            "train/loss": 0.3,
            "train/step": 200,
            "eval/accuracy": 0.95
        }
        
        logger.log_metric(test_metrics)
        logger.close()
        
        # Check if tensorboard files were created
        files = os.listdir(temp_dir)
        assert any(f.startswith('events.out.tfevents') for f in files), "Tensorboard files not created"
        print("✓ Tensorboard logging works")


def test_unified_logging():
    """Test logging with both wandb and tensorboard."""
    print("Testing unified logging...")
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = UnifiedLogger(use_wandb=True, use_tensorboard=True,
                              tensorboard_log_dir=temp_dir)
        
        test_metrics = {
            "train/loss": 0.2,
            "train/step": 300,
            "eval/accuracy": 0.98
        }
        
        logger.log_metric(test_metrics)
        logger.close()
        
        # Check wandb was called
        wandb_mock.log.assert_called_with(test_metrics)
        
        # Check tensorboard files were created
        files = os.listdir(temp_dir)
        assert any(f.startswith('events.out.tfevents') for f in files), "Tensorboard files not created"
        print("✓ Unified logging works")


def test_global_logger():
    """Test global logger interface."""
    print("Testing global logger interface...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize global logger
        init_logger(use_wandb=True, use_tensorboard=True, 
                   tensorboard_log_dir=temp_dir)
        
        test_metrics = {
            "train/loss": 0.1,
            "train/step": 400,
            "eval/accuracy": 0.99
        }
        
        # Use global logging function
        log_metric(test_metrics)
        
        # Check wandb was called
        wandb_mock.log.assert_called_with(test_metrics)
        print("✓ Global logger interface works")


def main():
    """Run all tests."""
    print("Running unified logging tests...\n")
    
    try:
        test_wandb_only()
        test_tensorboard_only()
        test_unified_logging()
        test_global_logger()
        
        print("\n🎉 All tests passed! The unified logging interface is working correctly.")
        print("\nUsage examples:")
        print("1. Enable both: --use-wandb --use-tensorboard --tensorboard-log-dir ./tb_logs")
        print("2. Wandb only: --use-wandb")
        print("3. Tensorboard only: --use-tensorboard --tensorboard-log-dir ./tb_logs")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
