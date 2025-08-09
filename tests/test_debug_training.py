#!/usr/bin/env python3
"""
Integration test for the training pipeline.

Tests that the training pipeline can:
1. Load configuration
2. Initialize model and dataset
3. Run a few training steps
4. Handle both single and multi-GPU setups
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import pytest
import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from unsloth_multigpu.utils import load_training_config
from unsloth_multigpu.data_model.data_model import TrainingConfig


def get_test_config_path():
    """Get path to debug config file."""
    return Path(__file__).parent / "debug_config.yaml"


def create_test_config(output_dir, num_gpus=1):
    """Create a test configuration with specified settings."""
    config_path = get_test_config_path()
    
    # Load base config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Update output directory
    config_data['trainer_config']['output_dir'] = str(output_dir)
    
    # Update devices based on available GPUs
    if num_gpus == 1:
        config_data['devices'] = [0]
    else:
        config_data['devices'] = list(range(num_gpus))
    
    # Write temporary config
    temp_config = output_dir / "test_config.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config_data, f)
    
    return str(temp_config)


class TestDebugTraining:
    """Test suite for debug training pipeline."""
    
    def test_config_loading(self):
        """Test that configuration loads correctly."""
        config_path = get_test_config_path()
        config = load_training_config(str(config_path))
        
        assert isinstance(config, TrainingConfig)
        assert config.dataset_config.use_debug_dataset is True
        assert config.trainer_config.max_steps == 3
        assert config.train_on_responses is True
    
    def test_single_gpu_training(self, tmp_path):
        """Test training on single GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create test config
        config_path = create_test_config(tmp_path, num_gpus=1)
        
        # Run training
        cmd = [
            sys.executable, "-m", "unsloth_multigpu.train",
            "--config", config_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Check that training completed
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        assert "Training completed" in result.stdout or "steps completed" in result.stdout.lower()
    
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2+ GPUs")
    def test_multi_gpu_training(self, tmp_path):
        """Test training on multiple GPUs."""
        # Create test config
        num_gpus = min(2, torch.cuda.device_count())
        config_path = create_test_config(tmp_path, num_gpus=num_gpus)
        
        # Run training
        cmd = [
            sys.executable, "-m", "unsloth_multigpu.train",
            "--config", config_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Check that training completed
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        assert "Training completed" in result.stdout or "steps completed" in result.stdout.lower()
        
        # Verify multiple processes were spawned
        assert f"world_size={num_gpus}" in result.stdout.lower() or f"rank" in result.stdout.lower()
    
    def test_dataset_loading(self):
        """Test that debug dataset loads correctly."""
        from unsloth_multigpu.dataset.select_dataset import get_training_dataset
        from unsloth_multigpu.data_model.data_model import DatasetConfig
        
        # Create config with debug dataset
        config = DatasetConfig(use_debug_dataset=True)
        
        # Load dataset (tokenizer not needed for debug dataset)
        dataset = get_training_dataset(config, tokenizer=None)
        
        assert "train" in dataset
        assert "validation" in dataset
        assert len(dataset["train"]) > 0
        assert len(dataset["validation"]) > 0
        
        # Check data format
        sample = dataset["train"][0]
        assert "text" in sample
        assert "<|start_header_id|>user<|end_header_id|>" in sample["text"]
        assert "<|start_header_id|>assistant<|end_header_id|>" in sample["text"]
    
    def test_custom_dataset_error(self):
        """Test that custom dataset raises appropriate error when not implemented."""
        from unsloth_multigpu.dataset.select_dataset import get_training_dataset
        from unsloth_multigpu.data_model.data_model import DatasetConfig
        
        # Create config with custom dataset
        config = DatasetConfig(use_debug_dataset=False)
        
        # Should raise NotImplementedError
        with pytest.raises((NotImplementedError, ImportError)):
            get_training_dataset(config, tokenizer=None)


def test_quick_smoke():
    """Quick smoke test that imports work."""
    from unsloth_multigpu.train import main
    from unsloth_multigpu.worker_process import setup_distributed_environment
    from unsloth_multigpu.multigpu_trainer import MultiGPUTrainer
    
    # Just verify imports work
    assert main is not None
    assert setup_distributed_environment is not None
    assert MultiGPUTrainer is not None
    print("✓ All imports successful")


if __name__ == "__main__":
    # Run quick smoke test
    test_quick_smoke()
    
    # Run pytest if available
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        print("Install pytest to run full test suite: pip install pytest")
        print("Quick smoke test passed!")