# Unsloth MultiGPU

Multi-GPU training extension for Unsloth using process isolation to prevent memory multiplication issues.

## Overview

`unsloth_multigpu` provides a distributed training framework that extends Unsloth's capabilities to multiple GPUs. The core innovation is using **process isolation** - each GPU runs in its own subprocess with isolated CUDA contexts - to address memory multiplication problems that can occur when using Unsloth's memory optimizations with standard PyTorch DDP.

## Key Features

### ✅ **Currently Working**
- **Process-Isolated Multi-GPU Training**: Each GPU runs in its own subprocess to prevent memory conflicts
- **Single GPU Training**: Fully functional with Unsloth integration and LoRA support
- **Multi-GPU Coordination**: Fully functional with Unsloth integration and LoRA support
- **Configuration Management**: YAML-based configuration with Pydantic validation
- **Debug Dataset**: Built-in test dataset with Llama-3 Instruct formatting
- **Training Pipeline**: Complete training workflow with logging and monitoring
- **Memory Monitoring**: Real-time GPU memory usage tracking

### ⚠️ **Partially Implemented**
- **Custom Datasets**: Template provided but requires user implementation

### 🚧 **Known Limitations**
- **Evaluation Metrics**: Metrics computation returns empty results (needs implementation)
- **Production Testing**: Limited testing at scale with large datasets/models

## Architecture

The system uses a **master-worker pattern**:

1. **Master Process**: Orchestrates training, spawns worker subprocesses, monitors progress
2. **Worker Processes**: Each handles one GPU with isolated CUDA context, communicates via PyTorch DDP
3. **Custom Trainer**: Extends UnslothTrainer with proper gradient synchronization control

## Installation

### From Source
```bash
git clone https://github.com/JacekWydra/unsloth_multigpu.git
cd unsloth_multigpu
pip install -e .
```

### For Development
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Basic Training (Single GPU)
```bash
# Set configuration path
export CONFIG_PATH=tests/debug_config.yaml

# Run training
python -m unsloth_multigpu.train
```

### 2. Configuration Example
```yaml
devices: [0]  # GPU IDs to use
train_on_responses: true
trainer_config:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: 3
  learning_rate: 2e-4
  output_dir: ./debug_output
pretrained_model_config:
  model_name: unsloth/Llama-3.2-1B-Instruct
  max_seq_length: 512
dataset_config:
  use_debug_dataset: true  # Use built-in debug dataset
```

### 3. Custom Dataset Implementation
To use your own data, implement the dataset loader:

```python
# In unsloth_multigpu/dataset/custom_dataset.py
def get_dataset(config: DatasetConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    # Load your training and validation data
    # Format as Llama-3 Instruct: <|begin_of_text|><|start_header_id|>...
    # Return DatasetDict with 'train' and 'validation' splits
    pass
```

Then set `use_debug_dataset: false` in your config.

## Requirements

- Python >= 3.8
- PyTorch with CUDA support  
- Unsloth >= 2025.5.7
- transformers, datasets, pydantic
- Multiple NVIDIA GPUs for multi-GPU training

## Project Status

This is a **proof-of-concept** implementation that demonstrates process isolation for multi-GPU Unsloth training. While the core architecture is sound and single-GPU training works reliably, the project needs:

- [ ] Comprehensive multi-GPU testing at scale
- [ ] Performance benchmarking vs. standard DDP approaches  
- [ ] Production-ready dataset handling
- [ ] Broader model compatibility testing
- [ ] Evaluation metrics implementation

## Contributing

We welcome contributions! Please see:

1. Run formatters: `black unsloth_multigpu --line-length 100`, `isort unsloth_multigpu --profile black`
2. Run linters: `flake8 unsloth_multigpu`, `mypy unsloth_multigpu`  
3. Run tests: `pytest tests/`

## Testing

```bash
# Run debug training test
python tests/test_debug_training.py

# Run with pytest (requires pytest installation)
pytest tests/ -v
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
## Acknowledgments

Built on top of [Unsloth](https://github.com/unslothai/unsloth) for fast and memory-efficient LLM training.

## Disclaimer

This project addresses a specific memory multiplication issue encountered when using Unsloth with multi-GPU setups. The effectiveness compared to other approaches has not been empirically validated. Use at your own discretion and test thoroughly for your specific use case.