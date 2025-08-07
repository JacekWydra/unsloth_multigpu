# Unsloth MultiGPU

Multi-GPU training support for Unsloth - Fast and memory-efficient LLM training.

## Overview

`unsloth_multigpu` is a Python library that extends Unsloth's capabilities to support distributed training across multiple GPUs, enabling faster training of large language models while maintaining Unsloth's memory efficiency.

## Features

- Multi-GPU distributed training support for Unsloth
- Memory-efficient training strategies
- Easy integration with existing Unsloth workflows
- Support for various parallelization strategies

## Installation

### From Source

```bash
git clone https://github.com/jacek.wydra1992/unsloth_multigpu.git
cd unsloth_multigpu
pip install -e .
```

### For Development

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import unsloth_multigpu

# Your implementation here
```

## Requirements

- Python >= 3.8
- PyTorch with CUDA support
- Unsloth

## Project Structure

```
unsloth_multigpu/
├── unsloth_multigpu/       # Main package directory
│   └── __init__.py         # Package initialization
├── pyproject.toml          # Project configuration
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
├── .gitignore             # Git ignore patterns
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Author

Jacek Wydra - jacek.wydra1992@gmail.com

## Acknowledgments

- Built on top of [Unsloth](https://github.com/unslothai/unsloth) for efficient LLM training