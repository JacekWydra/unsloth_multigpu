#!/bin/bash

# Multi-GPU Training Script
# Runs our custom trainer implementation with proper timeout and memory efficiency

set -e  # Exit on any error

echo "=== STARTING MULTI-GPU TRAINING ==="
echo "Timestamp: $(date)"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "Loading environment variables..."
    export $(cat .env | xargs)
fi

# Set configuration path (use default if not set)
export CONFIG_PATH="${CONFIG_PATH:-configs/example_config.yaml}"
echo "Config path: $CONFIG_PATH"

# Run training
echo "Starting training..."
python -m unsloth_multigpu.train

echo "=== MULTI-GPU TRAINING COMPLETED SUCCESSFULLY ==="
echo "Timestamp: $(date)"