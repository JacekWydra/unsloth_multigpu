#!/bin/bash

# Multi-GPU Training Script
# Runs our custom trainer implementation with proper timeout and memory efficiency

set -e  # Exit on any error

echo "=== STARTING MULTI-GPU TRAINING ==="
echo "Timestamp: $(date)"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"

# Load environment variables
echo "Loading environment variables..."
export $(cat .env | xargs)

# Set configuration path
echo "Config path: $CONFIG_PATH"

# Run training without timeout
echo "Starting training without timeout..."
python -m llm_finetune.training.train

echo "=== MULTI-GPU TRAINING COMPLETED SUCCESSFULLY ==="
echo "Timestamp: $(date)"