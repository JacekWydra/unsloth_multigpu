# Multi-GPU Testing Guide

Simple guide for testing multi-GPU functionality on GCP.

## Setup GCP Instances

### 1. Create Instances

```bash
# For 2-GPU testing
./scripts/create_gcp_instance.sh your-project-id 2gpu

# For 4-GPU testing  
./scripts/create_gcp_instance.sh your-project-id 4gpu

# For both
./scripts/create_gcp_instance.sh your-project-id both
```

### 2. SSH to Instances

```bash
# 2-GPU instance
gcloud compute ssh unsloth-multigpu-test-2gpu --zone=europe-central2-a --project=your-project-id

# 4-GPU instance
gcloud compute ssh unsloth-multigpu-test-4gpu --zone=europe-central2-a --project=your-project-id
```

### 3. Setup Environment

```bash
# Copy setup script to instance (if needed)
gcloud compute scp scripts/setup_instance.sh unsloth-multigpu-test-2gpu:~ --zone=europe-central2-a --project=your-project-id

# Run setup
./setup_instance.sh
```

## Running Tests

### Simple Tests

```bash
# Single GPU test
python -m unsloth_multigpu.train --config tests/debug_config.yaml

# 2-GPU test
python -m unsloth_multigpu.train --config tests/multigpu_2gpu_config.yaml

# 4-GPU test  
python -m unsloth_multigpu.train --config tests/multigpu_4gpu_config.yaml
```

### Automated Test Suite

```bash
# Run all available tests
python scripts/run_multigpu_tests.py --tests all

# Run specific tests
python scripts/run_multigpu_tests.py --tests single 2gpu

# With timeout
python scripts/run_multigpu_tests.py --tests all --timeout 300
```

### Monitor GPUs

```bash
# Show current GPU status
python scripts/simple_gpu_monitor.py

# Start monitoring
python scripts/simple_gpu_monitor.py --monitor --interval 2

# Monitor for specific duration
python scripts/simple_gpu_monitor.py --monitor --duration 60
```

## Test Configurations

- **Single GPU**: `tests/debug_config.yaml` - 3 steps, basic functionality
- **2-GPU**: `tests/multigpu_2gpu_config.yaml` - 10 steps, 2-GPU coordination
- **4-GPU**: `tests/multigpu_4gpu_config.yaml` - 12 steps, 4-GPU scaling

All use the expanded debug dataset with 20 training samples and 8 validation samples.

## Expected Results

### Successful Test Indicators

1. **Process Isolation**: Each GPU shows in separate subprocess logs
2. **DDP Coordination**: Logs show distributed training initialization
3. **Training Progress**: Loss values decrease over steps
4. **Clean Exit**: Processes exit with code 0

### Log Messages to Look For

```
[CUSTOM GPU 0] === CUSTOM WORKER PROCESS 0 STARTING ON GPU 0 ===
[CUSTOM GPU 0] ✅ Proper device isolation achieved!
[CUSTOM GPU 1] === CUSTOM WORKER PROCESS 1 STARTING ON GPU 1 ===  
[CUSTOM GPU 1] ✅ Proper device isolation achieved!
...
✅ Training completed successfully!
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `per_device_train_batch_size` in config
2. **DDP Timeout**: Check network connectivity between GPUs
3. **Import Errors**: Ensure all dependencies installed via `setup_instance.sh`

### Debug Commands

```bash
# Check GPU status
nvidia-smi

# Check CUDA
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# Test DDP
python -c "import torch.distributed as dist; print('DDP available')"
```

## Cleanup

### Delete Instances

```bash
# Delete specific instance
gcloud compute instances delete unsloth-multigpu-test-2gpu --zone=europe-central2-a --project=your-project-id

# Delete both
gcloud compute instances delete unsloth-multigpu-test-2gpu unsloth-multigpu-test-4gpu --zone=europe-central2-a --project=your-project-id
```

## Cost Estimation

- **2-GPU Instance (T4)**: ~$1.50/hour
- **4-GPU Instance (T4)**: ~$3.00/hour
- **Expected Test Duration**: 10-20 minutes per instance

Total estimated cost: ~$10-15 for complete testing.