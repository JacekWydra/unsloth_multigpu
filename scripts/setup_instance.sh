#!/bin/bash
# Setup script for GCP instances - installs dependencies and prepares environment
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Setting up GCP instance for Unsloth MultiGPU testing ===${NC}"

# Update system
echo -e "${YELLOW}Updating system packages...${NC}"
sudo apt-get update

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
sudo apt-get install -y \
    git \
    wget \
    curl \
    htop \
    nvidia-smi \
    tmux \
    vim

# Check GPU setup
echo -e "${YELLOW}Checking GPU setup...${NC}"
nvidia-smi
echo ""

# Check CUDA
echo -e "${YELLOW}Checking CUDA installation...${NC}"
nvcc --version || echo "CUDA not found in PATH"
echo ""

# Install miniconda if not present
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}Installing Miniconda...${NC}"
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    rm Miniconda3-latest-Linux-x86_64.sh
fi

# Source conda
source $HOME/miniconda3/bin/activate

# Create conda environment
echo -e "${YELLOW}Creating conda environment...${NC}"
conda create -n unsloth python=3.10 -y
conda activate unsloth

# Install PyTorch with CUDA
echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install Unsloth
echo -e "${YELLOW}Installing Unsloth...${NC}"
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"

# Install additional dependencies
echo -e "${YELLOW}Installing additional dependencies...${NC}"
pip install \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    pydantic \
    PyYAML \
    psutil \
    GPUtil \
    matplotlib \
    seaborn \
    pytest

# Clone repository
if [ ! -d "unsloth_multigpu" ]; then
    echo -e "${YELLOW}Cloning Unsloth MultiGPU repository...${NC}"
    git clone https://github.com/JacekWydra/unsloth_multigpu.git
fi

cd unsloth_multigpu

# Install package in development mode
echo -e "${YELLOW}Installing Unsloth MultiGPU package...${NC}"
pip install -e .

# Create workspace directories
echo -e "${YELLOW}Creating workspace directories...${NC}"
mkdir -p ~/workspace/test_results
mkdir -p ~/workspace/logs
mkdir -p ~/workspace/models

# Create convenience scripts
echo -e "${YELLOW}Creating convenience scripts...${NC}"

# Script to activate environment and run tests
cat > ~/run_tests.sh << 'EOF'
#!/bin/bash
source $HOME/miniconda3/bin/activate
conda activate unsloth

cd ~/unsloth_multigpu

echo "=== Running Multi-GPU Tests ==="
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Current directory: $(pwd)"
echo ""

# Test single GPU first
echo "Testing single GPU..."
python -m unsloth_multigpu.train --config tests/debug_config.yaml

# Detect GPU count and run appropriate multi-GPU test
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Detected $GPU_COUNT GPUs"

if [ $GPU_COUNT -ge 2 ]; then
    echo "Testing 2-GPU configuration..."
    python -m unsloth_multigpu.train --config tests/multigpu_2gpu_config.yaml
fi

if [ $GPU_COUNT -ge 4 ]; then
    echo "Testing 4-GPU configuration..."
    python -m unsloth_multigpu.train --config tests/multigpu_4gpu_config.yaml
fi

echo "=== Tests Complete ==="
EOF

chmod +x ~/run_tests.sh

# Script to monitor GPU usage
cat > ~/monitor_gpus.sh << 'EOF'
#!/bin/bash
echo "GPU Monitoring - Press Ctrl+C to stop"
while true; do
    clear
    date
    echo "=================================="
    nvidia-smi
    echo "=================================="
    echo "Memory Usage:"
    free -h
    echo "=================================="
    sleep 2
done
EOF

chmod +x ~/monitor_gpus.sh

# Create tmux session setup script
cat > ~/setup_tmux.sh << 'EOF'
#!/bin/bash
# Create tmux session for multi-GPU testing
tmux new-session -d -s multigpu
tmux send-keys -t multigpu 'source $HOME/miniconda3/bin/activate' C-m
tmux send-keys -t multigpu 'conda activate unsloth' C-m
tmux send-keys -t multigpu 'cd ~/unsloth_multigpu' C-m

# Split window for monitoring
tmux split-window -h -t multigpu
tmux send-keys -t multigpu:0.1 '~/monitor_gpus.sh' C-m

# Select left pane for commands
tmux select-pane -t multigpu:0.0

echo "Tmux session 'multigpu' created"
echo "Attach with: tmux attach -t multigpu"
EOF

chmod +x ~/setup_tmux.sh

# Set up environment activation in .bashrc
echo -e "${YELLOW}Setting up environment activation...${NC}"
cat >> ~/.bashrc << 'EOF'

# Auto-activate unsloth environment
if [ -f "$HOME/miniconda3/bin/activate" ]; then
    source $HOME/miniconda3/bin/activate
    conda activate unsloth 2>/dev/null || true
fi

# Add convenience aliases
alias gpu='nvidia-smi'
alias gpuwatch='watch -n1 nvidia-smi'
alias runtest='~/run_tests.sh'
alias monitorgpu='~/monitor_gpus.sh'
alias setupscreen='~/setup_tmux.sh'

echo "Unsloth MultiGPU environment ready!"
echo "Commands:"
echo "  runtest     - Run multi-GPU tests"
echo "  monitorgpu  - Monitor GPU usage"
echo "  setupscreen - Setup tmux session"
echo "  gpu         - Show GPU status"
EOF

# Test installation
echo -e "${YELLOW}Testing installation...${NC}"
cd ~/unsloth_multigpu
python -c "
import torch
import unsloth
from unsloth_multigpu.utils import load_training_config
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'Unsloth MultiGPU: Package imported successfully')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Source environment: source ~/.bashrc"
echo "2. Run tests: ~/run_tests.sh"
echo "3. Monitor GPUs: ~/monitor_gpus.sh"
echo "4. Setup tmux: ~/setup_tmux.sh"
echo ""
echo "Repository location: ~/unsloth_multigpu"
echo "Test results will be saved to: ~/workspace/test_results"