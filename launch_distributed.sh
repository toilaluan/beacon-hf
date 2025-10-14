#!/bin/bash

# Distributed Training Launch Script for PyTorch
# This script launches distributed training across multiple GPUs

# Number of GPUs to use (adjust based on your setup)
NUM_GPUS=${1:-$(nvidia-smi -L | wc -l)}

echo "Launching distributed training on $NUM_GPUS GPUs..."

# Launch with torchrun (recommended for PyTorch 1.9+)
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    seed.py

# Alternative: Launch with torch.distributed.launch (older PyTorch versions)
# python -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --use_env \
#     seed.py

echo "Training complete!"

