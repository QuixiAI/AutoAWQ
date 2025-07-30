#!/bin/bash

# Script to run Kimi quantization with proper ROCm environment setup

# Activate virtual environment
source .venv/bin/activate

# Set environment variables to handle ROCm/CUDA conflicts
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
export TRANSFORMERS_OFFLINE=1
export DISABLE_FLASH_ATTN=1

# For ROCm systems, ensure we're using the right backend
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # Adjust based on your GPU
export PYTORCH_ROCM_ARCH="gfx1100"  # Adjust based on your GPU

# Disable CUDA-specific optimizations
export TORCH_CUDA_ARCH_LIST=""

echo "Starting Kimi-K2 quantization with AutoAWQ..."
echo "Environment configured for ROCm"

# Run the quantization script
python quant_kimi.py "$@"