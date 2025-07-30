# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoAWQ is a Python package for 4-bit quantization of Large Language Models (LLMs) using the Activation-aware Weight Quantization (AWQ) algorithm. The project provides 3x speedup and 3x memory reduction compared to FP16 models.

**Note**: This project is officially deprecated. Use vLLM's llm-compressor as an alternative: https://github.com/vllm-project/llm-compressor

## Common Development Commands

### Environment Setup
```bash
# Always activate the virtual environment first
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev,eval]"

# Install with GPU kernels
pip install -e ".[kernels]"

# Install for CPU (Intel)
pip install -e ".[cpu]"
```

### Running Tests
```bash
# Activate environment first
source .venv/bin/activate

# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_quantization.py

# Run with verbose output
python -m pytest -v tests/
```

### Common Tasks
```bash
# Quantize a model (example)
python examples/quantize.py

# Run inference
python examples/generate.py

# Benchmark performance
python examples/benchmark.py --model_path <hf_model> --batch_size 1

# CLI interface
python examples/cli.py
```

## Architecture Overview

### Core Components

1. **Model Architecture** (`awq/models/`)
   - `base.py`: Base class for all AWQ models
   - `auto.py`: AutoModel factory for loading quantized models
   - Model-specific implementations (llama.py, mistral.py, etc.) handle architecture differences

2. **Quantization Engine** (`awq/quantize/`)
   - `quantizer.py`: Core AWQ quantization algorithm implementation
   - `scale.py`: Scaling computations for quantization
   - Implements 4-bit weight quantization with group-wise scaling

3. **Kernel Modules** (`awq/modules/`)
   - `linear/`: Different linear layer implementations (GEMM, GEMV, ExLlama, Marlin)
   - `fused/`: Fused operations for performance (attention, MLP, normalization)
   - `triton/`: Triton-based kernels for GPU acceleration

4. **Evaluation** (`awq/evaluation/`)
   - Tools for evaluating quantized model performance
   - KL divergence measurement between original and quantized models

### Key Design Patterns

- **GEMM vs GEMV**: Two quantization versions optimized for different use cases
  - GEMV: Faster for batch size 1, memory-bound operations
  - GEMM: Better for larger batch sizes, compute-bound operations

- **Fused Modules**: Combine multiple operations into single kernels for efficiency
  - Enabled with `fuse_layers=True` during model loading
  - Requires Linux, uses FasterTransformer optimizations

- **Multi-backend Support**: 
  - CUDA (primary)
  - ROCm (via ExLlamaV2 kernels)
  - Intel CPU/GPU (via Intel Extension for PyTorch)

### Model Support

The package supports 30+ model architectures including Llama, Mistral, Falcon, MPT, and others. Each model has its own module in `awq/models/` that handles architecture-specific details while inheriting from the base quantization framework.

### Quantization Process

1. Load pretrained model
2. Generate calibration data
3. Compute activation statistics
4. Apply AWQ algorithm to find optimal quantization parameters
5. Pack weights into INT4 format
6. Save quantized model

The quantization preserves model quality while reducing memory usage by ~3x and improving inference speed.