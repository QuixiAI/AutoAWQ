"""
Quantization script for Kimi-K2 with AutoAWQ
Kimi-K2 is based on DeepSeek architecture with some modifications
"""

import os
import sys
import torch

# Enable ROCm Triton backend for flash attention
os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

# Now import AWQ and transformers
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Monkey patch to handle Kimi/DeepSeek rotary embeddings
def patch_kimi_rotary():
    """Patch the quantizer to handle Kimi models without rotary_emb issues"""
    import awq.quantize.quantizer
    
    original_quantize = awq.quantize.quantizer.AwqQuantizer.quantize
    
    def patched_quantize(self):
        # Temporarily set transformers version to bypass rotary_emb check
        import transformers
        original_version = transformers.__version__
        
        # Check if this is a Kimi/DeepSeek model
        model_type = getattr(self.awq_model, 'model_type', '')
        model_config = getattr(self.awq_model, 'config', None)
        
        # Check for both deepseek and kimi model types
        if ('deepseek' in model_type.lower() or 
            'kimi' in model_type.lower() or
            (model_config and hasattr(model_config, 'model_type') and 
             ('deepseek' in str(model_config.model_type).lower() or 
              'kimi' in str(model_config.model_type).lower()))):
            # Temporarily downgrade version to skip rotary_emb computation
            transformers.__version__ = "4.47.0"
            print("Applied Kimi/DeepSeek rotary embedding patch")
        
        try:
            # Call original quantize method
            original_quantize(self)
        finally:
            # Restore original version
            transformers.__version__ = original_version
    
    awq.quantize.quantizer.AwqQuantizer.quantize = patched_quantize

# Apply the patch before loading the model
patch_kimi_rotary()

# Model paths
model_path = os.path.expanduser('~/models/Kimi-K2-Instruct-BF16')
quant_path = os.path.expanduser('~/models/Kimi-K2-Instruct-AWQ')

# Quantization config
# Note: Kimi-K2 has more experts (384 vs 256) and larger vocabulary (163840 vs 129280)
# Using smaller group size to better handle the increased number of experts
quant_config = { 
    "zero_point": True, 
    "q_group_size": 64,  # Same as DeepSeek
    "w_bit": 4, 
    "version": "GEMM"  # GEMM is recommended for MoE models
}

print(f"Loading Kimi-K2 model from {model_path}...")
print("Note: This is a large model with 61 shards, loading may take some time...")

try:
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        safetensors=True,  # Use safetensors format
        device_map="cpu",  # Load on CPU first to save memory
        trust_remote_code=True,  # Kimi may have custom code
        torch_dtype=torch.bfloat16,  # Preserve BF16 precision
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=False  # In case tiktoken requires slow tokenizer
    )
    
    print("Starting quantization...")
    print(f"Quantization config: {quant_config}")
    print("Note: Kimi-K2 has 384 experts per layer (vs 256 in DeepSeek), this may require more memory")
    
    # Quantize the model
    model.quantize(
        tokenizer, 
        quant_config=quant_config,
        calib_data="pileval",  # Use a standard calibration dataset
        max_calib_samples=128,  # Adjust based on available memory
        max_calib_seq_len=512,  # Kimi supports long context but we use shorter for calibration
    )
    
    print("Quantization completed successfully!")
    
    # Save quantized model
    print(f"Saving quantized model to {quant_path}...")
    os.makedirs(quant_path, exist_ok=True)
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    
    # Save quantization config for reference
    import json
    with open(os.path.join(quant_path, "quantization_config.json"), "w") as f:
        json.dump({
            "quant_method": "awq",
            "config": quant_config,
            "model_type": "kimi-k2",
            "base_architecture": "deepseek",
            "num_experts": 384,
            "vocab_size": 163840
        }, f, indent=2)
    
    print(f'Model is quantized and saved at "{quant_path}"')
    print("\nModel info:")
    print(f"- Architecture: Kimi-K2 (DeepSeek-based)")
    print(f"- Experts per layer: 384")
    print(f"- Vocabulary size: 163,840")
    print(f"- Quantization: {quant_config['w_bit']}-bit AWQ")
    
except Exception as e:
    print(f"\nError during quantization: {e}")
    print("\nTroubleshooting tips:")
    print("1. Ensure you have enough memory (RAM + VRAM) - Kimi-K2 is a large model")
    print("2. Try reducing max_calib_samples or max_calib_seq_len")
    print("3. Make sure the model files are complete (61 safetensor shards)")
    print("4. Check that custom Kimi model code (modeling_deepseek.py, tokenization_kimi.py) is accessible")
    print("5. Consider using a machine with more memory or quantizing layer by layer")
    
    # Additional debugging
    import traceback
    print("\nFull error traceback:")
    traceback.print_exc()