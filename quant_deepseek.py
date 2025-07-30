"""
Quantization script for DeepSeek-R1 with AutoAWQ on ROCm
This version includes a workaround for the rotary_emb issue
"""

import os
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Monkey patch to handle DeepSeek rotary embeddings
def patch_deepseek_rotary():
    """Patch the quantizer to handle DeepSeek models without rotary_emb"""
    import awq.quantize.quantizer
    
    original_quantize = awq.quantize.quantizer.AwqQuantizer.quantize
    
    def patched_quantize(self):
        # Temporarily set transformers version to bypass rotary_emb check
        import transformers
        original_version = transformers.__version__
        
        # Check if this is a DeepSeek model
        model_type = getattr(self.awq_model, 'model_type', '')
        if 'deepseek' in model_type.lower():
            # Temporarily downgrade version to skip rotary_emb computation
            transformers.__version__ = "4.47.0"
        
        try:
            # Call original quantize method
            original_quantize(self)
        finally:
            # Restore original version
            transformers.__version__ = original_version
    
    awq.quantize.quantizer.AwqQuantizer.quantize = patched_quantize
    print("Applied DeepSeek rotary embedding patch")

# Apply the patch before loading the model
patch_deepseek_rotary()

# Model paths
model_path = '/home/hotaisle/workspace/models/DeepSeek-R1-0528-bf16'
quant_path = '/home/hotaisle/workspace/models/DeepSeek-R1-0528-awq'

# Quantization config
quant_config = { 
    "zero_point": True, 
    "q_group_size": 64, 
    "w_bit": 4, 
    "version": "GEMM"  # GEMM is recommended for ROCm
}

print(f"Loading model from {model_path}...")
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    safetensors=True,  # Use safetensors if available
    device_map="cpu",  # Load on CPU first to save memory
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Starting quantization...")
print(f"Quantization config: {quant_config}")

try:
    # Quantize the model
    model.quantize(
        tokenizer, 
        quant_config=quant_config,
        calib_data="pileval",  # Use a standard calibration dataset
        max_calib_samples=128,  # Reduce if you encounter OOM
        max_calib_seq_len=512,  # Reduce if you encounter OOM
    )
    
    print("Quantization completed successfully!")
    
    # Save quantized model
    print(f"Saving quantized model to {quant_path}...")
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    
    print(f'Model is quantized and saved at "{quant_path}"')
    
except Exception as e:
    print(f"Error during quantization: {e}")
    print("\nTroubleshooting tips:")
    print("1. Ensure you have enough memory (RAM + VRAM)")
    print("2. Try reducing max_calib_samples or max_calib_seq_len")
    print("3. Make sure ROCm and PyTorch are properly installed")
    print("4. Check that the model path is correct and accessible")
