"""
Patch for AutoAWQ quantizer to support DeepSeek-V3/R1 models on ROCm
This addresses the rotary_emb attribute error
"""

import torch
import transformers
from awq.quantize.quantizer import AwqQuantizer
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

class PatchedAwqQuantizer(AwqQuantizer):
    def quantize(self):
        """Override quantize method to handle DeepSeek-V3 rotary embeddings"""
        from tqdm import tqdm
        from awq.utils.utils import clear_memory, get_best_device
        from awq.utils.module import (
            append_str_prefix,
            get_op_name,
            get_named_linears,
            set_op_by_name,
            exclude_layers_to_not_quantize,
        )
        from awq.quantize.scale import apply_scale, apply_clip
        
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            # Move module and inputs to correct device
            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                if torch.cuda.is_available():
                    best_device = "cuda:" + str(i % torch.cuda.device_count())
                else:
                    best_device = get_best_device()

                self.modules[i] = self.modules[i].to(best_device)
                common_device = next(self.modules[i].parameters()).device

            if self.module_kwargs.get("position_ids") is not None:
                self.module_kwargs["position_ids"] = self.module_kwargs[
                    "position_ids"
                ].to(common_device)

            if self.module_kwargs.get("attention_mask") is not None:
                self.module_kwargs["attention_mask"] = self.module_kwargs[
                    "attention_mask"
                ].to(common_device)

            self.inps = self.inps.to(common_device)

            # Move embeddings
            self.awq_model.move_embed(self.model, common_device)

            # Handle position embeddings for transformers >= 4.48.0
            if (
                transformers.__version__ >= "4.48.0"
                and self.module_kwargs.get("position_embeddings") is None
            ):
                # Check if this is a DeepSeek model
                model_type = getattr(self.awq_model, 'model_type', '')
                if 'deepseek' in model_type.lower():
                    # For DeepSeek models, skip position embeddings computation
                    # as they handle it internally
                    pass
                else:
                    # For other models, try to compute position embeddings
                    if hasattr(self.model.model, 'rotary_emb'):
                        self.module_kwargs["position_embeddings"] = self.model.model.rotary_emb(
                            self.inps, self.module_kwargs["position_ids"]
                        )

            if (transformers.__version__ >= "4.48.0"
                and self.module_kwargs.get('attention_mask') is None):
                self.module_kwargs['attention_mask'] = None

            for k, v in self.module_kwargs.items():
                # position embeddings found in tuple
                if isinstance(v, tuple):
                    self.module_kwargs[k] = tuple(
                        item.to(common_device) if isinstance(item, (torch.Tensor, torch.nn.Module)) 
                        else item for item in v
                    )

            # Get layer, extract linear modules, extract input features
            named_linears = get_named_linears(self.modules[i])

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )

            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()

            # Compute and apply scale list
            module_config = self.awq_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )
            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in module_config
            ]
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.modules[i]) + "."
            )

            # Compute and apply clipping list
            if self.apply_clip:
                clip_list = self._search_best_clip(
                    self.modules[i], named_linears, input_feat
                )
                apply_clip(self.modules[i], clip_list)
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.modules[i]) + "."
                )

            # Quantize weights
            if not self.export_compatible:
                self._apply_quant(self.modules[i], named_linears)

            clear_memory()


# Monkey patch the original quantizer
def patch_autoawq_for_deepseek():
    """Apply the patch to AutoAWQ for DeepSeek models"""
    import awq.quantize.quantizer
    awq.quantize.quantizer.AwqQuantizer = PatchedAwqQuantizer
    print("Applied DeepSeek compatibility patch to AutoAWQ")


if __name__ == "__main__":
    # Apply the patch
    patch_autoawq_for_deepseek()
    
    # Now run your quantization
    model_path = '/home/hotaisle/workspace/models/DeepSeek-R1-0528-bf16'
    quant_path = '/home/hotaisle/workspace/models/DeepSeek-R1-0528-awq'
    quant_config = { "zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM" }
    
    print(f"Loading model from {model_path}...")
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("Starting quantization...")
    model.quantize(tokenizer, quant_config=quant_config)
    
    print("Saving quantized model...")
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    
    print(f'Model is quantized and saved at "{quant_path}"')
