import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM


class KimiK2AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "DeepseekV3DecoderLayer"  # Kimi-K2 uses DeepSeek architecture
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["gate"]  # Don't quantize the MoE gate

    @staticmethod
    def get_model_layers(model):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(
        module, input_feat, module_kwargs
    ):
        layers = []

        # Handle attention layers - Kimi-K2 uses the same structure as DeepSeek V3
        if hasattr(module.self_attn, "q_proj"):
            # attention input
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.self_attn.q_proj,
                        module.self_attn.kv_a_proj_with_mqa,
                    ],
                    inp=input_feat["self_attn.q_proj"],
                    module2inspect=module.self_attn,
                    kwargs=module_kwargs,
                )
            )
        else:
            # attention input with separate q_a_proj
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.self_attn.q_a_proj,
                        module.self_attn.kv_a_proj_with_mqa,
                    ],
                    inp=input_feat["self_attn.q_a_proj"],
                    module2inspect=module.self_attn,
                    kwargs=module_kwargs,
                )
            )
            layers.append(
                dict(
                    prev_op=module.self_attn.q_a_layernorm,
                    layers=[
                        module.self_attn.q_b_proj,
                    ],
                    inp=input_feat["self_attn.q_b_proj"],
                )
            )

        # kv layernorm
        layers.append(
            dict(
                prev_op=module.self_attn.kv_a_layernorm,
                layers=[
                    module.self_attn.kv_b_proj,
                ],
                inp=input_feat["self_attn.kv_b_proj"],
            )
        )

        # Handle MLP layers
        if hasattr(module.mlp, "gate"):
            # MoE layer (layers 1-60 in Kimi-K2)
            # Check which key is available for MoE input
            if "mlp" in input_feat:
                mlp_inp_key = "mlp"
            else:
                # Look for expert-specific keys
                expert_keys = [k for k in input_feat.keys() if "mlp.experts" in k or "mlp.shared_experts" in k]
                if expert_keys:
                    mlp_inp_key = expert_keys[0]
                    print(f"DEBUG: MoE layer using key: {mlp_inp_key}")
                else:
                    print(f"DEBUG: MoE layer - available keys: {list(input_feat.keys())}")
                    raise KeyError(f"No MLP input found for MoE. Available keys: {list(input_feat.keys())}")
            
            # linear in
            # For MoE layers, we need to process each weight separately to avoid issues
            # Handle gate_proj for all experts
            for i, expert in enumerate(module.mlp.experts):
                expert_gate_key = f"mlp.experts.{i}.gate_proj" if f"mlp.experts.{i}.gate_proj" in input_feat else mlp_inp_key
                layers.append(
                    dict(
                        prev_op=module.post_attention_layernorm,
                        layers=[expert.gate_proj],
                        inp=input_feat[expert_gate_key],
                    )
                )
            
            # Handle up_proj for all experts
            for i, expert in enumerate(module.mlp.experts):
                expert_up_key = f"mlp.experts.{i}.up_proj" if f"mlp.experts.{i}.up_proj" in input_feat else mlp_inp_key
                layers.append(
                    dict(
                        prev_op=module.post_attention_layernorm,
                        layers=[expert.up_proj],
                        inp=input_feat[expert_up_key],
                    )
                )
            
            # Handle shared experts
            shared_gate_key = "mlp.shared_experts.gate_proj" if "mlp.shared_experts.gate_proj" in input_feat else mlp_inp_key
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.shared_experts.gate_proj],
                    inp=input_feat[shared_gate_key],
                )
            )
            
            shared_up_key = "mlp.shared_experts.up_proj" if "mlp.shared_experts.up_proj" in input_feat else mlp_inp_key
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.shared_experts.up_proj],
                    inp=input_feat[shared_up_key],
                )
            )

            # linear out for each expert
            for i, expert in enumerate(module.mlp.experts):
                layers.append(
                    dict(
                        prev_op=expert.up_proj,
                        layers=[expert.down_proj],
                        inp=input_feat[f"mlp.experts.{i}.down_proj"],
                    )
                )
            # linear out for shared experts
            layers.append(
                dict(
                    prev_op=module.mlp.shared_experts.up_proj,
                    layers=[module.mlp.shared_experts.down_proj],
                    inp=input_feat[f"mlp.shared_experts.down_proj"],
                )
            )
        else:
            # Dense layer (layer 0 in Kimi-K2)
            # Based on the debug output, we need to use the specific keys available
            # linear 1
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.up_proj"],  # Use up_proj as input since gate_proj key doesn't exist
                    module2inspect=module.mlp,
                )
            )

            # linear 2
            layers.append(
                dict(
                    prev_op=module.mlp.up_proj,
                    layers=[module.mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )

        return layers