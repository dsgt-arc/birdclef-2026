"""Adapter-hub/adapters LoRA backend for EAT encoders.

Requires: pip install adapters
"""

import logging

import adapters
from adapters import AdapterModelInterface, LoRAConfig
from adapters.methods.lora import LoRAMergedLinear

logger = logging.getLogger(__name__)

# Workaround for adapters bug: LoRAMergedLinear.get_n_heads() counts "o" in
# attn_matrices, but the fused QKV layer only has Q/K/V. This causes a shape
# mismatch in the grouped conv1d (lora_B sized for 4 groups, enable_lora has 3).
# Fix: only count Q/K/V matrices for the merged (fused QKV) layer.
_QKV_MATRICES = frozenset({"q", "k", "v"})
_orig_get_n_heads = LoRAMergedLinear.get_n_heads


def _patched_get_n_heads(self, lora):
    return len(set(lora.attn_matrices) & _QKV_MATRICES)


LoRAMergedLinear.get_n_heads = _patched_get_n_heads

# Maps EAT's module names to the standardized hook points the adapters library
# needs. EAT is structurally a standard ViT with fused QKV (same pattern as
# ModernBERT, which is already supported via attn_qkv_proj).
#
# base_model is omitted (defaults to None) so that EATModel itself — which IS
# a PreTrainedModel — serves as the base model (EATModel.base_model_prefix="").
# Model-level paths use "model." prefix to reach through to the EAT backbone.
# Layer-level paths are relative to each AltBlock.
#
# Resolved module hierarchy:
#   EATModel.model.blocks[i].attn.qkv  -> nn.Linear(768, 2304) [fused Q/K/V]
#   EATModel.model.blocks[i].attn.proj -> nn.Linear(768, 768)   [output projection]
#   EATModel.model.blocks[i].mlp.fc1   -> nn.Linear(768, 3072)  [FFN intermediate]
#   EATModel.model.blocks[i].mlp.fc2   -> nn.Linear(3072, 768)  [FFN output]

EAT_ADAPTER_INTERFACE = AdapterModelInterface(
    adapter_methods=["lora"],
    model_embeddings="model.local_encoder",
    model_layers="model.blocks",
    layer_self_attn="attn",
    layer_cross_attn=None,
    attn_qkv_proj="qkv",
    attn_o_proj="proj",
    layer_intermediate_proj="mlp.fc1",
    layer_output_proj="mlp.fc2",
    layer_pre_self_attn="norm1",
    layer_pre_ffn="norm2",
    layer_ln_1=None,  # EAT uses pre-norm architecture
    layer_ln_2=None,
)


def apply_adapters_lora(encoder, config: dict):
    """Apply LoRA using adapter-hub/adapters library.

    Maps the same YAML config format used by PEFT to adapters LoRAConfig:
      target_modules "qkv"  -> attn_matrices includes q,k,v (fused QKV)
      target_modules "proj" -> attn_matrices includes o
      target_modules "fc1"  -> intermediate_lora=True
      target_modules "fc2"  -> output_lora=True
    """
    lora_cfg = config["lora"]
    target_modules = lora_cfg["target_modules"]

    # Map target_modules to adapters LoRA config parameters
    attn_matrices = []
    if "qkv" in target_modules:
        attn_matrices.extend(["q", "k", "v"])
    if "proj" in target_modules:
        attn_matrices.append("o")

    intermediate_lora = "fc1" in target_modules
    output_lora = "fc2" in target_modules

    # EAT config uses embed_dim/num_heads; adapters expects hidden_size/num_attention_heads
    encoder.model.config.hidden_size = encoder.model.config.embed_dim
    encoder.model.config.num_attention_heads = encoder.model.config.num_heads

    # EATModel is an audio model without a token vocabulary, so its
    # get_input_embeddings() raises NotImplementedError. The adapters library's
    # EmbeddingAdaptersMixin calls it during init to cache the default embedding.
    # Point it at the patch embedding (local_encoder) which is the closest analog.
    model = encoder.model
    model.get_input_embeddings = lambda: model.model.local_encoder
    model.set_input_embeddings = lambda v: setattr(model.model, "local_encoder", v)

    # Initialize adapters on the EAT model.
    # EATModel.base_model_prefix="" means EATModel IS its own base model.
    # Since EATModel is a PreTrainedModel, it already has the interface the
    # adapters mixins expect — avoiding the crash that occurred when the plain
    # nn.Module EAT backbone was used as base model.
    adapters.init(model, interface=EAT_ADAPTER_INTERFACE)

    # Create and add LoRA adapter
    lora_config = LoRAConfig(
        r=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        attn_matrices=attn_matrices,
        intermediate_lora=intermediate_lora,
        output_lora=output_lora,
    )

    encoder.model.add_adapter("lora", config=lora_config)
    encoder.model.set_active_adapters("lora")
    encoder.model.train_adapter("lora")

    # Print trainable parameter summary (matches PEFT output format)
    trainable = sum(p.numel() for p in encoder.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.model.parameters())
    pct = trainable / total * 100
    logger.info(
        f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}"
    )

    return encoder
