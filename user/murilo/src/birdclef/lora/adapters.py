"""Adapter-hub/adapters LoRA backend for EAT encoders.

Requires: pip install adapters

The EAT model interface is auto-detected by adapters.init() via the
"eat" entry in CUSTOM_INTERFACES (added in adapter-hub/adapters#832).
Prerequisites: #830 (LoRAMergedLinear fused QKV fix) and #831
(NotImplementedError handling for get_input_embeddings).
"""

import logging

import adapters
from adapters import LoRAConfig

logger = logging.getLogger(__name__)


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

    model = encoder.model

    # Auto-detects "eat" interface from CUSTOM_INTERFACES
    adapters.init(model)

    # Create and add LoRA adapter
    lora_config = LoRAConfig(
        r=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        dropout=lora_cfg.get("dropout", 0.0),
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
