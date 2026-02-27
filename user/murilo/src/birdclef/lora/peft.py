"""PEFT LoRA adapter application for pretrained audio encoders.

Requires: pip install peft
"""

import logging

from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


def apply_lora(encoder, config: dict):
    """Apply PEFT LoRA to the encoder's internal model.

    PEFT wraps encoder.model (the HuggingFace EAT model). The wrapper delegates
    attribute access (including extract_features) through to the base model,
    so EATEncoder.forward() continues to work unchanged.
    """
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )

    encoder.model = get_peft_model(encoder.model, lora_config)
    encoder.model.print_trainable_parameters()

    return encoder
