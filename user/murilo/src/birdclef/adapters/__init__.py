"""Adapter backends for applying LoRA to pretrained audio encoders.

Both backends are optional dependencies — import the specific module you need:

    from birdclef.adapters.peft import apply_lora          # requires: peft
    from birdclef.adapters.eat import apply_adapters_lora   # requires: adapters
"""
