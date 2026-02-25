# Integrating the `adapters` Library with EAT (Non-Standard Audio Model)

**Date:** 2026-02-25
**Library:** [adapter-hub/adapters](https://github.com/adapter-hub/adapters) v1.2.0
**Model:** [EAT-base](https://huggingface.co/worstchan/EAT-base_epoch30_pretrain) (Efficient Audio Transformer)

## Context

We integrated the `adapters` library's LoRA support with the EAT audio encoder
using the [plugin interface](https://docs.adapterhub.ml/plugin_interface.html)
(`AdapterModelInterface`). EAT is structurally a ViT with fused QKV projections,
similar to ModernBERT (supported since PR [#798](https://github.com/adapter-hub/adapters/pull/798)).

This document captures the issues encountered and workarounds applied.

## EAT Model Hierarchy

```
EATEncoder (nn.Module)            # our wrapper
  .model = EATModel               # PreTrainedModel (base_model_prefix="")
    .model = EAT                  # plain nn.Module backbone
      .local_encoder = PatchEmbed # Conv2d patch embedding (no token vocab)
      .blocks = ModuleList[12]    # transformer layers
        [i].attn.qkv              # fused Q/K/V  → nn.Linear(768, 2304)
        [i].attn.proj             # output proj   → nn.Linear(768, 768)
        [i].mlp.fc1               # FFN expand    → nn.Linear(768, 3072)
        [i].mlp.fc2               # FFN project   → nn.Linear(3072, 768)
        [i].norm1                 # pre-attention LayerNorm
        [i].norm2                 # pre-FFN LayerNorm
      .pre_norm = LayerNorm
```

## Issue 1: Plain `nn.Module` Backbone as Base Model

### Problem

Setting `base_model="model"` in the interface makes the EAT backbone
(`EATModel.model`, a plain `nn.Module`) the base model. `adapters.init()`
dynamically wraps the base model's class with `EmbeddingAdaptersMixin` and
`ModelBaseAdaptersMixin`. These mixins expect `PreTrainedModel` methods
(`get_input_embeddings`, `base_model`, etc.) that `nn.Module` doesn't have.

### Fix

Omit `base_model` from the interface (defaults to `None`). Since
`EATModel.base_model_prefix=""`, the `EATModel` itself (a `PreTrainedModel`)
becomes the base model. Model-level interface fields use dotted paths to reach
through to the backbone:

```python
EAT_ADAPTER_INTERFACE = AdapterModelInterface(
    model_layers="model.blocks",          # EATModel.model.blocks
    model_embeddings="model.local_encoder",  # EATModel.model.local_encoder
    # base_model omitted → EATModel is base model
    ...
)
```

The adapters library resolves dotted paths via internal `multigetattr()` /
`multihasattr()` helpers, so this works.

### Takeaway

For models where the backbone is a plain `nn.Module` nested inside a
`PreTrainedModel` wrapper, prefer making the `PreTrainedModel` the base model
and using dotted paths for model-level fields.

## Issue 2: `get_input_embeddings()` for Audio Models

### Problem

`EATModel` inherits `get_input_embeddings()` from `PreTrainedModel`, but the
default implementation raises `NotImplementedError` (audio models have no token
vocabulary). The `EmbeddingAdaptersMixin`, mixed into the base model during
`adapters.init()`, calls this method at the end of `init_adapters()`:

```python
# model_mixin.py:499
if isinstance(self, EmbeddingAdaptersMixin):
    self.loaded_embeddings["default"] = self.get_input_embeddings()
```

### Fix

Monkey-patch before calling `adapters.init()`:

```python
model.get_input_embeddings = lambda: model.model.local_encoder
model.set_input_embeddings = lambda v: setattr(model.model, "local_encoder", v)
```

### Takeaway

Any model that doesn't implement `get_input_embeddings()` will hit this. The
adapters library unconditionally calls it whenever `EmbeddingAdaptersMixin` is
applied — even if no embedding adapters are being used. A possible library fix
would be to make this call conditional on whether embedding adapters are
actually requested.

## Issue 3: Fused QKV LoRA Shape Mismatch (Library Bug)

### Problem

When using `attn_qkv_proj` (fused QKV) + `attn_o_proj` (separate output proj)
with `attn_matrices=["q", "k", "v", "o"]`:

`LoRAMergedLinear.get_n_heads()` returns `len(set(attn_matrices)) = 4`,
counting "o". But the fused QKV layer only has 3 components (Q, K, V). The
`add_adapter()` method correctly sets `enable_lora = [True, True, True]`
(3 entries for Q, K, V only).

This causes `_get_lora_shapes()` to create LoRA weights sized for 4 groups:
- `lora_A`: shape `(r * 4, in_features)` = `(32, 768)`
- `lora_B`: shape `(out_features // 3 * 4, r)` = `(3072, 8)`

But the forward pass uses `conv1d(groups=sum(enable_lora))` = `conv1d(groups=3)`,
which expects `3 * r = 24` input channels, not `4 * r = 32`:

```
RuntimeError: Given groups=3, weight of size [3072, 8, 1],
expected input[2, 32, 257] to have 24 channels, but got 32 channels instead
```

### Root Cause

```python
# adapters/methods/lora.py — LoRAMergedLinear
def get_n_heads(self, lora):
    return len(set(lora.attn_matrices))  # Bug: counts "o" → returns 4

def add_adapter(self, adapter_name, layer_idx):
    ...
    lora.enable_lora = [
        "q" in lora_config.attn_matrices,
        "k" in lora_config.attn_matrices,
        "v" in lora_config.attn_matrices,
    ]  # Correct: only Q, K, V → 3 entries
```

`get_n_heads()` includes "o" but `enable_lora` doesn't. The shape computation
and the forward pass disagree on the number of groups.

### Fix (Monkey-Patch)

```python
from adapters.methods.lora import LoRAMergedLinear

_QKV_MATRICES = frozenset({"q", "k", "v"})
LoRAMergedLinear.get_n_heads = lambda self, lora: len(
    set(lora.attn_matrices) & _QKV_MATRICES
)
```

### Proposed Library Fix

In `LoRAMergedLinear.get_n_heads()`, only count Q/K/V matrices since the merged
layer only contains these three projections. The output projection ("o") is
handled by a separate `LoRALinear` layer via `attn_o_proj`:

```python
def get_n_heads(self, lora):
    return len(set(lora.attn_matrices) & {"q", "k", "v"})
```

### Reproduction

```python
import adapters
from adapters import AdapterModelInterface, LoRAConfig
from transformers import AutoModelForMaskedLM

# Any model with fused QKV + separate output projection
model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")
adapters.init(model)

# This crashes with the shape mismatch:
model.add_adapter("lora", config=LoRAConfig(
    r=8, alpha=16,
    attn_matrices=["q", "k", "v", "o"],  # "o" triggers the bug
))
```

**Note:** This bug affects any model using `attn_qkv_proj` (fused QKV) when
`attn_matrices` includes `"o"`. It was introduced in PR #798 which added
support for merged QKV layers. It does not affect models with separate Q/K/V
projections.

## Working Configuration

```python
from adapters import AdapterModelInterface

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
    layer_ln_1=None,
    layer_ln_2=None,
)
```

## References

- [Plugin Interface docs](https://docs.adapterhub.ml/plugin_interface.html)
- [PR #798: ModernBERT & Gemma2 unified QKV support](https://github.com/adapter-hub/adapters/pull/798)
- [Issue #813: Plugin interface limitations (Parallel composition)](https://github.com/adapter-hub/adapters/issues/813)
- Our implementation: `user/murilo/src/birdclef/lora/adapters.py`
- Tests: `user/murilo/tests/006-lora-finetune/test_adapters_lora.py`
