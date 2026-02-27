# Upstream PR: Add ConvNeXt to adapter-hub/adapters

**Fork**: https://github.com/mgustineli/adapters (cloned at `~/clef/adapters/`)
**Upstream**: https://github.com/adapter-hub/adapters
**Blocked on**: [PR #831](https://github.com/adapter-hub/adapters/pull/831) — fixes `get_input_embeddings()` crash for vision/audio models in `init_adapters()` (submitted in exp 006, CI passing, awaiting review)

## Background

Exp 007 showed ConvNeXt-Base with PEFT LoRA on MLP layers (fc1, fc2) reaches cmAP=0.527
on BirdCLEF 2025 — only 1.7% below EAT's best (0.536). Adding ConvNeXt to the adapters
library's `CUSTOM_INTERFACES` would enable `adapters.init(model)` to work out-of-the-box
with HuggingFace's `ConvNextModel`, expanding the library beyond transformer-only models.

## Goal

Add `"convnext"` to `CUSTOM_INTERFACES` so `adapters.init(model)` works with HuggingFace's
`ConvNextModel`. The upstream PR targets HuggingFace transformers (not timm, which our
experiment uses).

## ConvNeXt Architecture (HuggingFace transformers)

```
ConvNextModel
  .embeddings                              # stem (patch embed + norm)
  .encoder.stages                          # nn.ModuleList of 4 ConvNextStage
    [i].downsampling_layer                 # downsample between stages
    [i].layers                             # nn.Sequential of ConvNextLayer (blocks)
      [j].dwconv                           # depthwise Conv2d (spatial mixing)
      [j].layernorm                        # pre-FFN norm
      [j].pwconv1                          # nn.Linear — MLP expand (LoRA target)
      [j].pwconv2                          # nn.Linear — MLP reduce (LoRA target)
```

ConvNeXt-Base: 4 stages with [3, 3, 27, 3] blocks = 36 total blocks.
No attention layers — LoRA targets MLP only (`pwconv1`, `pwconv2`).

**timm vs HuggingFace naming** (our exp uses timm):

| Aspect | timm (our exp 007) | HuggingFace (upstream target) |
|--------|-------------------|-------------------------------|
| Stages | `stages[i].blocks[j]` | `encoder.stages[i].layers[j]` |
| MLP expand | `mlp.fc1` | `pwconv1` |
| MLP reduce | `mlp.fc2` | `pwconv2` |
| Embeddings | `stem` | `embeddings` |

## Main Challenge: Nested Stages

The adapters library's `model_layers` expects a flat `ModuleList` (e.g., BERT's
`encoder.layer` with 12 layers). ConvNeXt has a **two-level hierarchy**:
`encoder.stages[i].layers[j]` — 4 stages containing variable numbers of blocks.

**Proposed solution**: Extend `AdapterModelInterface` with a `layer_blocks` field:
```python
model_layers="convnext.encoder.stages",
layer_blocks="layers",  # new field: iterate stages[i].layers[j]
```
This would also benefit Swin Transformer, EfficientNet, and other stage-based architectures.

**Recommendation**: File an issue first to discuss the design with maintainers before coding.

## No Attention Layers

ConvNeXt is a pure CNN — all `attn_*` fields are `None` in the interface. Need to verify
the library handles no-attention models gracefully (e.g., `init_lora()` may assume attention
modules exist). This would be the first CNN in `CUSTOM_INTERFACES`.

## Proposed Interface Entry

```python
"convnext": AdapterModelInterface(
    adapter_methods=["lora"],
    model_embeddings="convnext.embeddings",
    model_layers="convnext.encoder.stages",   # needs layer_blocks support
    layer_self_attn=None,                     # no attention in ConvNeXt
    layer_cross_attn=None,
    attn_qkv_proj=None,
    attn_o_proj=None,
    layer_intermediate_proj="pwconv1",
    layer_output_proj="pwconv2",
    layer_pre_self_attn=None,
    layer_pre_ffn="layernorm",
    layer_ln_1=None,
    layer_ln_2=None,
),
```

## Test Config

`ConvNextConfig` is in `transformers` (standard import, no `trust_remote_code`),
and `VisionAdapterTestBase` handles `pixel_values` input:

```python
from transformers import ConvNextConfig

"ConvNext": {
    "config_class": ConvNextConfig,
    "config_params": {
        "num_channels": 3,
        "hidden_sizes": [32, 64, 128, 256],  # tiny for fast tests
        "depths": [1, 1, 1, 1],
        "patch_size": 4,
    },
    "test_base": VisionAdapterTestBase,
},
```

## Adapters Repo Key File Paths

```
src/adapters/
  wrappers/interfaces.py          # CUSTOM_INTERFACES dict — where to add "convnext"
  wrappers/model.py               # adapters.init() implementation
  interface.py                    # AdapterModelInterface dataclass — add layer_blocks here
  model_mixin.py                  # ModelAdaptersMixin, EmbeddingAdaptersMixin
  methods/lora.py                 # LoRALinear, init_lora() — verify no-attention works

tests/
  test_methods/test_all_custom_interfaces.py  # MODEL_CONFIGS dict — add "ConvNext" here
  test_methods/base.py                        # VisionAdapterTestBase (use for ConvNeXt)
```

## PR Contribution Pattern

Recent model additions (PRs #810 Qwen3, #807 Gemma3, #803 Phi, #802 Qwen2) follow:
1. Add entry to `CUSTOM_INTERFACES` in `interfaces.py`
2. Add entry to `MODEL_CONFIGS` in `test_all_custom_interfaces.py`
3. Update `docs/model_overview.md`

For these transformer models it's a 2-3 file change. ConvNeXt is more complex because of
the nested stages (needs `interface.py` changes) and being the first no-attention model.

## Files to Change

1. `src/adapters/interface.py` — add `layer_blocks` field to `AdapterModelInterface`
2. `src/adapters/wrappers/interfaces.py` — add `"convnext"` to `CUSTOM_INTERFACES`
3. `tests/test_methods/test_all_custom_interfaces.py` — add `"ConvNext"` to `MODEL_CONFIGS`
4. `docs/model_overview.md` — add ConvNeXt to supported models
5. Layer iteration logic (location TBD) — handle `layer_blocks` nested iteration

## Implementation Plan

1. File issue proposing nested stage support (`layer_blocks`) — get maintainer feedback
2. Wait for [PR #831](https://github.com/adapter-hub/adapters/pull/831) to merge
3. Implement ConvNeXt interface with agreed-upon nested layer approach
4. Add tests + docs, submit PR
