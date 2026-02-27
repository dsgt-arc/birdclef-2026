# Upstream PRs: adapter-hub/adapters

**Fork**: https://github.com/mgustineli/adapters (cloned at `~/clef/adapters/`)
**Upstream**: https://github.com/adapter-hub/adapters
**Related docs**: `docs/references/adapters-lora-eat.md` (integration workarounds)

## PR #830 — Fix LoRAMergedLinear.get_n_heads() (bug fix)

**URL**: https://github.com/adapter-hub/adapters/pull/830
**Branch**: `fix/lora-merged-linear-get-n-heads`
**Status**: CI passing, awaiting reviewer approval

**What it fixes**: `LoRAMergedLinear.get_n_heads()` counts `"o"` in `attn_matrices`,
but the fused QKV layer only has Q/K/V. This causes a shape mismatch in the grouped
`conv1d` when `attn_matrices=["q","k","v","o"]`. Affects any model using `attn_qkv_proj`.

**Changes**:
- `src/adapters/methods/lora.py`: filter to `{"q","k","v"}` in `get_n_heads()`
- `tests/test_methods/test_all_custom_interfaces.py`: regression test with all 4 matrices

**Our workaround** (can be removed after merge):
`user/murilo/src/birdclef/lora/adapters.py` lines 14-26 — monkey-patches `LoRAMergedLinear.get_n_heads`.

## PR #831 — Handle NotImplementedError from get_input_embeddings() (enhancement)

**URL**: https://github.com/adapter-hub/adapters/pull/831
**Branch**: `fix/embedding-mixin-notimplemented`
**Status**: CI passing, awaiting reviewer approval

**What it fixes**: `init_adapters()` unconditionally calls `self.get_input_embeddings()`
to cache the default token embedding. Audio/vision models raise `NotImplementedError`
since they have no token vocabulary, crashing `adapters.init()` even for LoRA-only usage.

**Changes**:
- `src/adapters/model_mixin.py`: wrap the call in try/except `NotImplementedError`

**Our workaround** (can be removed after merge):
`user/murilo/src/birdclef/lora/adapters.py` lines 88-92 — monkey-patches
`get_input_embeddings()` to return `model.model.local_encoder`.

## Next PR — Add EAT to CUSTOM_INTERFACES (blocked on #830 + #831)

**Goal**: Add `"eat"` to the `CUSTOM_INTERFACES` dict so `adapters.init(model)` auto-detects
EAT models without requiring a manual `interface=` argument.

### Files to change in the adapters repo

1. **`src/adapters/wrappers/interfaces.py`** — add `"eat"` entry to `CUSTOM_INTERFACES`:
   ```python
   "eat": AdapterModelInterface(
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
       layer_ln_1=None,   # EAT uses pre-norm architecture
       layer_ln_2=None,
   ),
   ```

2. **`tests/test_methods/test_all_custom_interfaces.py`** — add `"EAT"` to `MODEL_CONFIGS`

3. **`docs/model_overview.md`** — add EAT to supported models list

### Blockers and open questions

#### 1. Config attribute naming mismatch

EAT's config uses non-standard names. The adapters library reads these from `model.config`:

| adapters expects | EAT uses | Where it's read |
|-----------------|----------|-----------------|
| `hidden_size` | `embed_dim` | `model_mixin.py` (bottleneck, invertible), `heads/base.py`, `methods/reft.py` |
| `num_attention_heads` | `num_heads` | `methods/prefix_tuning.py` |

**For LoRA-only this isn't a problem** — LoRA code doesn't read these config attributes.
But if someone tries bottleneck/ReFT/prefix tuning/heads on EAT, it would crash.

**Options**:
- (a) Add `adapter_methods=["lora"]` only (limits scope, but honest about what works)
- (b) Patch config in the interface setup (e.g., add `config_hidden_size` field to
  `AdapterModelInterface` — this would be a design change the maintainers need to weigh in on)
- (c) Fix in EAT's own config class (add `hidden_size = embed_dim` alias) — separate PR
  to the EAT model repo

**Recommendation**: start with (a) since LoRA is the primary use case. File a follow-up
issue for config aliasing support.

#### 2. Test framework: trust_remote_code models

The existing test pattern:
```python
MODEL_CONFIGS = {
    "ModernBERT": {
        "config_class": ModernBertConfig,     # imported from transformers
        "config_params": { ... },             # tiny model for fast tests
        "test_base": TextAdapterTestBase,     # provides dummy input generation
    },
}
```

EAT's config class (`EATConfig`) is NOT in `transformers` — it's in the model's HuggingFace
repo, loaded via `trust_remote_code=True`. This means:
- Can't `from transformers import EATConfig`
- Need to either: (a) load dynamically via `AutoConfig.from_pretrained()`, (b) vendor the
  config class into the test, or (c) skip if model not available

Also need an appropriate test base. EAT takes mel-spectrogram inputs, not text tokens.
The `VisionAdapterTestBase` (`pixel_values` input) is closest but not exact. May need a
custom test base or adapt `AudioAdapterTestBase` (currently expects `input_features`).

**EAT input format**: `(batch, 1, T, 128)` mel-spectrogram tensors. The model's forward
pass in `modeling_eat.py` calls `self.model(source=fbank, ...)` where `fbank` is the
mel input.

**EAT model card**: https://huggingface.co/worstchan/EAT-base_epoch30_pretrain

**EAT config** (tiny version for tests):
```python
# Minimal config for fast tests (actual model has embed_dim=768, depth=12)
EATConfig(
    embed_dim=32,
    depth=4,
    num_heads=4,
    mlp_ratio=4.0,
    in_chans=1,
    img_size=[64, 128],  # [T, mel_bins] — small for tests
    patch_size=16,
    stride=16,
    mel_bins=128,
)
```

#### 3. EAT model hierarchy (base_model resolution)

EAT has `base_model_prefix=""` which means `EATModel` itself is the base model.
The backbone (`EAT`) is at `model.model` — a plain `nn.Module`. The interface uses
dotted paths (`"model.blocks"`, `"model.local_encoder"`) to reach through.

This works correctly with the plugin interface as-is. No special handling needed.

### Adapters repo key file paths

```
src/adapters/
  wrappers/interfaces.py          # CUSTOM_INTERFACES dict + get_adapter_interface()
  wrappers/model.py               # adapters.init() implementation
  interface.py                    # AdapterModelInterface dataclass
  model_mixin.py                  # ModelAdaptersMixin, EmbeddingAdaptersMixin
  methods/lora.py                 # LoRAMergedLinear, LoRALinear, init_lora()
  configuration/adapter_config.py # LoRAConfig (attn_matrices default: ["q","v"])

tests/
  test_methods/test_all_custom_interfaces.py  # MODEL_CONFIGS dict, test classes
  test_methods/base.py                        # TextAdapterTestBase, VisionAdapterTestBase, AudioAdapterTestBase
  test_methods/method_test_impl/base.py       # AdapterMethodBaseTestMixin, run_forward_test()
```

### PR contribution pattern (from existing PRs)

Recent model additions (#810 Qwen3, #807 Gemma3, #803 Phi, #802 Qwen2) follow this pattern:

1. Add entry to `CUSTOM_INTERFACES` in `interfaces.py`
2. Add entry to `MODEL_CONFIGS` in `test_all_custom_interfaces.py`
3. Update `docs/model_overview.md`
4. That's it — 2-3 files changed

For EAT, the non-standard config and trust_remote_code requirement make it more complex
than the typical PR. Discuss with maintainers before implementing.

### Our current workarounds (all in `user/murilo/src/birdclef/lora/adapters.py`)

| Lines | Workaround | Upstream fix |
|-------|-----------|-------------|
| 14-26 | Monkey-patch `get_n_heads()` to exclude "o" | PR #830 |
| 82-84 | Patch config: `hidden_size = embed_dim`, `num_attention_heads = num_heads` | Needs config aliasing or EAT config fix |
| 88-92 | Monkey-patch `get_input_embeddings()` to return patch embedding | PR #831 |
| 43-57 | Interface definition with dotted paths (not a workaround, correct usage) | Add to `CUSTOM_INTERFACES` |

After PRs #830 and #831 merge, lines 14-26 and 88-92 can be removed. Lines 82-84
(config patching) will still be needed until config aliasing is addressed.
