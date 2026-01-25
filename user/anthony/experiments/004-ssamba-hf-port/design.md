# Design: SSAMBA HuggingFace Port

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      SSAMBAModel                            │
├─────────────────────────────────────────────────────────────┤
│  Input: (batch, time_frames, freq_bins) e.g. (B, 1024, 128) │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              AudioPatchEmbed                          │  │
│  │  Conv2d(1, embed_dim, kernel=(16,16), stride=(16,16)) │  │
│  │  Output: (B, num_patches, embed_dim)                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  + cls_token + pos_embed                              │  │
│  │  Output: (B, num_patches + 1, embed_dim)              │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         BidirectionalMambaBlock × depth               │  │
│  │  ┌─────────────────┐  ┌─────────────────┐             │  │
│  │  │  MambaMixer     │  │  MambaMixer     │             │  │
│  │  │  (forward)      │  │  (backward)     │             │  │
│  │  └────────┬────────┘  └────────┬────────┘             │  │
│  │           │      output = fwd + bwd.flip()            │  │
│  │           └────────────────────┘                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    RMSNorm                            │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  Output: mean(tokens[1:]) → (B, embed_dim)                  │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. SSAMBAConfig

```python
@dataclass
class SSAMBAConfig:
    # Model size
    embed_dim: int = 384          # 192/384/768 for tiny/small/base
    depth: int = 24               # Number of Mamba blocks

    # Patch embedding
    img_size: tuple = (128, 1024) # (freq_bins, time_frames)
    patch_size: int = 16
    in_chans: int = 1

    # Mamba SSM
    d_state: int = 16             # SSM state dimension
    d_conv: int = 4               # Convolution kernel size
    expand: int = 2               # Expansion factor for inner dim

    # Other
    num_classes: int = 0          # 0 = feature extraction only
    use_cls_token: bool = True
```

### 2. Weight Mapping

SSAMBA checkpoint keys → HuggingFace model keys:

```
module.v.patch_embed.proj.weight    → patch_embed.proj.weight
module.v.patch_embed.proj.bias      → patch_embed.proj.bias
module.v.pos_embed                  → pos_embed
module.v.cls_token                  → cls_token

module.v.layers.{i}.mixer.in_proj.weight   → layers.{i}.forward_mamba.in_proj.weight
module.v.layers.{i}.mixer.in_proj.bias     → layers.{i}.forward_mamba.in_proj.bias
module.v.layers.{i}.mixer.conv1d.weight    → layers.{i}.forward_mamba.conv1d.weight
module.v.layers.{i}.mixer.conv1d.bias      → layers.{i}.forward_mamba.conv1d.bias
module.v.layers.{i}.mixer.x_proj.weight    → layers.{i}.forward_mamba.x_proj.weight
module.v.layers.{i}.mixer.dt_proj.weight   → layers.{i}.forward_mamba.dt_proj.weight
module.v.layers.{i}.mixer.dt_proj.bias     → layers.{i}.forward_mamba.dt_proj.bias
module.v.layers.{i}.mixer.A_log            → layers.{i}.forward_mamba.A_log
module.v.layers.{i}.mixer.D                → layers.{i}.forward_mamba.D
module.v.layers.{i}.mixer.out_proj.weight  → layers.{i}.forward_mamba.out_proj.weight
module.v.layers.{i}.mixer.out_proj.bias    → layers.{i}.forward_mamba.out_proj.bias
module.v.layers.{i}.norm.weight            → layers.{i}.norm.weight

module.v.norm_f.weight              → norm_f.weight
```

### 3. Bidirectional Processing

VisionMamba's `bimamba_type="v2"` does:

```python
# Original VisionMamba bidirectional (from both_models.py:359-370)
for i in range(len(self.v.layers) // 2):
    hidden_states_f, residual_f = self.v.layers[i * 2](hidden_states, residual)
    hidden_states_b, residual_b = self.v.layers[i * 2 + 1](
        hidden_states.flip([1]),
        None if residual is None else residual.flip([1])
    )
    hidden_states = hidden_states_f + hidden_states_b.flip([1])
    residual = residual_f + residual_b.flip([1])
```

Key insight: **layers are paired** - even indices are forward, odd indices are backward.
The model has 24 layers total = 12 bidirectional pairs.

Our implementation:
```python
class BidirectionalMambaStack(nn.Module):
    def __init__(self, config):
        self.pairs = nn.ModuleList([
            nn.ModuleDict({
                'forward': MambaMixer(mamba_config, layer_idx=i*2),
                'backward': MambaMixer(mamba_config, layer_idx=i*2+1),
                'norm': RMSNorm(config.embed_dim)
            })
            for i in range(config.depth // 2)
        ])

    def forward(self, x):
        residual = None
        for pair in self.pairs:
            x_f = pair['forward'](x)
            x_b = pair['backward'](x.flip([1])).flip([1])
            x = x_f + x_b
            # Handle residual connection...
        return x
```

### 4. MambaConfig for HuggingFace

```python
from transformers import MambaConfig

def create_mamba_config(ssamba_config):
    return MambaConfig(
        hidden_size=ssamba_config.embed_dim,
        state_size=ssamba_config.d_state,      # 16
        num_hidden_layers=1,                    # We manage layers ourselves
        expand=ssamba_config.expand,            # 2
        conv_kernel=ssamba_config.d_conv,       # 4
        use_bias=True,
        use_conv_bias=True,
        hidden_act="silu",
        residual_in_fp32=True,
        use_cache=False,
        use_mambapy=False,  # Use naive/slow path (no CUDA needed)
    )
```

## Open Questions

1. **Residual handling**: VisionMamba's Block has specific residual+norm patterns with `fused_add_norm`. Need to verify the non-fused path matches.

2. **Layer pairing**: Are SSAMBA's 24 layers truly 12 pairs, or is each layer used bidirectionally? Need to inspect checkpoint.

3. **Norm placement**: Is norm applied before or after the Mamba mixer? Pre-norm vs post-norm affects weight loading.

## Testing Strategy

1. **Unit test each component**:
   - `AudioPatchEmbed`: verify output shape
   - `RMSNorm`: compare against mamba_ssm RMSNorm
   - `MambaMixer`: single layer output comparison

2. **Integration test**:
   - Full model forward pass
   - Compare against original SSAMBA

3. **Numerical precision**:
   - Use float64 for comparison
   - Track where differences accumulate
