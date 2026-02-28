# EAT Patch Embedding Architecture

## PatchEmbed_new (local_encoder)

Located in cached HF model: `worstchan/EAT-base_epoch30_pretrain`
File: `model_core.py`

```python
class PatchEmbed_new(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=16):
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x = self.proj(x)                      # (B, 1, 512, 128) → (B, 768, 32, 8)
        x = x.flatten(2).transpose(1, 2)      # → (B, 256, 768)
        return x
```

**It's a single Conv2d** — not a multi-layer CNN. Weights shape: `(768, 1, 16, 16)`.

## EAT-base Config Defaults
- `img_size`: (1024, 128) — default; project uses (512, 128) for 5s clips
- `patch_size`: 16, `stride`: 16 (non-overlapping)
- `in_chans`: 1, `embed_dim`: 768
- For 5s clips: 512/16 × 128/16 = 32×8 = **256 tokens**

## Full encode() Pipeline
```
local_encoder(x)           → (B, 256, 768) patch embeddings
+ sincos 2D pos encoding   → positional info for 32×8 grid
cat CLS token              → (B, 257, 768)
pre_norm (LayerNorm)
pos_drop (Dropout)
12× AltBlock               → transformer layers
```

## Model Hierarchy
```
EATModel (PreTrainedModel)
└── .model (EAT)
    ├── .local_encoder (PatchEmbed_new)
    │   └── .proj: Conv2d(1, 768, 16, 16)  — ~196K params
    ├── .extra_tokens (CLS token, shape 1×1×768)
    ├── .fixed_positional_encoder (sincos 2D)
    ├── .pre_norm (LayerNorm(768))
    ├── .pos_drop (Dropout)
    └── .blocks (12× AltBlock)
        ├── .attn: qkv Linear(768, 2304), proj Linear(768, 768)
        └── .mlp: fc1 Linear(768, 3072), fc2 Linear(3072, 768)
```

## Key Insight for Frame Tokenization
- Changing patch_size from 16×16 to 128×W changes Conv2d kernel shape
- Pretrained weights (768, 1, 16, 16) CANNOT be reused for (768, 1, 128, W)
- Patch embedding must be randomly initialized and fully trained
- Positional encoding must be regenerated for new grid (1×256 for 128×2, 1×128 for 128×4)
- Transformer blocks are shape-agnostic (process any sequence length of 768-dim tokens)
