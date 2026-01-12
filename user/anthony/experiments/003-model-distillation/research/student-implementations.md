# Student Architecture Implementations

Code examples for each student architecture in Experiment 003.

## 1. Mamba2Student (HuggingFace transformers)

```bash
pip install transformers
```

```python
from transformers import Mamba2Config, Mamba2Model
import torch.nn as nn

class Mamba2Student(nn.Module):
    """
    Mamba2 for sequence contextualization.
    Uses inputs_embeds - no tokenizer needed.
    """
    def __init__(self, in_dim=768, hidden_size=256, num_layers=4, out_dim=768):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_size)

        config = Mamba2Config(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            state_size=64,
            head_dim=32,
            num_heads=hidden_size // 32,
            expand=2,
            conv_kernel=4,
            vocab_size=1,  # Dummy
        )
        self.mamba = Mamba2Model(config)
        self.proj_out = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        # x: (B, 248, 768)
        x = self.proj_in(x)
        x = self.mamba(inputs_embeds=x).last_hidden_state
        return self.proj_out(x)
```

| Config | hidden_size | layers | params |
|--------|-------------|--------|--------|
| Tiny | 256 | 2 | ~2M |
| Small | 256 | 4 | ~4M |
| Base | 384 | 6 | ~8M |

**Source**: [HuggingFace Mamba2](https://huggingface.co/docs/transformers/model_doc/mamba2)

---

## 2. ConvNeXt1DStudent (custom)

```python
import torch
import torch.nn as nn

class ConvNeXt1DStudent(nn.Module):
    """
    ConvNeXt-style 1D architecture with dilated depthwise convolutions.
    """
    def __init__(
        self,
        in_dim=768,
        hidden_dim=256,
        out_dim=768,
        num_blocks=6,
        kernel_size=7,
        expansion=4,
        dilations=[1, 2, 4, 8, 16, 32],
        drop_path=0.1,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ConvNeXt1DBlock(
                dim=hidden_dim,
                kernel_size=kernel_size,
                expansion=expansion,
                dilation=dilations[i % len(dilations)],
                drop_path=drop_path * i / num_blocks,
            )
            for i in range(num_blocks)
        ])
        self.proj_out = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(self.proj_out(x))


class ConvNeXt1DBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, expansion=4, dilation=1, drop_path=0.0):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size,
            padding=padding, dilation=dilation, groups=dim
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + self.drop_path(x)


class DropPath(nn.Module):
    """Stochastic depth."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask
```

| Config | hidden_dim | blocks | params | receptive field |
|--------|------------|--------|--------|-----------------|
| Tiny | 192 | 4 | ~1.2M | 91 |
| Small | 256 | 6 | ~2.5M | 379 |
| Base | 384 | 8 | ~6M | 1000+ |

---

## 3. TCNStudent (pytorch-tcn)

```bash
pip install pytorch-tcn
```

```python
from pytorch_tcn import TCN
import torch.nn as nn

class TCNStudent(nn.Module):
    """TCN wrapper for sequence contextualization."""
    def __init__(self, in_dim=768, hidden_dim=256, out_dim=768, num_blocks=6):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.tcn = TCN(
            num_inputs=hidden_dim,
            num_channels=[hidden_dim] * num_blocks,
            kernel_size=7,
            dilations=[1, 2, 4, 8, 16, 32],
            dropout=0.1,
            causal=False,
            use_norm='layer_norm',
            input_shape='NLC',
        )
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.tcn(x)
        return self.proj_out(x)
```

**Source**: [paul-krug/pytorch-tcn](https://github.com/paul-krug/pytorch-tcn)

---

## 4. MLPMixerStudent (custom)

```python
import torch.nn as nn

class MLPMixerStudent(nn.Module):
    """MLP-Mixer for sequence contextualization."""
    def __init__(
        self,
        in_dim=768,
        hidden_dim=256,
        out_dim=768,
        seq_len=248,
        num_blocks=4,
        token_expansion=0.5,
        channel_expansion=4,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            MixerBlock(hidden_dim, seq_len, token_expansion, channel_expansion)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        return self.proj_out(self.norm(x))


class MixerBlock(nn.Module):
    def __init__(self, dim, seq_len, token_expansion=0.5, channel_expansion=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(
            nn.Linear(seq_len, int(seq_len * token_expansion)),
            nn.GELU(),
            nn.Linear(int(seq_len * token_expansion), seq_len),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, dim * channel_expansion),
            nn.GELU(),
            nn.Linear(dim * channel_expansion, dim),
        )

    def forward(self, x):
        # x: (B, T, C)
        # Token mixing: transpose to (B, C, T), mix, transpose back
        y = self.norm1(x).transpose(1, 2)
        y = self.token_mix(y).transpose(1, 2)
        x = x + y
        # Channel mixing
        x = x + self.channel_mix(self.norm2(x))
        return x
```

| Config | hidden_dim | blocks | params |
|--------|------------|--------|--------|
| Tiny | 256 | 2 | ~2M |
| Small | 256 | 4 | ~4M |

---

## 5. TinyTransformerStudent (custom)

```python
import torch.nn as nn

class TinyTransformerStudent(nn.Module):
    """Small transformer for baseline comparison."""
    def __init__(
        self,
        in_dim=768,
        hidden_dim=256,
        out_dim=768,
        num_layers=2,
        num_heads=4,
        ff_dim=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.transformer(x)
        return self.proj_out(x)
```

| Config | hidden_dim | layers | heads | params |
|--------|------------|--------|-------|--------|
| Tiny | 256 | 2 | 4 | ~2M |
| Small | 384 | 4 | 6 | ~8M |

**Note**: O(n²) complexity, but 248 tokens is manageable.

---

## Summary

| Student | Library | Install | Complexity |
|---------|---------|---------|------------|
| Mamba2 | transformers | `pip install transformers` | O(n) |
| ConvNeXt1D | custom | - | O(n) |
| TCN | pytorch-tcn | `pip install pytorch-tcn` | O(n) |
| MLPMixer | custom | - | O(n) |
| TinyTransformer | custom | - | O(n²) |
