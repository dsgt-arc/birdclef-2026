---
name: "WavTokenizer"
source: "https://github.com/jishengpeng/WavTokenizer"
license: "MIT"
category: "tokenization"
pipeline_stage:
  - "tokenization"
relevance: "SOTA neural audio codec achieving 40-75 tokens/second with rich semantic information"
status: "active"
---

# WavTokenizer

## Overview

WavTokenizer is a state-of-the-art discrete audio codec that achieves extreme compression (40-75 tokens per second) while maintaining high reconstruction fidelity. It uses a single codebook design with large vocabulary (4096 tokens) and is specifically designed for audio language modeling applications like GPT-4o-style models. Published at ICLR 2025.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `WavTokenizer` | Main model class for encoding/decoding | `decoder/pretrained.py` |
| Encoder | Audio to discrete codes conversion | `encoder/` |
| Decoder | Discrete codes to audio reconstruction | `decoder/` |
| Configs | Model configurations (small/medium/large) | `configs/` |

## Relevance to Pipeline

- **Tokenization**: Primary candidate for Phase 1 birdcall tokenization; offers 40-75 tokens/second which is more compact than traditional codecs
- **Representation Learning**: Discrete codes contain rich semantic information suitable for language model-style training
- **Inference**: Single codebook design simplifies token prediction compared to multi-codebook RVQ approaches

## Usage

```python
from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

# Load model
config_path = "./configs/wavtokenizer_large.yaml"
model_path = "./wavtokenizer_large.ckpt"
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)

# Encode audio to discrete tokens
wav, sr = torchaudio.load("audio.wav")
wav = convert_audio(wav, sr, 24000, 1)  # Resample to 24kHz mono
bandwidth_id = torch.tensor([0])

features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)

# Decode back to audio
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
```

## Dependencies

- Python 3.9
- PyTorch
- torchaudio
- PyTorch Lightning (for training)

## Notes

- Models available on HuggingFace: `novateur/WavTokenizer`
- Three variants: small (600/320 hop), medium, large
- Operates at 24kHz sample rate
- Large model trained on 80,000 hours covering speech, audio, and music
- Compression factor ~90x compared to raw audio
- Better reconstruction quality than EnCodec while using fewer tokens
