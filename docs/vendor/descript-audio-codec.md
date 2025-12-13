---
name: "descript-audio-codec"
source: "https://github.com/descriptinc/descript-audio-codec"
license: "MIT"
category: "tokenization"
pipeline_stage:
  - "tokenization"
relevance: "High-fidelity neural audio codec with improved RVQGAN; 90x compression at 44.1kHz"
status: "reference"
---

# Descript Audio Codec (DAC)

## Overview

Descript Audio Codec is a high-fidelity general neural audio codec that compresses 44.1kHz audio into discrete codes at 8 kbps, achieving approximately 90x compression while maintaining exceptional fidelity. It uses an improved RVQGAN architecture and works across all audio domains (speech, music, environment), making it a drop-in replacement for EnCodec in audio language modeling applications.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `dac.DAC` | Main codec model class | `dac/model/dac.py` |
| `dac.utils.download()` | Model weight downloading | `dac/utils/` |
| Encoder | Audio to latent/codes | `dac/model/` |
| Decoder | Codes to audio reconstruction | `dac/model/` |
| `DACFile` | Compressed file format (.dac) | `dac/` |

## Relevance to Pipeline

- **Tokenization**: Alternative to WavTokenizer for audio-to-discrete-code conversion; supports 16kHz, 24kHz, and 44.1kHz
- **Representation Learning**: Discrete codes can serve as targets for self-supervised learning
- **Inference**: Efficient compression/decompression for storage and transmission

## Usage

```python
import dac
from audiotools import AudioSignal

# Download and load model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)
model.to('cuda')

# Encode audio to codes
signal = AudioSignal('input.wav')
signal.to(model.device)
x = model.preprocess(signal.audio_data, signal.sample_rate)
z, codes, latents, _, _ = model.encode(x)

# Decode back to audio
y = model.decode(z)

# Or use compress/decompress for long files
compressed = model.compress(signal.cpu())
compressed.save("audio.dac")
decompressed = model.decompress(dac.DACFile.load("audio.dac"))
```

## Dependencies

- Python 3.x
- PyTorch
- audiotools
- einops

## Notes

- Three model variants: 16kHz, 24kHz, 44.1kHz
- 8 kbps bitrate vs EnCodec's 24 kbps (with better quality)
- Uses residual vector quantization (RVQ) with improved training
- Docker support available for reproducible environments
- CLI available: `python -m dac encode/decode`
