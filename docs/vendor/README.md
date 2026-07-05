# External Project Notes

This directory documents external repositories, datasets, and model families used as implementation references during BirdCLEF 2026. A document here does **not** imply the upstream repository is checked out under `vendor/`.

See [`../../vendor/README.md`](../../vendor/README.md) for the small set of submodules that are still retained because code or reproducibility scripts import them directly.

## Repository Summary

| Repository | Category | Pipeline Stage | Local checkout? | Status |
|------------|----------|----------------|-----------------|--------|
| [bambird + scikit-maad](bambird-scikit-maad.md) | utility | data-processing | `vendor/bambird`, `vendor/scikit-maad` | legacy active |
| [BirdNET-Analyzer](birdnet-analyzer.md) | pretrained-model | inference, embedding | no | reference |
| [BirdSet](birdset.md) | dataset | data-processing, representation-learning | no | reference |
| [ProtoCLR](protoclr.md) | self-supervised | representation-learning | no | reference |
| [Wav2vec2.0](wav2vec2.md) | self-supervised | tokenization, representation-learning | no | reference |
| [WavTokenizer](wavtokenizer.md) | tokenization | tokenization | `vendor/WavTokenizer` | legacy active |
| [best-rq-pytorch](best-rq-pytorch.md) | self-supervised | tokenization, representation-learning | no | reference |
| [bioacoustics-model-zoo](bioacoustics-model-zoo.md) | utility | inference, embedding | `vendor/bioacoustics-model-zoo` | legacy active |
| [birdclef-2025](birdclef-2025.md) | utility | tokenization, embedding, inference | no | reference |
| [descript-audio-codec](descript-audio-codec.md) | tokenization | tokenization | no | reference |
| [dinosr](dinosr.md) | self-supervised | representation-learning | no | reference |
| [hubert](hubert.md) | self-supervised | tokenization, representation-learning | no | reference |
| [model2vec](model2vec.md) | self-supervised | embedding | no | reference |
| [opensoundscape](opensoundscape.md) | utility | data-processing, inference | no | reference |
| [perch](perch.md) | pretrained-model | embedding, inference | no; use package/ONNX/checkpoints | active reference |
| [AST](ast.md) | pretrained-model | embedding, inference | no; use HuggingFace/upstream | active reference |
| [EAT](eat.md) | self-supervised | embedding | no; use HuggingFace/upstream | active reference |
| [BEATs](beats.md) | pretrained-model | embedding | `vendor/unilm` | legacy active |
| [SSAST](ssast.md) | self-supervised | embedding | `vendor/ssast` | legacy active |
| [SSAMBA](ssamba.md) | self-supervised | embedding | `vendor/ssamba` | legacy active |

## Pipeline Mapping

### Tokenization

- **[WavTokenizer](wavtokenizer.md)**: neural codec with 40-75 tokens/second.
- **[descript-audio-codec](descript-audio-codec.md)**: high-fidelity RVQGAN codec reference.
- **[best-rq-pytorch](best-rq-pytorch.md)** and **[hubert](hubert.md)**: discrete-unit SSL references.

### Encoders / Inference

- **[perch](perch.md)**: strongest frozen bioacoustic embedding baseline in this project.
- **[BEATs](beats.md)**, **[AST](ast.md)**, **[EAT](eat.md)**, **[SSAST](ssast.md)**, **[SSAMBA](ssamba.md)**: general/foundation audio encoder comparisons.
- **[BirdNET-Analyzer](birdnet-analyzer.md)** and **[bioacoustics-model-zoo](bioacoustics-model-zoo.md)**: species-model and API references.

### Data & Evaluation

- **[BirdSet](birdset.md)**: large-scale bioacoustics benchmark reference.
- **[ESC-50](esc50.md)**: sanity-check dataset used by legacy encoder benchmarks.

## Status Definitions

- **legacy active**: still has code or scripts that may rely on a retained local checkout.
- **active reference**: still important to the project, but obtained via package, model hub, ONNX/checkpoint artifact, or upstream URL.
- **reference**: conceptual/provenance documentation only.
