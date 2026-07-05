# Working Note: Can Tokens Compete?

**Status**: current cleanup anchor
**Updated**: 2026-07-04

## Framing

The BirdCLEF+ 2026 work is now framed as a comparison between a strong supervised leaderboard system and token/foundation-model representations for bioacoustic soundscapes.

Working-note title:

> **Can Tokens Compete? Token Representations against Supervised CNN Backbones for BirdCLEF+ 2026**

Central question:

> Can token-based audio representations compete with a practical supervised CNN/Perch system under the BirdCLEF+ 2026 labeled-soundscape and CPU-inference constraints?

## Answer in one paragraph

The strongest system is the supervised/pragmatic ensemble: frozen Perch v2 probes, an HGNetV2-B0 SED model, and a non-bird prototypical head reach **0.931 public / 0.936 private LB**. Token/foundation encoders remain scientifically useful, but they do not beat the specialist Perch-backed baseline in this setting. WavTokenizer codec tokens fail the bioacoustic semantic gates; Perch is the strongest frozen encoder; BEATs/EAT/AST/BirdNET/SSAMBA trail on deployed leaderboard probes.

## System pillar

Best system components:

| Component | Role |
|---|---|
| Frozen Perch v2 + MLP probes | Strong bird/soundscape representation without backbone training |
| HGNetV2-B0 SED | Supervised log-mel sound-event detector |
| Non-bird prototypical head | Uses Perch embeddings to recover signal for taxa outside Perch's bird-focused label space |
| Equal-weight blend | Combines Perch and SED probabilities within CPU budget |

Leaderboard ablation:

| Configuration | Public LB | Private LB |
|---|---:|---:|
| Perch v2 + ridge probe | 0.866 | 0.862 |
| Perch v2 + MLP probes | 0.920 | 0.918 |
| HGNetV2-B0 SED | 0.917 | 0.924 |
| Perch/HGNet MLP-only ensemble | 0.929 | 0.925 |
| **MLP ensemble + non-bird head** | **0.931** | **0.936** |

## Token/foundation pillar

### Codec tokens

WavTokenizer did not preserve bioacoustic semantics well enough for the proposed retrieval use:

| Gate | Target | Result |
|---|---:|---:|
| Round-trip Perch embedding cosine | >0.90 | 0.313 |
| Round-trip precision@1 | >0.80 | 0.09 |
| Focal retrieval precision@10 | >0.80 | 0.195 |

### Semantic encoders

Frozen ridge probes put every tested encoder on the same BirdCLEF 2026 yardstick:

| Encoder | LOSO AUROC | Public LB | Private LB |
|---|---:|---:|---:|
| **Perch** | **0.620** | **0.866** | **0.862** |
| BEATs | 0.612 | 0.777 | 0.760 |
| BirdNET | 0.588 | 0.814 | 0.798 |
| AST | 0.576 | 0.778 | 0.768 |
| EAT | 0.569 | 0.780 | 0.766 |
| SSAMBA | 0.563 | 0.731 | 0.689 |

Interpretation: offline LOSO rankings catch part of the story, but Kaggle deployment reveals a larger advantage for the specialist bioacoustic Perch model.

## Durable lessons

1. **Specialist bioacoustic pretraining matters.** Perch is both stronger and faster than the general encoders tested here.
2. **Leaderboard-aligned validation is hard.** Offline validation AUROC could anti-correlate with public LB during SED ablations.
3. **Codec tokens are not automatically semantic tokens.** Reconstruction quality for human-audio codecs does not imply species-discriminative bioacoustic retrieval.
4. **The token paradigm remains future-facing.** SSAMBA/token geometry, prompting, multimodality, and generative audio are plausible research directions, but not the source of the final score.
5. **Docs should track outcomes, not every brainstorm.** Superseded concept notes should be deleted or collapsed into experiment results.

## Source-of-truth policy

- Project-level state: [../STATUS.md](../STATUS.md)
- Concept index: [README.md](README.md)
- Experiment evidence: `../../user/<name>/experiments/*/results.md`
- Publication prose: keep the repo-level docs standalone enough to understand the paper without relying on external local paths.
