# Project Status

> **Last updated**: 2026-07-04

## Phase

BirdCLEF+ 2026 competition work is complete. The repository is now being cleaned up around the working-note narrative and the experiment record.

## Headline result

Best submitted system: **0.931 public / 0.936 private leaderboard**.

| Component | Public LB | Private LB | Notes |
|---|---:|---:|---|
| Perch v2 + ridge probe | 0.866 | 0.862 | Frozen embedding baseline |
| Perch v2 + MLP probes | 0.920 | 0.918 | Frozen Perch, learned probe heads |
| HGNetV2-B0 SED, 4-fold | 0.917 | 0.924 | Log-mel SED member |
| Perch/HGNet ensemble, MLP-only | 0.929 | 0.925 | Equal-weight Perch + SED blend |
| **MLP ensemble + non-bird prototypical head** | **0.931** | **0.936** | Best final system |

## Current paper framing

Working-note title: **Can Tokens Compete? Token Representations against Supervised CNN Backbones for BirdCLEF+ 2026**.

The paper has two pillars:

1. **Competitive supervised baseline** — a pragmatic Perch/HGNet/prototype ensemble that fits the labeled soundscape setting and stays within the CPU inference budget.
2. **Token representation study** — codec tokens, semantic audio encoders, and SSAMBA-style sequence models evaluated against the supervised baseline.

## Key findings

- The final leaderboard score came from **Perch v2 probes + HGNetV2-B0 SED + non-bird prototypes**, not from a novel token model.
- **Frozen Perch v2** transferred best to the BirdCLEF 2026 labeled-soundscape task among tested encoders.
- **General audio encoders** remained useful but below Perch on deployed LB: BEATs, AST, EAT, BirdNET, and SSAMBA all trailed the Perch ridge baseline in the common frozen-probe deployment.
- **WavTokenizer codec tokens failed** the bioacoustic retrieval gates: round-trip Perch cosine 0.313 vs >0.90 target, precision@1 0.09 vs >0.80 target, focal precision@10 0.195 vs >0.80 target.
- **Offline validation was unreliable** for leaderboard decisions; 20+ single-variable SED submissions showed validation AUROC could anti-correlate with public LB.
- **Pseudo-labeling and many recipe sweeps regressed** in this setting; the reliable SED gains were frontend/architecture changes.

## Encoder comparison snapshot

Frozen ridge probes deployed to the leaderboard:

| Encoder | Dim | LOSO AUROC | Public LB | Private LB | CPU min / 1k clips |
|---|---:|---:|---:|---:|---:|
| **Perch** | 1536 | **0.620** | **0.866** | **0.862** | **11.5** |
| BEATs | 768 | 0.612 | 0.777 | 0.760 | 29.6 |
| BirdNET | 1024 | 0.588 | 0.814 | 0.798 | 34.5 |
| AST | 768 | 0.576 | 0.778 | 0.768 | 38.2 |
| EAT | 768 | 0.569 | 0.780 | 0.766 | 25.5 |
| SSAMBA | 384 | 0.563 | 0.731 | 0.689 | 72.5 |

## Documentation cleanup policy

This repo should stay mostly standalone. Keep concise project docs here; keep detailed evidence in experiment folders; prefer marking concept status or re-indexing exploratory notes over deleting them.

## Where to look

| Need | Location |
|---|---|
| Paper framing | [concepts/working-note.md](concepts/working-note.md) |
| Documentation index | [README.md](README.md) |
| Operational notes | [operations.md](operations.md) |
| Experiment evidence | `user/<name>/experiments/*/results.md` |
| Assistant rules / venv policy | [../AGENTS.md](../AGENTS.md) |
