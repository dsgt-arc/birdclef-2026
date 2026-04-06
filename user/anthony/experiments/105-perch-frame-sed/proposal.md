# Proposal: Perch v2 Frame-Level SED

**Status**: in-progress
**Created**: 2026-04-04
**Author**: anthony

## Hypothesis

A small sequence head trained on **pre-pool frame features** from a frozen Perch v2 backbone will beat exp 102 Run N (frozen EfficientNet-ImageNet + SED head) on window-level cmAP on the BirdCLEF 2026 labeled soundscape subset, because (a) Perch is bird-domain pretrained and (b) the pre-pool feature map preserves sub-window temporal structure (~hundreds of ms per step) that is lost in the standard 5s global-pooled embedding.

Stronger form: the frame-level head will also beat a linear probe on Perch's pooled 5s embedding, demonstrating that the additional temporal resolution — not just the backbone — is the source of the lift.

## Background

- **Exp 102 (EfficientNet SED)**: fine-tuning the ImageNet backbone on focal recordings consistently regresses Kaggle LB (Run K 0.609 → L 0.528 → M 0.553) even as offline metrics improve. Frozen-backbone Run N is the current bet. Takeaway: backbone choice matters and backbone fine-tuning is dangerous.
- **Exp 007 (Perch linear probe / AudioSet SSAMBA transfer)**: prior Perch-family linear probes landed around 0.173 cmAP with older weights — the frozen-feature-extractor recipe is viable but the quality ceiling depends heavily on backbone domain match.
- **Exp 104 (WavTokenizer retrieval)**: already uses Perch v2 ONNX as the downstream evaluator, so the inference plumbing exists in this repo.
- **Concept**: Perch's global pool collapses ~200 time steps/minute of CNN feature-map structure into 12 vectors/minute. Sound event detection literature consistently finds frame-level features beat pooled embeddings for temporally-resolved tasks. Combining a strong frozen bird encoder with an SED-style sequence head is a natural — and under-explored in this repo — combination.

Related file: `user/anthony/experiments/102-efficientnet-sed/` (baseline SED architecture to mirror).

## Method

### Approach

Tap Perch v2's pre-pool feature map (the CNN activation just before global average pooling), cache frame features for the training + validation soundscapes, and train three lightweight heads on the 0.6% labeled soundscape subset:

- **(a) Linear probe** on the pooled 5s embedding — controls for "backbone swap alone".
- **(b) Attention pool** on frame features — controls for "small head, but with temporal resolution".
- **(c) Tiny 2-layer transformer** on frame features — full sequence model.

Backbone is frozen at all times; only heads are trained. Gradients never flow into Perch (ONNX inference is sufficient).

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2026 train soundscapes, 0.6% labeled window subset for supervised loss, teacher pseudo-labels from exp 101 for auxiliary distillation (optional, phase 2) |
| Compute | ONNX feature extraction: 1× RTX 6000 or L40S, ~few hours for full cache. Head training: CPU-feasible or single GPU for <1h per head. No long SLURM jobs needed. |
| Dependencies | `onnxruntime-gpu`, `onnx`, `onnx-graphsurgeon` (for graph tap), Perch v2 ONNX from https://huggingface.co/justinchuby/Perch-onnx, project torch stack |
| Code | New package `user/anthony/experiments/105-perch-frame-sed/`. Reuses labeled window loader from exp 101/102. |
| Storage | Frame feature cache lives on shared project storage, not home dir. Estimated 50–100× the size of the pooled-embedding cache (rough: low hundreds of GB for full train set; verify once tap shape is known). |

### Procedure

1. **Graph inspection** — Load the Perch v2 ONNX in Netron and/or `onnx.load`; identify the node immediately preceding the global-average-pool. Record its name, shape, and dtype.
2. **ONNX tap** — Use `onnx.utils.extract_model` or `onnx-graphsurgeon` to re-export a variant ONNX that exposes the pre-pool tensor as an additional output. Fall back to re-exporting from upstream Perch v2 (TF/JAX) if the ONNX graph doesn't admit a clean tap.
3. **Sanity check** — Run the tapped model on a handful of train clips; verify the frame-feature tensor has the expected (C, T') shape and that the original final output is unchanged bit-for-bit.
4. **Feature cache** — Batch-extract both (i) pooled 5s embeddings and (ii) frame features for all labeled soundscape windows. Cache on shared storage in an efficient format (e.g. npy / memmap / parquet).
5. **Head (a) — linear probe** on pooled embeddings. Establishes the "Perch backbone alone" baseline.
6. **Head (b) — attention pool** over frame features. Single learned attention query pooling followed by a linear classifier.
7. **Head (c) — tiny transformer** (2 layers, small dim, e.g. 128–256) over frame features, followed by attention pool + classifier.
8. **Evaluate** all three on the held-out labeled soundscape windows (same split as exp 102) and compare to exp 102 Run N.
9. **Phase 2 (conditional)** — If (c) wins cleanly, add teacher-label distillation on the unlabeled soundscape pool and/or evaluate amphibian-only cmAP to probe the domain-mismatch risk. If it loses, write up negative result and stop.

### Variables

- **Independent**: head architecture (linear / attention pool / transformer), feature granularity (pooled 5s embedding vs pre-pool frame feature map).
- **Dependent**: window-level cmAP, per-taxon (bird vs amphibian) cmAP, head parameter count, training time.
- **Controlled**: backbone (frozen Perch v2), train/val split, loss function, optimizer, training budget, input preprocessing (matches Perch's expected mel-spec pipeline).

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Window-level cmAP | Primary metric, on the 0.6% labeled soundscape subset. Matches exp 102 offline eval. |
| Bird-only cmAP | Probes whether Perch's domain advantage is where we expect it. |
| Amphibian-only cmAP | Risk check: does Perch's bird-trained backbone transfer to frogs at all? |
| Kaggle LB score | Secondary, only for the winning head (one submission budgeted). |
| Frame-feature cache size | Practical feasibility for future iteration. |

### Baseline

- **Primary**: exp 102 Run N offline window-level cmAP (frozen EfficientNet-ImageNet + SED head). Number to be filled in once Run N completes.
- **Secondary**: exp 102 Run M-v2 Kaggle LB 0.586 (current best submitted offline-improving variant).
- **Internal control**: head (a) Perch-pooled linear probe vs head (b/c) frame-level heads — isolates the "temporal resolution" lift from the "better backbone" lift.

### Success Criteria

- **Confirm if**: head (c) beats exp 102 Run N offline cmAP by ≥0.02 absolute AND head (c) beats head (a) by ≥0.01 absolute (i.e. frame resolution contributes, not just the backbone).
- **Partial win**: head (a) or (b) beats exp 102 Run N by ≥0.02 — still useful (cheaper Perch recipe) but weakens the frame-level hypothesis; would not justify the cache cost.
- **Reject if**: no head beats exp 102 Run N within 0.02, OR amphibian-only cmAP collapses (<50% of the bird-only cmAP), indicating Perch doesn't generalize to the competition's dominant taxon.

## Findings from graph inspection (2026-04-04, job 6275860)

Running `scripts/inspect_onnx.py` on `justinchuby/Perch-onnx :: perch_v2_no_dft.onnx` revealed that the frame-level feature map is **already a named output of the stock ONNX**. No graph surgery is required.

**Model interface**:

| Tensor | Shape | Notes |
|--------|-------|-------|
| input `inputs` | `[batch, 160000]` float | Raw waveform, 5s @ 32 kHz |
| output `embedding` | `[batch, 1536]` float | Pooled 5s embedding (derived from `spatial_embedding` via reduce_sum + div, nodes 1135–1136) |
| output `spatial_embedding` | `[batch, 16, 4, 1536]` float | **The pre-pool frame feature map we wanted** — 16 time × 4 freq × 1536 channels per 5s window (~312 ms per time step, ~192 time steps per 60s clip) |
| output `spectrogram` | `[batch, 500, 128]` float | 128-mel, 500 frames/5s — free debugging output |
| output `label` | `[batch, 14795]` float | Full Perch classifier logits |

**Size**: 394 MB on disk, ~103M params, opset 18, 1137 nodes.

**Cache footprint** (tighter than initial estimate):
- Pooled `embedding`: 1536 × 4 B = 6 KB/window fp32
- `spatial_embedding`: 16 × 4 × 1536 × 4 B ≈ 393 KB/window fp32 (≈ 196 KB fp16)
- Frame cache is ~16× the pooled cache, not 50–100× as feared. Full train set is comfortably feasible on shared project storage.

**Consequences for the plan**:
- Tasks 2.1 (graph surgery) and 2.2 (tap-model sanity check) are obviated.
- Feature extraction is "call `sess.run(['embedding', 'spatial_embedding'], {'inputs': wav})`" — no custom ONNX edit path.
- The `spectrogram` output is a free debugging/visualization artifact; we can log mel specs without a separate librosa path.
- Input sample rate is confirmed as 32 kHz (160000 samples / 5s), matching the project's existing audio pipelines.

Artifacts: `artifacts/onnx_inspection/{summary.json, nodes.txt, candidates.txt}`, log `logs/inspect-onnx-6275860.out`.

## Results (2026-04-06)

### Per-window heads (trained on 239,711 focal windows, validated on 26,213 focal + 369 soundscape)

| Head | Params | val/cmap (focal) | val/sc_cmap (soundscape) | Best epoch | Job |
|------|--------|------------------|--------------------------|------------|-----|
| LinearHead(1536→206) | 316K | **0.684** | 0.052 | 10 | 6290291 |
| PerchSEDHead (GeM+proj+dual Conv1d) | 998K | **0.695** | **0.072** | 17 | 6290938 |

### Auxiliary comparisons

| Run | Head | Train data | val/cmap | val/sc_cmap | Notes |
|-----|------|-----------|----------|-------------|-------|
| Pilot | Linear | 59 soundscape | 0.011 | — | Pipeline smoke test |
| Soundscape-only | Linear | 352 soundscape | 0.059 | — | In-domain, tiny |
| Exp 012 | Linear | focal | 0.336 | — | EfficientNet-B1-BirdSet frozen, 0.649 Kaggle LB |
| Exp 007 | Linear | focal | 0.173 | — | SSAMBA AudioSet pretrained |

### Interpretation

1. **Perch pooled embedding doubles EfficientNet**: 0.684 vs 0.336 focal cmAP for the same linear probe setup.
2. **SED head barely helps on focal** (+0.011): signal is everywhere in clean focal recordings, so attention pooling ≈ mean pooling. The 16×4 patch grid is too coarse for fine-grained attention to add much.
3. **SED helps more on soundscape** (+0.020, 38% relative): framewise-max catches brief calls that get diluted in the mean. But the improvement is small in absolute terms.
4. **Domain shift is the dominant bottleneck**: 0.695 focal → 0.072 soundscape. Frozen Perch features trained on close-mic focal audio don't generalize well to far-field noisy soundscapes. No head architecture can fix this.
5. **In-domain training beats out-of-domain scale**: soundscape-only linear probe (0.059) nearly matches focal-trained linear probe on soundscape (0.052), despite 700× less data.

### Next direction

Full-clip sequence model: concatenate all 12 windows of a 60s soundscape into one 192-step sequence. The SED head is sequence-length agnostic (Conv1d kernel=1 + softmax attention). This gives the model temporal context across the full clip — call bout patterns, noise rejection, and matches the Kaggle eval format. Training on labeled soundscape clips (in-domain) rather than focal may also help.

## Limitations

- **Amphibian coverage**: BirdCLEF 2026 is amphibian-dominated (exp 100 EDA). Perch v2 is trained on Xeno-canto *birds*. There is a real risk that Perch's features are uninformative for frog species regardless of head architecture. Amphibian cmAP is the explicit check — **not yet computed**.
- **Domain shift**: confirmed as the dominant problem. Perch is trained on focal recordings; soundscapes are far-field with overlap. Frozen Perch features hit a ceiling of ~0.07 sc_cmap regardless of head.
- **ONNX tap risk**: RESOLVED — `spatial_embedding` is a stock output.
- **Frozen backbone only**: no end-to-end fine-tuning possible via ONNX. Caps the soundscape ceiling.
- **Storage cost**: 57 GB on scratch (8 focal shards + soundscape). Manageable.
- **Perch `label` output is NOT free**: requesting the 14795-d classifier head slows Perch from ~12 to ~4 win/s.
- **CPU training is slow**: SED head takes ~30 min/epoch on cpu-small. GPU would be 100× faster.
- **Exp 102 Run N baseline is still pending** — the direct comparator hasn't landed yet.
