# Concepts

Research ideas, hypotheses, and framing notes for BirdCLEF+ 2026.

These documents are not all current plans. Treat them as a research log: some are active, some are historical, and a small number record approaches that failed or were shelved. The purpose of this index is to separate those cases without deleting useful provenance.

## Current anchors

| Concept | Status | Purpose |
|---|---|---|
| [working-note](working-note.md) | current | Paper framing, result map, and cleanup anchor |
| [strong-baseline-mel-overhaul](strong-baseline-mel-overhaul.md) | current/historical | Path to the strong SED member and frontend lessons |
| [soundscape-domain-shift](soundscape-domain-shift.md) | current | Persistent focal→soundscape transfer problem |
| [ssamba-audio-mamba](ssamba-audio-mamba.md) | current research thread | SSAMBA/Mamba architecture and experiment summary |
| [temporal-resolution-sequence-length](temporal-resolution-sequence-length.md) | current research thread | Token geometry, context length, and CPU scaling |
| [audio-pipeline-bottlenecks](audio-pipeline-bottlenecks.md) | operational | Throughput and data-loading bottlenecks |

## Historical or superseded, but useful

| Concept | Why keep it |
|---|---|
| [data-and-compute-scale](data-and-compute-scale.md) | Dataset/compute scale context |
| [research-leads](research-leads.md) | Early model/paper triage |
| [lora-finetuning-audio](lora-finetuning-audio.md) | Documents the LoRA approach used in early encoder experiments |
| [ssast-audio-transformer](ssast-audio-transformer.md) | Transformer baseline context for SSAMBA work |
| [supervised-vit-audio-pipeline](supervised-vit-audio-pipeline.md) | Historical ViT baseline planning/results context |
| [vit-mae-bioacoustics-pretraining](vit-mae-bioacoustics-pretraining.md) | Deferred MAE planning artifact |
| [pretraining-compute-feasibility](pretraining-compute-feasibility.md) | Compute-budget reasoning for SSL/pretraining choices |
| [mlp-mixer-distillation](mlp-mixer-distillation.md) | Earlier CPU-student idea; superseded by later evidence |

## Exploratory / future-work seeds

These are not final claims. Keep them because they capture hypotheses, design constraints, or partially explored directions.

| Concept | Notes |
|---|---|
| [acoustic-atom-basis](acoustic-atom-basis.md) | Prototype/basis idea for low-resource species |
| [bi-mamba-1bit-ssm](bi-mamba-1bit-ssm.md) | Speculative efficient SSM direction |
| [cnn-mamba-hybrid-frontend](cnn-mamba-hybrid-frontend.md) | Frontend idea for SSAMBA-like models |
| [context-recovery](context-recovery.md) | Context-loss framing from static-token experiments |
| [cross-attention-factored-hopfield](cross-attention-factored-hopfield.md) | Hopfield/prototype retrieval design thread |
| [distilled-sequence-model](distilled-sequence-model.md) | Early distillation design |
| [multi-stream-ssamba](multi-stream-ssamba.md) | Multi-stream codec/SSAMBA variant |
| [pretrained-mamba-transfer](pretrained-mamba-transfer.md) | Transfer mechanics for Mamba audio models |
| [prompted-sound-event-detection](prompted-sound-event-detection.md) | Reference-conditioned SED idea; evidence inconclusive |
| [source-separation-augmentation](source-separation-augmentation.md) | Separation as augmentation/data-engine idea |
| [source-separation-training](source-separation-training.md) | MixIT separator training plan |
| [star-schema-tensor-decomposition](star-schema-tensor-decomposition.md) | Star-schema/Hopfield formalism |
| [unsupervised-event-detection-labeling](unsupervised-event-detection-labeling.md) | Event-labeling idea from weak labels/DSP |

## Falsified, regressed, or shelved threads

Only a few concepts are actually falsified or partially falsified. Keep them as negative-result records.

| Concept | Verdict |
|---|---|
| [codec-motif-search](codec-motif-search.md) / [discrete-audio-token-sequence-model](discrete-audio-token-sequence-model.md) | WavTokenizer-specific route failed semantic/retrieval gates in exp 104; broader codec-token idea is not proven impossible |
| [semi-supervised-box-annotation-loop](semi-supervised-box-annotation-loop.md) | Insect routing/propagation gate failed; thread shelved for that target |
| [weak-supervision-label-fusion](weak-supervision-label-fusion.md) | Direct use of fused/soft labels regressed downstream models; useful as a cautionary record |

## Cleanup rule

Prefer re-indexing or marking status over deleting exploratory docs. Delete only when a note is duplicated elsewhere, contains no durable information, or has been fully absorbed into an experiment result.
