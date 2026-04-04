# Tasks

## 1. Setup
- [x] 1.1 Download Perch v2 ONNX from https://huggingface.co/justinchuby/Perch-onnx (done in job 6275860; cached in HF hub)
- [x] 1.2 Inspect ONNX graph — **frame features are already an output**, `spatial_embedding` shape `[B, 16, 4, 1536]`; no graph surgery needed. See `artifacts/onnx_inspection/`.
- [ ] 1.3 Confirm exp 102 Run N offline cmAP number (baseline to beat)
- [ ] 1.4 Set up experiment package skeleton (`pyproject.toml`, `src/`, `sbatch/`, `configs/`) mirroring exp 102 layout
- [ ] 1.5 Verify labeled soundscape window split matches exp 102 for apples-to-apples comparison

## 2. Implementation
- [x] 2.1 ~~Graph-surgery script~~ — **not needed**: `spatial_embedding` is a stock output of `perch_v2_no_dft.onnx`. Just request it from `sess.run()`.
- [x] 2.2 ~~Sanity-check tapped model~~ — covered by job 6275860 forward pass: all four outputs returned with expected shapes on a zero-input batch.
- [ ] 2.3 Feature extraction pipeline: iterate labeled windows at 32 kHz (160000 samples/window), run stock Perch ONNX, cache `(embedding [1536], spatial_embedding [16, 4, 1536], label)` to shared storage
- [ ] 2.4 ~~Estimate full cache size~~ — concrete from graph: ~393 KB/window fp32 or ~196 KB fp16 for `spatial_embedding`; ~16× the pooled embedding, not 50–100×.
- [ ] 2.5 Head (a): linear probe on pooled 5s embedding
- [ ] 2.6 Head (b): attention pool over frame features + linear classifier
- [ ] 2.7 Head (c): 2-layer transformer over frame features + attention pool + classifier
- [ ] 2.8 Shared training loop: BCE/focal loss, same optimizer/schedule/split as exp 102, TensorBoard logging

## 3. Execution
- [ ] 3.1 Run feature cache build on RTX 6000 or L40S (sbatch, shared storage output)
- [ ] 3.2 Train head (a) — linear probe
- [ ] 3.3 Train head (b) — attention pool
- [ ] 3.4 Train head (c) — tiny transformer
- [ ] 3.5 Log window-level cmAP, bird-only cmAP, amphibian-only cmAP for each head

## 4. Analysis
- [ ] 4.1 Compare all three heads against exp 102 Run N offline cmAP
- [ ] 4.2 Isolate the "temporal resolution" contribution: (c) vs (a)
- [ ] 4.3 Amphibian-only cmAP sanity check — decide whether Perch is viable for this competition at all
- [ ] 4.4 If winning head exists, run one Kaggle LB submission
- [ ] 4.5 Document findings in `results.md` via `/experiment:result`
- [ ] 4.6 Update `docs/STATUS.md` and memory index
