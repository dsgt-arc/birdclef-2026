# Tasks

## 1. Setup
- [ ] 1.1 Download Perch v2 ONNX from https://huggingface.co/justinchuby/Perch-onnx into shared storage (not home dir)
- [ ] 1.2 Inspect ONNX graph in Netron; identify the pre-pool tensor (node name, shape, dtype)
- [ ] 1.3 Confirm exp 102 Run N offline cmAP number (baseline to beat)
- [ ] 1.4 Set up experiment package skeleton (`pyproject.toml`, `src/`, `sbatch/`, `configs/`) mirroring exp 102 layout
- [ ] 1.5 Verify labeled soundscape window split matches exp 102 for apples-to-apples comparison

## 2. Implementation
- [ ] 2.1 Write ONNX graph-surgery script to re-export Perch v2 with pre-pool tensor as an additional output (onnx-graphsurgeon or `onnx.utils.extract_model`)
- [ ] 2.2 Sanity-check tapped model: final output matches original bit-for-bit on a few clips; frame-feature shape is (C, T') as expected
- [ ] 2.3 Feature extraction pipeline: iterate labeled windows, run tapped ONNX, cache (pooled_embedding, frame_features, label) to shared storage
- [ ] 2.4 Estimate full cache size from first batch; adjust storage plan if needed
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
