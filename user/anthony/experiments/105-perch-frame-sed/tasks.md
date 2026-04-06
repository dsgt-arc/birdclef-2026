# Tasks

## 1. Setup
- [x] 1.1 Download Perch v2 ONNX (job 6275860; cached in HF hub)
- [x] 1.2 Inspect ONNX graph — `spatial_embedding` [B, 16, 4, 1536] is a stock output, no surgery needed
- [ ] 1.3 Confirm exp 102 Run N offline cmAP (still pending)
- [x] 1.4 Package skeleton (pyproject.toml, src/, sbatch/, configs/)
- [x] 1.5 Verify split matches exp 102 (seed=42 file-level 50/50 soundscape, stratified focal)

## 2. Implementation
- [x] 2.1 Graph surgery — NOT NEEDED
- [x] 2.2 Sanity check — covered by job 6275860
- [x] 2.3 Feature extraction: soundscape (job 6284733) + 8-shard focal array (job 6285722)
  - Cache includes: embedding fp16, spatial_embedding fp16, perch_logits fp16 (14795-d), labels int8
  - 57 GB total on scratch (8 focal shards + soundscape shard)
- [x] 2.4 Cache sizes verified: ~7 GB/focal shard, 158 MB soundscape
- [x] 2.5 Head (a): LinearHead(1536→206) — 316K params
- [x] 2.6 Head (b): PerchSEDHead (GeM + projection + dual Conv1d + attention) — 998K params
- [ ] 2.7 Head (c): full-clip sequence model (192 time steps from 12×5s windows) — NEXT
- [x] 2.8 Training loop: FocalBCELoss, dual clip+frame loss for SED, streaming shard dataloader

## 3. Execution
- [x] 3.1 Focal feature cache build (8-shard array, ~25 min wall, job 6285722)
- [x] 3.2 Train linear probe on focal (job 6290291) — **val/cmap 0.684, val/sc_cmap 0.052**
- [x] 3.3 Train SED head on focal (job 6290938) — **val/cmap 0.695, val/sc_cmap 0.072**
- [ ] 3.4 Train full-clip sequence model — NEXT
- [ ] 3.5 Per-taxon cmAP breakdown (bird vs amphibian) — not yet done

## 4. Analysis
- [x] 4.1 Linear vs SED: SED wins marginally on focal (+0.011) and meaningfully on soundscape (+0.020)
- [x] 4.2 Frame resolution contribution: small on focal (signal is everywhere), larger on soundscape (brief calls)
- [ ] 4.3 Amphibian cmAP — not yet computed
- [ ] 4.4 Kaggle LB submission — not yet done
- [x] 4.5 Results documented in this file + memory
- [ ] 4.6 Update docs/STATUS.md

## 5. Next direction: full-clip sequence model
- [ ] 5.1 Dataset that groups windows by recording, concatenates spatial features (12×16=192 time steps for 60s clips)
- [ ] 5.2 PerchSEDHead on (B, 192, 4, 1536) — no architecture change needed, just longer sequence
- [ ] 5.3 Clip-level labels (union of window labels)
- [ ] 5.4 Train on soundscape clips (in-domain!) + focal recordings
- [ ] 5.5 Compare val/sc_cmap against per-window SED baseline (0.072)
