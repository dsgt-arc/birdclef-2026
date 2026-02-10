# Tasks: 007 Pre-trained SSAMBA Transfer

## Implementation

- [x] Create directory, fork code from 006
- [x] Write `src/pretrained.py` (weight loading with shape matching)
- [x] Write `src/data.py` (10s windows, patch-aware padding masks)
- [x] Write `precompute_mels.py` (event-centered extraction)
- [x] Write `train.py` (pretrained + freeze + overfit CLI args)
- [x] Write `src/lightning_module.py` (freeze/unfreeze logic)
- [x] Create config files (16x16 + 128x1)
- [x] Write sbatch scripts

## Execution

- [ ] Pre-compute event-centered mel cache
- [ ] Fast-dev-run condition A (16x16) — shape/OOM check
- [ ] Fast-dev-run condition B (128x1) — shape/OOM check
- [ ] Overfit test condition A — verify loss→0, cmAP→1.0
- [ ] Overfit test condition B — verify loss→0, cmAP→1.0
- [ ] Full training condition A (16x16, freeze 5 epochs)
- [ ] Full training condition B (128x1, freeze 5 epochs)

## Verification Checklist

- [ ] Explore `detected_events` field format in BirdSet HSN
- [ ] Verify event-centered cache sample counts
- [ ] Log matched/skipped/re-initialized params for both conditions
- [ ] Fast-dev-run: loss finite, shapes correct, no OOM
- [ ] Overfit test: train loss→0, cmAP→1.0 within 50-100 epochs
- [ ] Padding mask: correct for 5s padded to 10s (both tokenizers)
- [ ] Compare early training curves against exp 006 baseline
- [ ] Final metric: val/cmAP > 0.15
