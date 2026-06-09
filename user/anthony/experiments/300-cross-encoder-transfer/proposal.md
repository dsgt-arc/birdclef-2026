# Experiment 300: Cross-Encoder Transfer (Frozen-Probe Comparison)

**Status**: scaffolded (not yet run)
**Depends on**: shared `birdclef.encoders`, exp 106 Perch cache, exp 010 BirdNET pattern

## Hypothesis

Foundational audio encoders can be placed on the *same* BirdCLEF 2026 yardstick as
Perch and the leaderboard systems by probing their **frozen** embeddings with a
ridge linear head on the 749-window 9-fold leave-one-site-out (LOSO) split. Today
only Perch has been measured on this target; BEATs/AST/EAT/SSAMBA exist only on
ESC-50/HSN proxies. This experiment produces the missing comparison that the
working note's §6 turns on.

## Design

**Roster (6 encoders).** General/token encoders BEATs, AST, EAT, SSAMBA-HF; plus
bioacoustic-specialist anchors Perch (1536-d) and BirdNET (1024-d). Perch also
carries both a LOSO AUROC *and* a Kaggle LB number, so it anchors the AUROC↔LB
metric bridge that makes "how do they compare" answerable.

**One yardstick.** Every encoder emits a fixed embedding per labeled 5 s window
into a single encoder-agnostic parquet cache (see `CLAUDE.md` schema), then a
ridge LOSO probe (`probe/ridge_probe.py`, ported from exp 106, Hopfield head
dropped per design decision) reports macro AUROC / AP and per-species AUROC. Ridge
is the agreed probe — simple, reproducible, scale-robust via per-fold standardization.

**Backends.** Four encoders share the `birdclef.encoders.AudioEncoder`
`extract_features((B,1,N)@16kHz)` interface (torch). Perch is *ingested* from exp
106's cache (no recompute). BirdNET runs through bioacoustics-model-zoo in a
separate TF venv. All three paths iterate the same labels CSV, so windows align.

## Evaluation

- Primary: per-encoder macro LOSO AUROC, ranked (`artifacts/probe_metrics_frozen.json`).
- Metric bridge: calibrate AUROC↔Kaggle LB through Perch (it has both).
- Reference anchors: Perch ridge ≈ 0.749 AUROC (exp 106) reproduces here as a sanity check.

## Notes / caveats

- **739 vs 749 windows.** The labeled cache currently holds 739 deduped windows;
  "749" in the note is the pre-dedup count. The LOSO fold count is data-driven
  (`len(unique_sites)`), not hardcoded — reconcile the headline number when writing.
- Frozen probe only. Encoder *adaptation* (LoRA + DAPT) is exp 301, which reuses
  this experiment's extractor and probe on adapted `variant=` caches.
- Subset breakdowns (birds/amphibians/insects, with/without focal) are not computed
  here to keep exp 300 free of external taxonomy/focal artifacts; per-species AUROC
  is saved so they can be grouped downstream using `taxonomy.csv` if wanted.
