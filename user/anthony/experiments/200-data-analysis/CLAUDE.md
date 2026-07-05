# Experiment 200: Data Analysis Scratch Space

This directory is for desktop-first, ad-hoc BirdCLEF 2026 data analysis. Keep scripts/notebooks self-contained and save lightweight outputs under `artifacts/`.

## Canonical data / metadata locations

Use the local scratch mirror as the source of truth:

- Dataset root: `/mnt/data/scratch/birdclef-2026/`
- Main training metadata: `/mnt/data/scratch/birdclef-2026/train.csv`
- Taxonomy / target labels: `/mnt/data/scratch/birdclef-2026/taxonomy.csv`
- Sample submission / row-id format: `/mnt/data/scratch/birdclef-2026/sample_submission.csv`
- Soundscape labels: `/mnt/data/scratch/birdclef-2026/train_soundscapes_labels.csv`
- Recording location info: `/mnt/data/scratch/birdclef-2026/recording_location.txt`
- Focal training audio: `/mnt/data/scratch/birdclef-2026/train_audio/`
- Training soundscapes: `/mnt/data/scratch/birdclef-2026/train_soundscapes/`
- Converted RavenPro MP3 soundscapes: `/mnt/data/scratch/birdclef-2026/raven_pro_mp3/`
- Original downloaded archive: `/mnt/data/scratch/birdclef-2026/birdclef-2026.zip`

Polars note: `train.csv` has mixed-looking label fields; use a large schema inference window, e.g. `pl.read_csv(path, infer_schema_length=10000)`.

## Perch inference outputs

Precomputed Perch v2 TensorFlow inference artifacts live at:

- Perch inference root: `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/`
- Focal/train-audio embeddings: `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_audio/embeddings.parquet`
- Focal/train-audio logits: `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_audio/logits.parquet`
- Soundscape embeddings: `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_soundscapes/embeddings.parquet`
- Soundscape logits: `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_soundscapes/logits.parquet`
- BirdCLEF-234 mapped soundscape probabilities:
  - `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_soundscapes/perch_234_probs.parquet`
  - `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_soundscapes/perch_234_raw_probs.parquet`
  - `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_soundscapes/perch_234_scores.parquet`
- Older partial run: `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf-old/`

Related Perch code/docs in the repo:

- Inference script: `user/murilo/inference/perch_inference.py`
- Batch wrappers: `user/murilo/inference/sbatch/run_perch.sh`, `user/murilo/inference/sbatch/perch_inference.sh`
- Shared Perch package helpers: `user/murilo/src/birdclef/perch/`
- Upstream source: https://github.com/google-research/perch
- Vendor notes: `docs/vendor/perch.md`

Experiment 106 also has useful Perch loading utilities and label mapping artifacts:

- `user/anthony/experiments/106-perch-bayesian/src/birdclef_106/data_loading.py`
- `user/anthony/experiments/106-perch-bayesian/src/birdclef_106/perch_onnx.py`
- `user/anthony/experiments/106-perch-bayesian/artifacts/perch_v2_labels.csv`
- `user/anthony/experiments/106-perch-bayesian/artifacts/perch_v2_ebird_classes.csv`
- `user/anthony/experiments/106-perch-bayesian/artifacts/perch_species_mapping.json`

## Working conventions

- Do not copy large scratch parquets or audio into this experiment directory.
- Put Python analysis code under `src/data_analysis/`; keep shell helpers in `scripts/`.
- Put figures, small tables, and summary markdowns in `artifacts/<topic>/`.
- Prefer `polars.scan_parquet()` / column projection for large Perch parquet files.
- This is desktop analysis; avoid SLURM assumptions unless a sub-analysis grows into a real numbered experiment.
