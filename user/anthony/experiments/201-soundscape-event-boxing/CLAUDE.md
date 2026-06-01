# Experiment 201: Soundscape Event Boxing

Desktop-first BirdCLEF 2026 analysis. Goal: turn the **window-level multi-label
presence** soundscape labels into precise **time-frequency bounding boxes**, then
**auto-propagate** boxes+labels to the unlabeled pool. v1 propagation targets
**insects** (the 25 `47158sonNN` sonotypes), which are spectrally stationary.

This experiment is **standalone and lean** — it does NOT depend on experiment 200.
The minimal DSP primitives (PCEN spectrogram, band discovery, ROI proposal,
`interval_iou`, the Label Studio serving patterns) were ported in, not imported.

## Canonical data / metadata locations

- Dataset root: `/mnt/data/scratch/birdclef-2026/`
- Taxonomy (has `class_name` for taxa grouping): `/mnt/data/scratch/birdclef-2026/taxonomy.csv`
- Soundscape labels (per-5s-window multi-label presence; HH:MM:SS times):
  `/mnt/data/scratch/birdclef-2026/train_soundscapes_labels.csv` — the CSV has 1,478 rows
  but **every row is duplicated exactly**, so there are **739 real windows** (66 files, 75 labels).
  `labels.load_windows()` dedups on load.
- Soundscape audio: `/mnt/data/scratch/birdclef-2026/train_soundscapes/` (10,658 recordings; 66 labeled)
- Perch v2 soundscape embeddings (all 10,658; v2 retrieval fallback):
  `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_soundscapes/embeddings.parquet`

Polars note: read the CSVs with `pl.read_csv(path, infer_schema_length=10000)`.

## Data facts

- Severe site skew: **S22 = 40/66** labeled files; no S01/S02 labeled files at all.
  Cross-site transfer is the key validation risk — report in-site vs cross-site separately.
- Insect sonotype presence-window counts (739-window basis): son25=84 (most common — the
  dominant background cicada, in 84/168 insect windows), son07=48 … son05=3. Gate signatures
  at `n_present >= 8` (drops son02=7, son09=6, son19=5, son12=5, son05=3 -> 20 usable).
- Labeled class breakdown: Aves 28, Insecta 25, Amphibia 17, Mammalia 4, Reptilia 1.

## Environment (venv on scratch — NEVER under home/project; 20GB home quota)

```bash
EXP=/mnt/data/research/arc/birdclef-2026/user/anthony/experiments/201-soundscape-event-boxing
uv venv /mnt/data/scratch/venvs/exp201-boxing/.venv --python 3.11
source /mnt/data/scratch/venvs/exp201-boxing/.venv/bin/activate
uv pip install -e "$EXP"      # NEVER bare pip, NEVER uv sync
```

Run modules with the package installed: `python -m birdclef_201.<module>`.

## Working conventions

- Do NOT copy scratch parquets/audio into the experiment dir.
- Outputs (figures, small tables, markdown) go under `artifacts/<topic>/` (gitignored).
- Derived clips + spectrogram PNGs for Label Studio go under
  `/mnt/data/scratch/birdclef-201-clips/` (a Label Studio serving root), regenerable.
- Desktop analysis; no SLURM.
