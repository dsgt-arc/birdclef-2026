# Experiment 201: Soundscape Event Boxing

Desktop-first BirdCLEF 2026 analysis. Goal: turn the **window-level multi-label
presence** soundscape labels into precise **time-frequency bounding boxes**, focused on
**insects** (the 25 `47158sonNN` sonotypes), which are spectrally stationary.

**Direction (2026-06-01):** v1's fully-automatic band+shape **propagation** was
falsified (cross-site P0.06/R0.17; see `results.md`) and its `signature.py` /
`propagate.py` strand was **removed**. The go-forward plan is a **semi-supervised
annotation loop** — Perch-prototype retrieval routes windows, the DSP front-end
pre-draws boxes, the human adjudicates only the margin, confirmations feed back. See
`docs/concepts/semi-supervised-box-annotation-loop.md`. Load-bearing code: `boxes.py`
(box proposal + discriminative bands + attribution) and `label_studio.py` (curation).

**Gate result (2026-06-02):** the loop's *routing* premise **FAILED its feasibility gate
for insects** (`retrieve.py gate`): the weak-label Perch prototype barely clears the
within-site base-rate floor, fires on ~33% of insect-free windows, and only son03 (the
lone 3-site sonotype) routes cleanly; per-recording centering made it worse (stationary
insects sit in the recording mean). Stage B was **not** built. Two live forks:
(a) **birds-first** — Perch is in-distribution for Aves; (b) **curate insect windows
without routing** — weak labels + DSP pre-boxing + `label_studio`, no Perch. The gate
harness `retrieve.py` is reusable for the birds-first probe.

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
- Perch v2 soundscape embeddings (the annotation loop's router/seed):
  `/mnt/data/scratch/birdclef-2026/inference/perch-v2-tf/train_soundscapes/embeddings.parquet`
  — **one vector per 5s window**, frequency collapsed (127,896 rows = 10,658 files × 12
  chunks; `start_sec`/`end_sec` = chunk bounds). It routes/seeds; it cannot produce a box.
  Note: cross-site routing **failed the gate for insects** (2026-06-02); likely viable for
  birds (in-distribution). All 739 labeled windows join on `(filename, start_sec)`.

Polars note: read the CSVs with `pl.read_csv(path, infer_schema_length=10000)`.

## Data facts

- Severe site skew: **S22 = 40/66** labeled files; no S01/S02 labeled files at all.
  Cross-site transfer is the key validation risk — report in-site vs cross-site separately.
- Insect sonotype presence-window counts (739-window basis): son25=84 (most common — the
  dominant background cicada, in 84/168 insect windows), son07=48 … son05=3. Gate the
  discriminative band at `n_present >= 8` (drops son02=7, son09=6, son19=5, son12=5,
  son05=3 -> 20 usable). son15≡16 and son22≡23 are inseparable from weak labels
  (identical present-sets) — merge or defer.
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
