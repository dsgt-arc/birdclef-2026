# Available Teacher Predictions

## Location

`/storage/project/ps-clef2026_birdclef-0/shared/inference/birdclef-2026/`

| Model | File | Shape | Kaggle LB |
|-------|------|-------|-----------|
| ProtoCLR/CvT-13 | `protoclr-2026-soundscape/labeled.npz` | (792, 234) | **0.736** |
| ConvNeXt-Base | `convnext-2026-soundscape/labeled.npz` | (792, 234) | 0.720 |
| EAT-base | `eat-2026-soundscape/labeled.npz` | (792, 234) | 0.701 |

## NPZ format

- `predictions`: (792, 234) float32 — sigmoid probabilities [0, 1]
- `seg_keys`: (792, 2) object — window identifiers [filename, window_index]
- `species`: (234,) — species labels matching submission columns

792 rows = 66 files × 12 windows/file. Highly sparse (median ~0.0001).

**`seg_keys` caveat**: pickled with newer numpy; may need version-matched numpy or
reconstruction from the known 66-file × 12-window structure.
