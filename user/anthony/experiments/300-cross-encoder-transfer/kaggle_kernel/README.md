# Kaggle baseline: exp 300 Perch ridge head

Minimal, backbone-agnostic competition submission for the exp 300 frozen-encoder
yardstick. `submission.py` encodes each 5 s window with Perch, applies the fixed ridge
head (`head.npz`), and fills the species the head can't cover from Perch's native logits.

## Pipeline
```
audio (32 kHz) → 12×5 s windows → Perch ONNX → emb(1536)+logits(14795)
              → (emb-mean)/std @ W → clip[0,1]            # head: 234 species
              → uncovered species ← sigmoid(mapped Perch logits)   # Perch-only fallback
              → submission.csv (row_id + 234 species cols)
```
Swapping backbones = swap `encode_windows` + retrain `head.npz`; the CSV logic is
unchanged. Non-Perch backbones drop the fallback (uncovered species → head output).

## Inputs (Kaggle, internet OFF)
| kind | slug | contents |
|------|------|----------|
| competition | `birdclef-2026` | `test_soundscapes/`, `sample_submission.csv` |
| dataset | `acmiyaguchi/birdclef-2026-exp300-head` | `head.npz`, `perch_species_mapping.json` |
| dataset | `acmiyaguchi/perch-v2-onnx` | `perch_v2_no_dft.onnx` (HF `justinchuby/Perch-onnx`) |
| dataset | `acmiyaguchi/birdclef-2025-deps` | py3.12 wheels: onnxruntime, librosa, soundfile |

The ONNX is the **same graph** that produced the training embeddings (exp 106
`perch_onnx.py`), so the head transfers with no SavedModel↔ONNX drift.

## Build & push
```bash
# 1. fit the head (sbatch, writes artifacts/head.npz) — from the experiment root
sbatch sbatch/fit-head.sbatch

# 2. stage the head dataset
cp ../artifacts/head.npz ../artifacts/perch_species_mapping.json head_dataset/
kaggle datasets create -p head_dataset       # first time
kaggle datasets version -p head_dataset -m "head vN"   # updates

# 3. one-time: perch ONNX dataset (download from HF, upload)
#    huggingface-cli download justinchuby/Perch-onnx perch_v2_no_dft.onnx
#    kaggle datasets create -p <dir with the .onnx + dataset-metadata.json>

# 4. push the kernel
kaggle kernels push -p .

# 5. submit / check
kaggle kernels status acmiyaguchi/birdclef-2026-exp300-perch-baseline
```

## Local dry-run (verification step 4)
```bash
TEST_SOUNDSCAPES_DIR=<dir with a few .ogg> \
SAMPLE_SUBMISSION=<sample_submission.csv> \
HEAD_DIR=../artifacts \
python submission.py          # needs perch-v2-onnx discoverable via find_perch_onnx
```
