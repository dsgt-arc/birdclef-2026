# Kaggle baseline: exp 300 frozen-encoder ridge head (backbone ablation)

Minimal, backbone-agnostic competition submission for the exp 300 frozen-encoder
yardstick. `submission.py` encodes each 5 s window with one frozen encoder, applies that
encoder's fixed ridge head (`head.npz`), and writes the 234-species CSV. The script is an
**ablation harness**: the SAME pipeline runs Perch / BEATs / BirdNET — the encoder is
chosen by `head["encoder"]`, so one code file serves all three kernels.

## Pipeline
```
audio → 5 s windows → <encoder> → emb(D)
      → (emb-mean)/std @ W → clip[0,1]            # head: 234 species
      → [Perch only] uncovered species ← sigmoid(mapped Perch logits)   # inert: 234/234 covered
      → submission.csv (row_id + 234 species cols)
```
Swapping backbones = retrain `head.npz` (encoder field set) + ship that encoder's model
dataset; the CSV logic is unchanged. focal+soundscape covers all 234 species for every
encoder (labels are encoder-independent), so the Perch logit fallback is inert and absent
for BEATs/BirdNET — the ablation is a pure `emb → head → 234`.

## Per-encoder inputs (Kaggle, internet OFF)
| encoder | head dataset | model dataset | extra deps |
|---------|--------------|---------------|------------|
| perch   | `birdclef-2026-exp300-head` (or `-perch-head`) | `perch-v2-onnx` (`*.onnx`) | `birdclef-2025-deps` (onnxruntime, librosa) |
| beats   | `birdclef-2026-exp300-beats-head` | `birdclef-2026-beats` (`*.pt` + `beats/` vendor code) | torch is in the Kaggle base image |
| birdnet | `birdclef-2026-exp300-birdnet-head` | `birdclef-2026-birdnet` (bmz TFLite weights) | `birdclef-2026-birdnet-deps` (bmz, ai-edge-litert) |

The encoder forward in-kernel is the SAME code that produced the training embeddings
(Perch ONNX = exp 106 graph; BEATs = vendored `extract_features`; BirdNET = bmz pooling),
so each head transfers with no train/inference drift. **Verify** (cosine > 0.999 on a few
shared windows) before trusting any LB number.

## Build & push (per encoder; replace <enc>)
```bash
# 1. focal extraction (GPU array for beats / CPU array for birdnet), from the exp root
ENCODER=beats sbatch sbatch/extract-focal-torch.sbatch     # 8 shards
sbatch sbatch/extract-focal-birdnet.sbatch                 # 8 shards (cpu)

# 2. tune λ + fit the head on focal+soundscape (sbatch)
FEATURES_DIR=~/scratch/birdclef-2026/features/exp300-deploy/<enc> \
  sbatch sbatch/sweep-lambda.sbatch
ENCODER=<enc> FEATURES_DIR=~/scratch/birdclef-2026/features/exp300-deploy/<enc> \
  LAMBDA=<best> OUT=artifacts/head_<enc>.npz sbatch sbatch/fit-head.sbatch

# 3. stage + push the head dataset
cp ../artifacts/head_<enc>.npz head_dataset_<enc>/head.npz
kaggle datasets create  -p head_dataset_<enc>            # first time
kaggle datasets version -p head_dataset_<enc> -m "head vN"

# 4. one-time model datasets
#    beats:   cp <ckpt>.pt + vendor/unilm/beats -> beats_model_dataset/{*.pt,beats/}
#    birdnet: stage bmz BirdNET weights -> birdnet_model_dataset/
#    kaggle datasets create -p beats_model_dataset   (and birdnet_model_dataset)

# 5. push the kernel from a CLEAN temp dir (submission.py + the chosen metadata only —
#    never the multi-hundred-MB model), then submit
tmp=$(mktemp -d); cp submission.py "$tmp/"
cp kernel-metadata-<enc>.json "$tmp/kernel-metadata.json"
kaggle kernels push -p "$tmp"
kaggle competitions submit birdclef-2026 -k acmiyaguchi/birdclef-2026-exp300-<enc>-ridge-baseline
```

## Local dry-run (verification step 4)
```bash
TEST_SOUNDSCAPES_DIR=<dir with a few .ogg> \
SAMPLE_SUBMISSION=<sample_submission.csv> \
HEAD_DIR=../artifacts \
PERCH_ONNX=<...onnx>  BEATS_DIR=<...>  BIRDNET_DIR=<...> \
python submission.py     # encoder is read from head.npz; set the matching *_DIR
```
