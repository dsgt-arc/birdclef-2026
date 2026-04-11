#!/usr/bin/env python3
"""Upload model weights to Kaggle datasets and push submission kernels.

Creates/updates Kaggle datasets with ONNX model weights and pushes the
unified submission kernel with model-specific metadata.

Model naming: {encoder}-{year}[-{variant}]
  - protoclr-2025, eat-2025, convnext-2025            (exps 006/007/009, 2025 data)
  - protoclr-2026, eat-2026, convnext-2026             (exp 011, 2026 80/10/10 split)
  - protoclr-2026-fulldata, eat-2026-fulldata, ...     (exp 011, 2026 100% focal)
  - protoclr-2026-soundscape, eat-2026-soundscape, ... (exp 012, focal + soundscapes, 234 classes)

Usage:
    # Upload 2025-trained models
    python scripts/upload_kaggle.py --model protoclr-2025
    python scripts/upload_kaggle.py --model all-2025

    # Upload 2026-trained models (80/10/10 split)
    python scripts/upload_kaggle.py --model protoclr-2026
    python scripts/upload_kaggle.py --model all-2026

    # Upload 2026-trained fulldata models (100% focal)
    python scripts/upload_kaggle.py --model protoclr-2026-fulldata
    python scripts/upload_kaggle.py --model all-2026-fulldata

    # Upload 2026-trained soundscape models (focal + soundscapes, 234 classes)
    python scripts/upload_kaggle.py --model protoclr-2026-soundscape
    python scripts/upload_kaggle.py --model all-2026-soundscape

    # Upload + push kernel
    python scripts/upload_kaggle.py --model protoclr-2026-soundscape --push-kernel

    # Dry run
    python scripts/upload_kaggle.py --model protoclr-2026-soundscape --dry-run
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_KAGGLE_USER = "murilogustineli"

# Separate deps dataset (onnxruntime wheels for offline Kaggle install)
DEPS_SLUG = "birdclef-deps"
DEPS_DIR = "artifacts/dist/dependencies"

MODELS = {
    # Models trained on BirdCLEF 2025 data (exps 006/007/009)
    "eat-2025": {
        "slug": "birdclef-2025-eat-weights",
        "weights_dir": "artifacts/weights/eat-2025",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-eat-2025.json",
    },
    "convnext-2025": {
        "slug": "birdclef-2025-convnext-weights",
        "weights_dir": "artifacts/weights/convnext-2025",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-convnext-2025.json",
    },
    "protoclr-2025": {
        "slug": "birdclef-2025-protoclr-weights",
        "weights_dir": "artifacts/weights/protoclr-2025",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-protoclr-2025.json",
    },
    # Models trained on BirdCLEF 2026 focal data, 80/10/10 split (exp 011)
    "eat-2026": {
        "slug": "birdclef-2026-eat-weights",
        "weights_dir": "artifacts/weights/eat-2026",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-eat-2026.json",
    },
    "convnext-2026": {
        "slug": "birdclef-2026-convnext-weights",
        "weights_dir": "artifacts/weights/convnext-2026",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-convnext-2026.json",
    },
    "protoclr-2026": {
        "slug": "birdclef-2026-protoclr-weights",
        "weights_dir": "artifacts/weights/protoclr-2026",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-protoclr-2026.json",
    },
    # Models trained on BirdCLEF 2026 focal data, 100% fulldata (exp 011 --train-all)
    "eat-2026-fulldata": {
        "slug": "birdclef-2026-eat-fulldata-weights",
        "weights_dir": "artifacts/weights/eat-2026-fulldata",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-eat-2026-fulldata.json",
    },
    "convnext-2026-fulldata": {
        "slug": "birdclef-2026-convnext-fulldata-weights",
        "weights_dir": "artifacts/weights/convnext-2026-fulldata",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-convnext-2026-fulldata.json",
    },
    "protoclr-2026-fulldata": {
        "slug": "birdclef-2026-protoclr-fulldata-weights",
        "weights_dir": "artifacts/weights/protoclr-2026-fulldata",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-protoclr-2026-fulldata.json",
    },
    # Models fine-tuned on focal + labeled soundscapes, 234 classes (exp 012)
    "eat-2026-soundscape": {
        "slug": "birdclef-2026-eat-soundscape-weights",
        "weights_dir": "artifacts/weights/eat-2026-soundscape",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-eat-2026-soundscape.json",
    },
    "convnext-2026-soundscape": {
        "slug": "birdclef-2026-convnext-soundscape-weights",
        "weights_dir": "artifacts/weights/convnext-2026-soundscape",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-convnext-2026-soundscape.json",
    },
    "protoclr-2026-soundscape": {
        "slug": "birdclef-2026-protoclr-soundscape-weights",
        "weights_dir": "artifacts/weights/protoclr-2026-soundscape",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-protoclr-2026-soundscape.json",
    },
    # Exp 015: Student models (pseudo-labeled training, 234 classes)
    "protoclr-2026-student-v1": {
        "slug": "birdclef-2026-protoclr-student-v1-weights",
        "weights_dir": "artifacts/weights/protoclr-2026-student-v1",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-protoclr-2026-student-v1.json",
    },
    "eat-2026-student-v1": {
        "slug": "birdclef-2026-eat-student-v1-weights",
        "weights_dir": "artifacts/weights/eat-2026-student-v1",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-eat-2026-student-v1.json",
    },
    # Exp 016: SED models (234 classes)
    "effnet-b0-sed-v1": {
        "slug": "birdclef-2026-effnet-b0-sed-v1-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v1",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v1.json",
    },
    "effnet-b0-sed-v1-10s": {
        "slug": "birdclef-2026-effnet-b0-sed-v1-10s-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v1-10s",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v1-10s.json",
    },
    "effnet-b0-sed-v1-10s-pp2": {
        "slug": "birdclef-2026-effnet-b0-sed-v1-10s-pp2-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v1-10s-pp2",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v1-10s-pp2.json",
    },
    "effnet-b0-sed-v2": {
        "slug": "birdclef-2026-effnet-b0-sed-v2-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v2",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v2.json",
    },
    "effnet-b0-sed-v3a": {
        "slug": "birdclef-2026-effnet-b0-sed-v3a-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v3a",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v3a.json",
    },
    "effnet-b0-sed-v3b": {
        "slug": "birdclef-2026-effnet-b0-sed-v3b-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v3b",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v3b.json",
    },
    "effnet-b0-sed-v3c": {
        "slug": "birdclef-2026-effnet-b0-sed-v3c-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v3c",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v3c.json",
    },
    "effnet-b0-sed-v3d": {
        "slug": "birdclef-2026-effnet-b0-sed-v3d-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v3d",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v3d.json",
    },
    # Exp 017: Mel overhaul (n_fft=4096, n_mels=224)
    "effnet-b0-sed-v4a": {
        "slug": "birdclef-2026-effnet-b0-sed-v4a-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v4a",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v4a.json",
    },
    "effnet-b0-sed-v4a-pp2": {
        "slug": "birdclef-2026-effnet-b0-sed-v4a-pp2-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v4a-pp2",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v4a-pp2.json",
    },
    # Exp 017 Step 2: Parallel ablations from v4a base
    "effnet-b0-sed-v4b": {
        "slug": "birdclef-2026-effnet-b0-sed-v4b-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v4b",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v4b.json",
    },
    "effnet-b0-sed-v4c": {
        "slug": "birdclef-2026-effnet-b0-sed-v4c-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v4c",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v4c.json",
    },
    "effnet-b0-sed-v4d": {
        "slug": "birdclef-2026-effnet-b0-sed-v4d-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v4d",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v4d.json",
    },
    "effnet-b0-sed-v4e": {
        "slug": "birdclef-2026-effnet-b0-sed-v4e-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v4e",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v4e.json",
    },
    "effnet-b0-sed-v4b-fulldata": {
        "slug": "birdclef-2026-effnet-b0-sed-v4b-fulldata-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v4b-fulldata",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v4b-fulldata.json",
    },
    "effnet-b0-sed-v4b-pp2": {
        "slug": "birdclef-2026-effnet-b0-sed-v4b-pp2-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v4b-pp2",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v4b-pp2.json",
    },
    # Exp 017 Step 3: Loss/head re-ablations with correct pairing
    "effnet-b0-sed-v4g": {
        "slug": "birdclef-2026-effnet-b0-sed-v4g-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v4g",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v4g.json",
    },
    "effnet-b0-sed-v4h": {
        "slug": "birdclef-2026-effnet-b0-sed-v4h-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v4h",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v4h.json",
    },
    # Exp 017 Step 3: Combined winners (v4b + v4e)
    "effnet-b0-sed-v4f": {
        "slug": "birdclef-2026-effnet-b0-sed-v4f-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v4f",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v4f.json",
    },
    # Exp 017 Step 5: Waveform MixUp
    "effnet-b0-sed-v4j": {
        "slug": "birdclef-2026-effnet-b0-sed-v4j-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v4j",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v4j.json",
    },
    # Exp 019 Step 1: SED ablations
    "effnet-b0-sed-v5a": {
        "slug": "birdclef-2026-effnet-b0-sed-v5a-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v5a",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v5a.json",
    },
    "effnet-b0-sed-v5b": {
        "slug": "birdclef-2026-effnet-b0-sed-v5b-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v5b",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v5b.json",
    },
    "effnet-b0-sed-v5c": {
        "slug": "birdclef-2026-effnet-b0-sed-v5c-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v5c",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v5c.json",
    },
    "effnet-b0-sed-v5d": {
        "slug": "birdclef-2026-effnet-b0-sed-v5d-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v5d",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v5d.json",
    },
    "hgnet-b0-sed-v5e": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5e-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5e",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5e.json",
    },
    # Exp 019 Step 2: ttahara full recipe reproduction
    "effnet-b0-sed-v5f": {
        "slug": "birdclef-2026-effnet-b0-sed-v5f-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v5f",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v5f.json",
    },
    "hgnet-b0-sed-v5g": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5g-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5g",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5g.json",
    },
    # Exp 019 Step 3: 4-fold ensemble
    "hgnet-b0-sed-v5g-4fold": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5g-4fold-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5g-4fold",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5g-4fold.json",
    },
    # Exp 019 Step 4: v5j 4-fold ensemble (best single-fold 0.906 LB)
    "hgnet-b0-sed-v5j-4fold": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5j-4fold-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5j-4fold",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5j-4fold.json",
    },
    # Exp 019 Step 4: v5g recipe ablations
    "hgnet-b0-sed-v5h": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5h-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5h",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5h.json",
    },
    "hgnet-b0-sed-v5i": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5i-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5i",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5i.json",
    },
    "hgnet-b0-sed-v5k": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5k-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5k",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5k.json",
    },
    "hgnet-b0-sed-v5l": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5l-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5l",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5l.json",
    },
    "hgnet-b0-sed-v5j": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5j-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5j",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5j.json",
    },
    # Exp 021 Phase F: v5j blend ablations (same checkpoint, different blend)
    "hgnet-b0-sed-v5j-fmax": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5j-fmax-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5j-fmax",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5j-fmax.json",
    },
    "hgnet-b0-sed-v5j-clip": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5j-clip-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5j-clip",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5j-clip.json",
    },
    # Exp 019 embed_dim hypothesis test
    "effnet-b0-sed-v5m": {
        "slug": "birdclef-2026-effnet-b0-sed-v5m-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v5m",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v5m.json",
    },
    "effnet-b0-sed-v5n": {
        "slug": "birdclef-2026-effnet-b0-sed-v5n-weights",
        "weights_dir": "artifacts/weights/effnet-b0-sed-v5n",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-sed-v5n.json",
    },
    # Exp 021 Phase A: v5j + site/hour priors (pp2)
    "hgnet-b0-sed-v5j-pp2": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5j-pp2-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5j-pp2",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5j-pp2.json",
    },
    "hgnet-b0-sed-v5j-4fold-pp2": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5j-4fold-pp2-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5j-4fold-pp2",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5j-4fold-pp2.json",
    },
    # Exp 021 Phase B: recipe ablations
    "hgnet-b0-sed-v6b": {
        "slug": "birdclef-2026-hgnet-b0-sed-v6b-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v6b",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v6b.json",
    },
    # Exp 021 Phase E: LSEHead multi-context
    "hgnet-b0-mc-lse-sed-v8a": {
        "slug": "birdclef-2026-hgnet-b0-mc-lse-sed-v8a-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-mc-lse-sed-v8a",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-mc-lse-sed-v8a.json",
    },
    "hgnet-b0-mc-lse-sed-v8b": {
        "slug": "birdclef-2026-hgnet-b0-mc-lse-sed-v8b-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-mc-lse-sed-v8b",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-mc-lse-sed-v8b.json",
    },
    # Exp 021 Phase F: competition-standard AttHead (v6k)
    "hgnet-b0-atthead-v6k": {
        "slug": "birdclef-2026-hgnet-b0-atthead-v6k-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-atthead-v6k",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-atthead-v6k.json",
    },
    # Exp 021 Phase C: AttHead + backbone scaling
    "hgnet-b2-sed-v6h": {
        "slug": "birdclef-2026-hgnet-b2-sed-v6h-weights",
        "weights_dir": "artifacts/weights/hgnet-b2-sed-v6h",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b2-sed-v6h.json",
    },
    "effnet-b0-ns-sed-v6j": {
        "slug": "birdclef-2026-effnet-b0-ns-sed-v6j-weights",
        "weights_dir": "artifacts/weights/effnet-b0-ns-sed-v6j",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-effnet-b0-ns-sed-v6j.json",
    },
    # Exp 021 Phase G: manifold mixup
    "hgnet-b0-sed-v6l": {
        "slug": "birdclef-2026-hgnet-b0-sed-v6l-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v6l",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v6l.json",
    },
    # Exp 022: PCEN normalization
    "hgnet-b0-sed-v10a": {
        "slug": "birdclef-2026-hgnet-b0-sed-v10a-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v10a",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v10a.json",
    },
    "hgnet-b0-sed-v10b": {
        "slug": "birdclef-2026-hgnet-b0-sed-v10b-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v10b",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v10b.json",
    },
    "hgnet-b0-sed-v10c": {
        "slug": "birdclef-2026-hgnet-b0-sed-v10c-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v10c",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v10c.json",
    },
    "hgnet-b0-sed-v5j-repro": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5j-repro-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5j-repro",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5j-repro.json",
    },
    # Exp 025: v5j + pseudo-label training (round 1)
    "hgnet-b0-sed-v5j-pl-v1": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5j-pl-v1-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5j-pl-v1",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5j-pl-v1.json",
    },
    # Exp 026: v5j + dual validation (soundscape checkpoint selection)
    "hgnet-b0-sed-v5j-dualval": {
        "slug": "birdclef-2026-hgnet-b0-sed-v5j-dualval-weights",
        "weights_dir": "artifacts/weights/hgnet-b0-sed-v5j-dualval",
        "kernel_metadata": "kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5j-dualval.json",
    },
}


def get_experiment_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def create_metadata(slug: str, kaggle_user: str, path: Path) -> None:
    """Create dataset-metadata.json for Kaggle."""
    meta = {
        "title": slug,
        "id": f"{kaggle_user}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = path / "dataset-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Created: {meta_path}")


def run_cmd(cmd: list[str], dry_run: bool = False) -> bool:
    """Run a command. Returns True if successful."""
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        output = result.stdout + result.stderr
        if "already in use" in output or "already exists" in output:
            return False
        return True
    except subprocess.CalledProcessError as e:
        combined = e.stdout + e.stderr
        if "already exists" in combined or "already in use" in combined or "403" in combined:
            return False
        if e.stdout:
            print(f"  stdout: {e.stdout.strip()}")
        if e.stderr:
            print(f"  stderr: {e.stderr.strip()}")
        raise


def upload_dataset(slug: str, path: Path, kaggle_user: str, dry_run: bool = False) -> None:
    """Upload or update a single Kaggle dataset."""
    if not path.exists():
        print(f"  WARNING: {path} does not exist, skipping")
        return

    files = [f for f in path.glob("*") if f.name != "dataset-metadata.json"]
    if not files:
        print(f"  WARNING: {path} is empty, skipping")
        return

    print(f"\n{'=' * 50}")
    print(f"Uploading: {kaggle_user}/{slug}")
    print(f"  Source: {path}")
    print("  Files:")
    for f in sorted(files):
        size_mb = f.stat().st_size / 1e6
        print(f"    {f.name} ({size_mb:.1f} MB)")

    create_metadata(slug, kaggle_user, path)

    # Try create, then version
    try:
        if run_cmd(["kaggle", "datasets", "create", "-p", str(path), "-r", "zip"], dry_run):
            print(f"  Created new dataset: {kaggle_user}/{slug}")
            return
    except subprocess.CalledProcessError:
        pass

    print("  Dataset exists, creating new version...")
    run_cmd(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(path),
            "-m",
            "Auto-update from HPC export",
            "-r",
            "zip",
        ],
        dry_run,
    )
    print(f"  Updated: {kaggle_user}/{slug}")


def upload_weights(model_name: str, kaggle_user: str, dry_run: bool = False) -> None:
    """Upload model weights to Kaggle dataset."""
    model_cfg = MODELS[model_name]
    experiment_dir = get_experiment_dir()
    weights_dir = experiment_dir / model_cfg["weights_dir"]

    if not weights_dir.exists():
        print(f"  WARNING: {weights_dir} does not exist. Run export first.")
        return

    # Copy species_mapping.json into model weights dir
    # Soundscape models use 234-species mapping, 2026 models use 206, 2025 uses 41-overlap
    if (
        "-soundscape" in model_name
        or "-student" in model_name
        or "-sed" in model_name
        or "-atthead" in model_name
    ):
        mapping_src = experiment_dir / "artifacts" / "weights" / "species_mapping_2026_234.json"
    elif "-2026" in model_name:
        mapping_src = experiment_dir / "artifacts" / "weights" / "species_mapping_2026.json"
    else:
        mapping_src = experiment_dir / "artifacts" / "weights" / "species_mapping.json"
    mapping_dst = weights_dir / "species_mapping.json"
    if mapping_src.exists():
        shutil.copy2(mapping_src, mapping_dst)
        print(f"  Copied {mapping_src.name} to {weights_dir}")
    elif not mapping_dst.exists():
        print(f"  WARNING: {mapping_src} not found, species_mapping.json missing")

    upload_dataset(model_cfg["slug"], weights_dir, kaggle_user, dry_run)


def push_kernel(model_name: str, dry_run: bool = False) -> None:
    """Push submission kernel to Kaggle."""
    model_cfg = MODELS[model_name]
    experiment_dir = get_experiment_dir()
    kernel_dir = experiment_dir / "kaggle_kernel"
    metadata_src = experiment_dir / model_cfg["kernel_metadata"]

    if not metadata_src.exists():
        print(f"  WARNING: {metadata_src} not found")
        return

    # Copy model-specific metadata as kernel-metadata.json
    metadata_dst = kernel_dir / "kernel-metadata.json"
    shutil.copy2(metadata_src, metadata_dst)
    print(f"  Using metadata: {metadata_src.name}")

    run_cmd(["kaggle", "kernels", "push", "-p", str(kernel_dir)], dry_run)
    print(f"  Kernel pushed for {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Upload Kaggle datasets and kernels")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys())
        + [
            "all",
            "all-2025",
            "all-2026",
            "all-2026-fulldata",
            "all-2026-soundscape",
            "all-2026-student",
            "all-2026-sed",
        ],
        help="Model to upload (e.g., protoclr-2026-soundscape, all-2026-soundscape)",
    )
    parser.add_argument("--kaggle-user", type=str, default=DEFAULT_KAGGLE_USER)
    parser.add_argument("--push-kernel", action="store_true", help="Also push the Kaggle kernel")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Check kaggle CLI
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: kaggle CLI not found or not configured")
        sys.exit(1)

    experiment_dir = get_experiment_dir()
    if args.model == "all":
        models = list(MODELS.keys())
    elif args.model == "all-2025":
        models = [k for k in MODELS if k.endswith("-2025")]
    elif args.model == "all-2026-fulldata":
        models = [k for k in MODELS if k.endswith("-2026-fulldata")]
    elif args.model == "all-2026-soundscape":
        models = [k for k in MODELS if k.endswith("-2026-soundscape")]
    elif args.model == "all-2026":
        # Only 80/10/10 models, not fulldata or soundscape
        models = [
            k
            for k in MODELS
            if k.endswith("-2026") and "-fulldata" not in k and "-soundscape" not in k
        ]
    else:
        models = [args.model]

    # Upload deps dataset (shared across all models)
    deps_path = experiment_dir / DEPS_DIR
    upload_dataset(DEPS_SLUG, deps_path, args.kaggle_user, args.dry_run)

    # Upload model weights + push kernels
    for model_name in models:
        upload_weights(model_name, args.kaggle_user, args.dry_run)
        if args.push_kernel:
            if not args.dry_run:
                import time

                print("\n  Waiting 30s for dataset to process before pushing kernel...")
                time.sleep(30)
            push_kernel(model_name, args.dry_run)

    print(f"\nDone! Uploaded deps + {len(models)} model(s).")


if __name__ == "__main__":
    main()
