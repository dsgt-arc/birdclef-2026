#!/usr/bin/env python3
"""BirdCLEF 2026 Perch + ProtoSSM v4 Submission (full ProtoSSM v5 pipeline).

v4 upgrades over v3 (Perch + Probes):
  1. ProtoSSM v2: Mamba-style SSM with cross-attention, metadata, prototypes
  2. 3-fold GroupKFold OOF for ProtoSSM ensemble weight optimization
  3. ResidualSSM second-pass correction
  4. TTA (temporal shift [0, 1, -1])
  5. 6-stage post-processing pipeline (temp scaling, file confidence, rank-aware,
     delta shift, per-class thresholds)
  6. Ensemble fusion: ProtoSSM + MLP probes (optimized weight)

Pipeline:
  Phase 1: Load labeled data, OOF priors, train MLP probes (~5s)
  Phase 2: ProtoSSM OOF + final training + ResidualSSM (~30s)
  Phase 3: Perch inference on test soundscapes (~80min)
  Phase 4: Score fusion + post-processing (~30s)

CRITICAL: Self-contained kernel — NO imports from birdclef.perch.*.
"""

# ---------------------------------------------------------------------------
# Offline dependency installation
# ---------------------------------------------------------------------------
import os
import subprocess
import sys
import time

START_TIME = time.time()
TIMEOUT_SECONDS = 88 * 60  # 88 min safety cutoff

KAGGLE_USER = "murilogustineli"


def _find_input_early(slug: str) -> str:
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


if os.path.exists("/kaggle/input"):
    print(f"Kaggle input contents: {os.listdir('/kaggle/input')}")

# Install TF 2.20
TF_WHEEL_CANDIDATES = [
    _find_input_early("birdclef-tf220-wheel"),
    "/kaggle/input/birdclef-tf220-wheel",
]
for tf_wheel_dir in TF_WHEEL_CANDIDATES:
    if os.path.exists(tf_wheel_dir):
        print(f"Installing TF 2.20 from {tf_wheel_dir}")
        for whl in sorted(os.listdir(tf_wheel_dir)):
            if whl.endswith(".whl"):
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        os.path.join(tf_wheel_dir, whl),
                    ]
                )
        break

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERCH_SR = 32000
WINDOW_SECONDS = 5
WINDOW_SAMPLES = PERCH_SR * WINDOW_SECONDS
SOUNDSCAPE_SECONDS = 60
N_WINDOWS = SOUNDSCAPE_SECONDS // WINDOW_SECONDS
N_SPECIES = 234

# Slugs
LABELED_EMB_SLUG = "birdclef-2026-perch-labeled-embeddings"
PERCH_ARTIFACTS_SLUG = "birdclef-2026-perch-ensemble-v1"

# Perch model candidates
PERCH_MODEL_CANDIDATES = [
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
    "/kaggle/input/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2/1",
    "/kaggle/input/bird-vocalization-classifier/tensorflow2/perch_v2/1",
    "/kaggle/input/bird-vocalization-classifier",
]

# OOF cross-validation (matching chiranjithdharma)
OOF_N_SPLITS = 5

# Probe hyperparameters (v3: Yuriy/jaejohn optimal)
PCA_DIM = 64
PROBE_MIN_POS = 8
PROBE_ALPHA = 0.40  # fixed (Yuriy/jaejohn), replaces dynamic formula

# MLP probe params (matching Yuriy)
MLP_HIDDEN = (128,)
MLP_ACTIVATION = "relu"
MLP_MAX_ITER = 300
MLP_EARLY_STOPPING = True
MLP_VALIDATION_FRACTION = 0.15
MLP_LR_INIT = 0.001
MLP_ALPHA = 0.01  # L2 regularization

# Prior lambdas
LAMBDA_EVENT = 0.4
LAMBDA_TEXTURE = 1.0
LAMBDA_PROXY_TEXTURE = 0.8

# Smoothing
SMOOTH_TEXTURE = 0.35
SMOOTH_EVENT = 0.15

# Bayesian smoothing k values (matching chiranjithdharma)
K_SITE = 8
K_HOUR = 8
K_INTERACTION = 4

TEXTURE_TAXA = {"Amphibia", "Insecta"}
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}

# ---------------------------------------------------------------------------
# ProtoSSM config (inline from config.py)
# ---------------------------------------------------------------------------
PROTO_SSM_D_MODEL = 128
PROTO_SSM_D_STATE = 16
PROTO_SSM_N_LAYERS = 2
PROTO_SSM_DROPOUT = 0.15
PROTO_SSM_N_SITES = 20
PROTO_SSM_META_DIM = 16
PROTO_SSM_USE_CROSS_ATTN = True
PROTO_SSM_CROSS_ATTN_HEADS = 4

PROTO_SSM_N_EPOCHS = 60
PROTO_SSM_LR = 1e-3
PROTO_SSM_WEIGHT_DECAY = 2e-3
PROTO_SSM_PATIENCE = 15
PROTO_SSM_POS_WEIGHT_CAP = 30.0
PROTO_SSM_DISTILL_WEIGHT = 0.1
PROTO_SSM_LABEL_SMOOTHING = 0.02
PROTO_SSM_OOF_N_SPLITS = 3
PROTO_SSM_MIXUP_ALPHA = 0.3
PROTO_SSM_FOCAL_GAMMA = 2.0
PROTO_SSM_SWA_START_FRAC = 0.7

RESIDUAL_SSM_D_MODEL = 64
RESIDUAL_SSM_D_STATE = 8
RESIDUAL_SSM_DROPOUT = 0.1
RESIDUAL_SSM_CORRECTION_WEIGHT = 0.3
RESIDUAL_SSM_N_EPOCHS = 30
RESIDUAL_SSM_LR = 1e-3
RESIDUAL_SSM_PATIENCE = 8

TEMPERATURE_AVES = 1.10
TEMPERATURE_TEXTURE = 0.95
FILE_LEVEL_TOP_K = 2
TTA_SHIFTS = [0, 1, -1]
RANK_AWARE_POWER = 0.5
DELTA_SHIFT_ALPHA = 0.15


# ===========================================================================
# Path resolution
# ===========================================================================


def _find_input(slug):
    for c in [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
        f"/kaggle/input/notebooks/{KAGGLE_USER}/{slug}",
    ]:
        if os.path.exists(c):
            return c
    return None


def find_labeled_emb_dir():
    path = _find_input(LABELED_EMB_SLUG)
    if path and os.path.exists(os.path.join(path, "labeled_perch_arrays.npz")):
        return Path(path)
    local = Path(
        "/storage/project/ps-clef2026_birdclef-0/shared/datasets/birdclef-2026/perch-labeled-embeddings"
    )
    if local.exists() and (local / "labeled_perch_arrays.npz").exists():
        return local
    return None


def find_artifacts_dir():
    path = _find_input(PERCH_ARTIFACTS_SLUG)
    if path:
        return Path(path)
    local = Path("artifacts")
    if (local / "perch_species_mapping.json").exists():
        return local
    shared = Path(
        "/storage/project/ps-clef2026_birdclef-0/shared/artifacts/birdclef-2026/perch-ensemble-v1"
    )
    if shared.exists():
        return shared
    return None


def find_perch_model_dir():
    for candidate in PERCH_MODEL_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            if (p / "saved_model.pb").exists():
                return p
            for subdir in sorted(p.iterdir()):
                if subdir.is_dir() and (subdir / "saved_model.pb").exists():
                    return subdir
            return p
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, _dirs, files in os.walk(str(kaggle_input)):
            if "saved_model.pb" in files and "perch" in root.lower():
                return Path(root)
    for local in [
        Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2-cpu"),
        Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2"),
    ]:
        if local.exists():
            return local
    return None


def find_competition_dir():
    for c in ["/kaggle/input/competitions/birdclef-2026", "/kaggle/input/birdclef-2026"]:
        if os.path.exists(c):
            return Path(c)
    return Path(os.environ.get("COMPETITION_DIR", "data/birdclef-2026"))


# ===========================================================================
# Perch inference (returns RAW LOGITS, not sigmoid)
# ===========================================================================


def load_perch_model(model_dir):
    import tensorflow as tf

    model = tf.saved_model.load(str(model_dir))
    sigs = list(model.signatures.keys()) if hasattr(model, "signatures") else []
    if hasattr(model, "infer_tf"):
        return model, model.infer_tf
    if "serving_default" in sigs:
        return model, model.signatures["serving_default"]
    if callable(model):
        return model, model
    if sigs:
        return model, model.signatures[sigs[0]]
    raise RuntimeError("No Perch inference method found")


def run_perch_inference(infer_fn, waveform_batch):
    """Returns (embeddings, logits) -- logits are RAW, no sigmoid."""
    import tensorflow as tf

    input_tensor = tf.constant(waveform_batch, dtype=tf.float32)
    try:
        output = infer_fn(inputs=input_tensor)
    except TypeError:
        output = infer_fn(input_tensor)

    if isinstance(output, dict):
        embeddings = logits = None
        for key in ["embedding", "embeddings", "output_0"]:
            if key in output:
                embeddings = output[key].numpy()
                break
        for key in ["label", "logits", "logit", "output_1"]:
            if key in output:
                logits = output[key].numpy()
                break
        if embeddings is None or logits is None:
            for _key, val in output.items():
                arr = val.numpy()
                if arr.shape[-1] == 1536 and embeddings is None:
                    embeddings = arr
                elif arr.shape[-1] > 10000 and logits is None:
                    logits = arr
        if embeddings is None or logits is None:
            raise RuntimeError(f"Cannot identify outputs: {list(output.keys())}")
    elif isinstance(output, tuple | list):
        arr0 = output[0].numpy() if hasattr(output[0], "numpy") else np.array(output[0])
        arr1 = output[1].numpy() if hasattr(output[1], "numpy") else np.array(output[1])
        if arr0.shape[-1] == 1536:
            embeddings, logits = arr0, arr1
        else:
            logits, embeddings = arr0, arr1
    else:
        raise RuntimeError(f"Unexpected output type: {type(output)}")

    return embeddings.astype(np.float32), logits.astype(np.float32)


def compute_perch_mapped_logits(logits, species_mapping):
    """Map Perch logits to 234 BirdCLEF species as RAW LOGITS (no sigmoid)."""
    birdclef_species = species_mapping["birdclef_species"]
    direct_map = species_mapping["direct_map"]
    genus_proxy_map = species_mapping["genus_proxy_map"]
    n_species = len(birdclef_species)
    n_samples = logits.shape[0]
    n_logits = logits.shape[1]

    scores = np.full((n_samples, n_species), -10.0, dtype=np.float32)
    for i, label in enumerate(birdclef_species):
        if label in direct_map:
            model_idx = direct_map[label] - 1
            if 0 <= model_idx < n_logits:
                scores[:, i] = logits[:, model_idx].astype(np.float32)
        elif label in genus_proxy_map:
            valid = [idx - 1 for idx in genus_proxy_map[label] if 0 <= idx - 1 < n_logits]
            if valid:
                scores[:, i] = logits[:, valid].max(axis=1).astype(np.float32)

    return scores


def load_and_window_audio(audio_path, sr=PERCH_SR):
    audio, file_sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    expected = SOUNDSCAPE_SECONDS * sr
    if len(audio) < expected:
        audio = np.pad(audio, (0, expected - len(audio)))
    elif len(audio) > expected:
        audio = audio[:expected]
    return np.stack(
        [audio[w * WINDOW_SAMPLES : (w + 1) * WINDOW_SAMPLES] for w in range(N_WINDOWS)],
        axis=0,
    ).astype(np.float32)


def _parse_soundscape_metadata(filename):
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 6:
        return None, None
    try:
        return parts[3], int(parts[5][:2])
    except (ValueError, IndexError):
        return parts[3], None


# ===========================================================================
# Prior tables (logit-space, matching chiranjithdharma)
# ===========================================================================


def fit_prior_tables(labels_matrix, meta_rows, birdclef_species):
    """Fit Bayesian site/hour prior tables. Returns probability tables."""
    sites = np.array([r["site"] for r in meta_rows])
    hours = np.array([r["hour"] for r in meta_rows])

    global_p = labels_matrix.mean(axis=0).astype(np.float32)

    unique_sites = sorted({s for s in sites if s is not None})
    unique_hours = sorted({int(h) for h in hours if h is not None and int(h) >= 0})

    # Site priors
    site_to_i = {s: i for i, s in enumerate(unique_sites)}
    site_n = np.zeros(len(unique_sites), dtype=np.float32)
    site_p = np.zeros((len(unique_sites), len(birdclef_species)), dtype=np.float32)
    for s in unique_sites:
        i = site_to_i[s]
        mask = sites == s
        site_n[i] = mask.sum()
        site_p[i] = labels_matrix[mask].mean(axis=0)

    # Hour priors
    hour_to_i = {h: i for i, h in enumerate(unique_hours)}
    hour_n = np.zeros(len(unique_hours), dtype=np.float32)
    hour_p = np.zeros((len(unique_hours), len(birdclef_species)), dtype=np.float32)
    for h in unique_hours:
        i = hour_to_i[h]
        mask = hours.astype(int) == h
        hour_n[i] = mask.sum()
        hour_p[i] = labels_matrix[mask].mean(axis=0)

    # Site-hour priors
    sh_to_i = {}
    sh_n_list = []
    sh_p_list = []
    for s in unique_sites:
        for h in unique_hours:
            mask = (sites == s) & (hours.astype(int) == h)
            n = mask.sum()
            sh_to_i[(s, h)] = len(sh_n_list)
            sh_n_list.append(n)
            sh_p_list.append(labels_matrix[mask].mean(axis=0) if n > 0 else global_p)

    return {
        "global_p": global_p,
        "site_to_i": site_to_i,
        "site_n": np.array(site_n),
        "site_p": np.array(site_p),
        "hour_to_i": hour_to_i,
        "hour_n": np.array(hour_n),
        "hour_p": np.array(hour_p),
        "sh_to_i": sh_to_i,
        "sh_n": np.array(sh_n_list, dtype=np.float32),
        "sh_p": np.stack(sh_p_list).astype(np.float32)
        if sh_p_list
        else np.zeros((0, len(birdclef_species)), dtype=np.float32),
    }


def prior_logits_from_tables(sites, hours, tables, eps=1e-4):
    """Compute prior as logits: log(p/(1-p)). Matches chiranjithdharma."""
    n = len(sites)
    p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)

    # Hour blending
    hour_idx = np.fromiter(
        (
            tables["hour_to_i"].get(int(h), -1) if h is not None and int(h) >= 0 else -1
            for h in hours
        ),
        dtype=np.int32,
        count=n,
    )
    valid = hour_idx >= 0
    if valid.any():
        nh = tables["hour_n"][hour_idx[valid]][:, None]
        wh = nh / (nh + K_HOUR)
        p[valid] = wh * tables["hour_p"][hour_idx[valid]] + (1.0 - wh) * p[valid]

    # Site blending
    site_idx = np.fromiter(
        (tables["site_to_i"].get(str(s), -1) if s is not None else -1 for s in sites),
        dtype=np.int32,
        count=n,
    )
    valid = site_idx >= 0
    if valid.any():
        ns = tables["site_n"][site_idx[valid]][:, None]
        ws = ns / (ns + K_SITE)
        p[valid] = ws * tables["site_p"][site_idx[valid]] + (1.0 - ws) * p[valid]

    # Site-hour blending
    sh_idx = np.fromiter(
        (
            tables["sh_to_i"].get((str(s), int(h)), -1)
            if s is not None and h is not None and int(h) >= 0
            else -1
            for s, h in zip(sites, hours, strict=False)
        ),
        dtype=np.int32,
        count=n,
    )
    valid = sh_idx >= 0
    if valid.any():
        nsh = tables["sh_n"][sh_idx[valid]][:, None]
        wsh = nsh / (nsh + K_INTERACTION)
        p[valid] = wsh * tables["sh_p"][sh_idx[valid]] + (1.0 - wsh) * p[valid]

    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def fuse_scores_with_priors(base_logits, sites, hours, tables, species_idx):
    """Add priors to base logits (6-category, matching chiranjithdharma). Returns (fused, prior)."""
    scores = base_logits.copy()
    prior = prior_logits_from_tables(sites, hours, tables)

    idx = species_idx

    # Mapped active event: additive prior (lambda=0.4)
    if len(idx["mapped_event"]):
        scores[:, idx["mapped_event"]] += LAMBDA_EVENT * prior[:, idx["mapped_event"]]
    # Mapped active texture: additive prior (lambda=1.0)
    if len(idx["mapped_texture"]):
        scores[:, idx["mapped_texture"]] += LAMBDA_TEXTURE * prior[:, idx["mapped_texture"]]
    # Proxy active texture: additive prior (lambda=0.8)
    if len(idx["proxy_texture"]):
        scores[:, idx["proxy_texture"]] += LAMBDA_PROXY_TEXTURE * prior[:, idx["proxy_texture"]]
    # Prior-only active event: REPLACE score with scaled prior
    if len(idx["prioronly_event"]):
        scores[:, idx["prioronly_event"]] = LAMBDA_EVENT * prior[:, idx["prioronly_event"]]
    # Prior-only active texture: REPLACE score with scaled prior
    if len(idx["prioronly_texture"]):
        scores[:, idx["prioronly_texture"]] = LAMBDA_TEXTURE * prior[:, idx["prioronly_texture"]]
    # Unmapped inactive: flat -8.0
    if len(idx["unmapped_inactive"]):
        scores[:, idx["unmapped_inactive"]] = -8.0

    # Temporal smoothing in logit space
    scores = smooth_texture_cols(scores, idx["all_active_texture"].tolist(), SMOOTH_TEXTURE)
    scores = smooth_event_cols(scores, idx["all_active_event"].tolist(), SMOOTH_EVENT)

    return scores.astype(np.float32), prior


def smooth_texture_cols(scores, cols, alpha):
    """Average-neighbor smoothing for texture species (logit space)."""
    if alpha <= 0 or not cols or len(scores) % N_WINDOWS != 0:
        return scores
    s = scores.copy()
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return s


def smooth_event_cols(scores, cols, alpha):
    """Local-max propagation for event species (logit space)."""
    if alpha <= 0 or not cols or len(scores) % N_WINDOWS != 0:
        return scores
    s = scores.copy()
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    local_max = np.maximum(x, np.maximum(prev_x, next_x))
    view[:, :, cols] = (1.0 - alpha) * x + alpha * local_max
    return s


# ===========================================================================
# Probe training + inference (logit space)
# ===========================================================================


def seq_features_1d(v):
    """Compute temporal context features within file blocks of 12 windows."""
    assert len(v) % N_WINDOWS == 0
    x = v.reshape(-1, N_WINDOWS)
    prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(axis=1), N_WINDOWS)
    max_v = np.repeat(x.max(axis=1), N_WINDOWS)
    std_v = np.repeat(x.std(axis=1), N_WINDOWS)
    return prev_v, next_v, mean_v, max_v, std_v


def build_class_features(Z, raw_col, prior_col, base_col):
    """Build 77-dim feature vector per class (matching Yuriy ProtoSSM v5).

    Features: PCA(64) + raw(1) + prior(1) + base(1) + prev(1) + next(1) +
              mean(1) + max(1) + std(1) + diff_mean(1) + diff_prev(1) +
              diff_next(1) + raw*prior(1) + raw*base(1) + prior*base(1) = 77
    """
    prev_b, next_b, mean_b, max_b, std_b = seq_features_1d(base_col)
    diff_mean = base_col - mean_b
    diff_prev = base_col - prev_b
    diff_next = base_col - next_b
    return np.concatenate(
        [
            Z,  # PCA embeddings (64)
            raw_col[:, None],  # raw Perch logit (1)
            prior_col[:, None],  # prior logit (1)
            base_col[:, None],  # fused base logit (1)
            prev_b[:, None],  # prev window base (1)
            next_b[:, None],  # next window base (1)
            mean_b[:, None],  # file mean base (1)
            max_b[:, None],  # file max base (1)
            std_b[:, None],  # file std base (1)
            diff_mean[:, None],  # deviation from mean (1)
            diff_prev[:, None],  # onset detection (1)
            diff_next[:, None],  # offset detection (1)
            (raw_col * prior_col)[:, None],  # interaction (1)
            (raw_col * base_col)[:, None],  # interaction (1)
            (prior_col * base_col)[:, None],  # interaction (1)
        ],
        axis=1,
    ).astype(np.float32)


def build_labels_matrix(meta_rows, labels_df, birdclef_species):
    """Build binary label matrix (n_windows, n_species)."""
    n_windows = len(meta_rows)
    n_species = len(birdclef_species)
    species_to_idx = {s: i for i, s in enumerate(birdclef_species)}
    labels_matrix = np.zeros((n_windows, n_species), dtype=np.float32)

    for window_idx, row in enumerate(meta_rows):
        start_s = row["window_start"]
        end_s = row["window_end"]
        start_str = (
            f"{int(start_s // 3600):02d}:{int((start_s % 3600) // 60):02d}:{int(start_s % 60):02d}"
        )
        end_str = f"{int(end_s // 3600):02d}:{int((end_s % 3600) // 60):02d}:{int(end_s % 60):02d}"

        mask = (
            (labels_df["filename"] == row["filename"])
            & (labels_df["start"] == start_str)
            & (labels_df["end"] == end_str)
        )
        for _, label_row in labels_df[mask].iterrows():
            for label in str(label_row["primary_label"]).split(";"):
                label = label.strip()
                if label in species_to_idx:
                    labels_matrix[window_idx, species_to_idx[label]] = 1.0

    return labels_matrix


# ===========================================================================
# Section 7: PyTorch models (inline from models.py)
# ===========================================================================


class SelectiveSSM(nn.Module):
    """Simplified Mamba-style selective state space model.

    Input-dependent (selective) discretization of continuous-time SSM.
    For T=12 bioacoustic windows, the sequential scan is efficient on CPU.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, T, D) -> (B, T, D)."""
        B_size, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        h = torch.zeros(B_size, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :]
            dA = torch.exp(A[None, :, :] * dt_t[:, :, None])
            dB = dt_t[:, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            y_t = (h * C[:, t, None, :]).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class TemporalCrossAttention(nn.Module):
    """Multi-head self-attention between temporal windows.

    Captures non-local patterns (e.g., dawn chorus onset, counter-singing)
    that sequential SSM may miss. Uses pre-norm residual blocks.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, T, D) -> (B, T, D)."""
        # Self-attention with pre-norm
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class ProtoSSMv2(nn.Module):
    """Prototypical State Space Model v4 with cross-attention and metadata.

    Architecture:
      1. Linear(1536 -> d_model) + LayerNorm + GELU + Dropout
      2. Learnable positional encoding (1, 12, d_model)
      3. Site/hour metadata embeddings -> projected to d_model
      4. Bidirectional SSM layers x n_ssm_layers
      5. Optional cross-attention layer
      6. Cosine similarity with learnable prototypes x temperature + bias
      7. Gated fusion with Perch logits: alpha*sim + (1-alpha)*perch
      8. Optional taxonomic auxiliary head
    """

    def __init__(
        self,
        d_input: int = 1536,
        d_model: int = 128,
        d_state: int = 16,
        n_ssm_layers: int = 2,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.15,
        n_sites: int = 20,
        meta_dim: int = 16,
        use_cross_attn: bool = True,
        cross_attn_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows

        # 1. Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        # 3. Metadata embeddings
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        # 4. Bidirectional SSM layers
        self.ssm_fwd = nn.ModuleList()
        self.ssm_bwd = nn.ModuleList()
        self.ssm_merge = nn.ModuleList()
        self.ssm_norm = nn.ModuleList()
        for _ in range(n_ssm_layers):
            self.ssm_fwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_bwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_merge.append(nn.Linear(2 * d_model, d_model))
            self.ssm_norm.append(nn.LayerNorm(d_model))
        self.ssm_drop = nn.Dropout(dropout)

        # 4b. Cross-attention after SSM
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(
                d_model, n_heads=cross_attn_heads, dropout=dropout
            )

        # 5. Learnable class prototypes
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))

        # 6. Per-class calibration bias
        self.class_bias = nn.Parameter(torch.zeros(n_classes))

        # 7. Per-class gated fusion with Perch logits
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        # 8. Taxonomic auxiliary head (initialized later if needed)
        self.n_families = 0
        self.family_head = None

    def init_prototypes_from_data(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Initialize prototypes as class-mean projected embeddings."""
        with torch.no_grad():
            h = self.input_proj(embeddings)
            for c in range(self.n_classes):
                mask = labels[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def init_family_head(self, n_families: int, class_to_family: list) -> None:
        """Initialize taxonomic auxiliary head."""
        self.n_families = n_families
        self.family_head = nn.Linear(self.d_model, n_families)
        self.register_buffer("class_to_family", torch.tensor(class_to_family, dtype=torch.long))

    def forward(
        self,
        emb: torch.Tensor,
        perch_logits: torch.Tensor | None = None,
        site_ids: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
    ) -> tuple:
        """Forward pass.

        Args:
            emb: (B, T, 1536) Perch embeddings
            perch_logits: (B, T, 234) Perch mapped logits
            site_ids: (B,) site indices
            hours: (B,) hour values (0-23)

        Returns:
            species_logits: (B, T, n_classes)
            family_logits: (B, n_families) or None
            h_temporal: (B, T, d_model) final SSM output
        """
        B, T, _ = emb.shape

        # Project embeddings
        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :T, :]

        # Add metadata embeddings
        if site_ids is not None and hours is not None:
            s_emb = self.site_emb(site_ids)
            h_emb = self.hour_emb(hours)
            meta = self.meta_proj(torch.cat([s_emb, h_emb], dim=-1))
            h = h + meta[:, None, :]

        # Bidirectional SSM
        for fwd, bwd, merge, norm in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm, strict=False
        ):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)

        # Cross-attention for non-local temporal patterns
        if self.use_cross_attn:
            h = self.cross_attn(h)

        h_temporal = h

        # Prototypical cosine similarity + class bias
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        temp = F.softplus(self.proto_temp)
        sim = torch.matmul(h_norm, p_norm.T) * temp + self.class_bias[None, None, :]

        # Gated fusion with Perch logits
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim

        # Taxonomic auxiliary prediction
        family_logits = None
        if self.family_head is not None:
            h_pool = h.mean(dim=1)
            family_logits = self.family_head(h_pool)

        return species_logits, family_logits, h_temporal

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualSSM(nn.Module):
    """Lightweight SSM for second-pass correction on first-pass errors.

    Takes embeddings + first-pass scores, predicts additive corrections.
    Output head is zero-initialized so corrections start small.
    """

    def __init__(
        self,
        d_input: int = 1536,
        d_scores: int = 234,
        d_model: int = 64,
        d_state: int = 8,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.1,
        n_sites: int = 20,
        meta_dim: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        # Project embeddings + first-pass scores
        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Metadata
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        # Single bidirectional SSM layer (lightweight)
        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm_drop = nn.Dropout(dropout)

        # Output: per-class correction (additive)
        self.output_head = nn.Linear(d_model, n_classes)

        # Initialize output near zero (corrections start small)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        emb: torch.Tensor,
        first_pass_scores: torch.Tensor,
        site_ids: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            emb: (B, T, 1536)
            first_pass_scores: (B, T, 234) logits from first pass
            site_ids: (B,)
            hours: (B,)

        Returns:
            correction: (B, T, n_classes) additive correction
        """
        B, T, _ = emb.shape

        # Concatenate embeddings with first-pass scores
        x = torch.cat([emb, first_pass_scores], dim=-1)
        h = self.input_proj(x)

        # Add metadata
        if site_ids is not None and hours is not None:
            site_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            hour_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([site_e, hour_e], dim=-1))
            h = h + meta.unsqueeze(1)

        h = h + self.pos_enc[:, :T, :]

        # Bidirectional SSM
        residual = h
        h_f = self.ssm_fwd(h)
        h_b = self.ssm_bwd(h.flip(1)).flip(1)
        h = self.ssm_merge(torch.cat([h_f, h_b], dim=-1))
        h = self.ssm_drop(h)
        h = self.ssm_norm(h + residual)

        # Output correction
        correction = self.output_head(h)
        return correction

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Section 8: Losses + mixup (inline from losses.py)
# ===========================================================================


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for multi-label classification.

    Reduces contribution of easy examples, focuses on hard ones.
    When gamma=0, equivalent to standard BCE.
    """
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * bce

    if reduction == "mean":
        return loss.mean()
    return loss


def mixup_files(
    emb: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    site_ids: np.ndarray,
    hours: np.ndarray,
    families: np.ndarray | None,
    alpha: float = 0.3,
) -> tuple:
    """File-level mixup augmentation for ProtoSSM training.

    Mixes pairs of files with random lambda from Beta(alpha, alpha).
    Lambda is enforced >= 0.5 so the dominant sample stays dominant.
    Discrete features (site, hour) keep the dominant sample's values.
    """
    n = len(emb)
    if alpha <= 0 or n < 2:
        return emb, logits, labels, site_ids, hours, families

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5

    perm = np.random.permutation(n)

    emb_mix = lam * emb + (1 - lam) * emb[perm]
    logits_mix = lam * logits + (1 - lam) * logits[perm]
    labels_mix = lam * labels + (1 - lam) * labels[perm]

    # For discrete features (site, hour), keep the dominant sample's values
    families_mix = lam * families + (1 - lam) * families[perm] if families is not None else None

    return emb_mix, logits_mix, labels_mix, site_ids, hours, families_mix


# ===========================================================================
# Section 9: Data utils (inline from data_utils.py)
# ===========================================================================


def macro_auc_skip_empty(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute macro AUC, skipping classes with no positive samples."""
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def reshape_to_files(
    flat_array: np.ndarray,
    meta_df: pd.DataFrame,
    n_windows: int = N_WINDOWS,
) -> tuple:
    """Reshape flat window-level arrays to (n_files, n_windows, ...) grouped by filename.

    Args:
        flat_array: (N, ...) window-level data
        meta_df: DataFrame with 'filename' column, ordered by file then window
        n_windows: windows per file (12)

    Returns:
        file_array: (n_files, n_windows, ...) reshaped data
        file_list: ordered list of unique filenames
    """
    assert len(flat_array) % n_windows == 0
    file_list = list(dict.fromkeys(meta_df["filename"].tolist()))
    n_files = len(file_list)
    assert n_files * n_windows == len(
        flat_array
    ), f"Expected {n_files}*{n_windows}={n_files * n_windows} rows, got {len(flat_array)}"

    rest_shape = flat_array.shape[1:]
    return flat_array.reshape(n_files, n_windows, *rest_shape), file_list


def build_site_mapping(
    meta_df: pd.DataFrame,
    n_sites_max: int = 20,
) -> tuple:
    """Map unique sites to integer indices.

    Returns:
        site_to_idx: mapping from site name to integer index
        n_sites: number of unique sites (capped at n_sites_max)
    """
    sites = meta_df["site"].dropna().unique().tolist() if "site" in meta_df.columns else []
    sites = sorted({str(s) for s in sites})
    site_to_idx = {s: min(i, n_sites_max - 1) for i, s in enumerate(sites)}
    return site_to_idx, min(len(sites), n_sites_max)


def get_file_metadata(
    meta_df: pd.DataFrame,
    file_list: list,
    site_to_idx: dict,
    n_sites_max: int = 20,
) -> tuple:
    """Extract per-file site_ids and hour_ids arrays.

    Returns:
        site_ids: (n_files,) integer site indices
        hour_ids: (n_files,) integer hours (0-23)
    """
    # Build lookup from filename to first row
    file_to_row = {}
    for i, row in meta_df.iterrows():
        fn = row["filename"]
        if fn not in file_to_row:
            file_to_row[fn] = i

    sites = meta_df["site"].values if "site" in meta_df.columns else np.full(len(meta_df), None)
    hours = meta_df["hour"].values if "hour" in meta_df.columns else np.zeros(len(meta_df))

    site_ids = np.zeros(len(file_list), dtype=np.int64)
    hour_ids = np.zeros(len(file_list), dtype=np.int64)

    for fi, fname in enumerate(file_list):
        row_idx = file_to_row.get(fname)
        if row_idx is not None:
            s = sites[row_idx]
            sid = site_to_idx.get(str(s), 0) if s is not None else 0
            site_ids[fi] = min(sid, n_sites_max - 1)
            h = hours[row_idx]
            hour_ids[fi] = (
                int(h) % 24 if h is not None and not (isinstance(h, float) and np.isnan(h)) else 0
            )

    return site_ids, hour_ids


def build_taxonomy_groups(
    taxonomy_df: pd.DataFrame,
    primary_labels: list,
) -> tuple:
    """Build family grouping for auxiliary taxonomic head.

    Returns:
        n_families: number of unique families
        class_to_family: list mapping class index to family index
        family_to_idx: mapping from family name to integer index
    """
    # BISECT: taxonomy change only (genus 161 → class_name 5)
    group_map = {}
    for col in ["family", "order", "class_name"]:
        if col in taxonomy_df.columns:
            group_map = taxonomy_df.set_index("primary_label")[col].to_dict()
            break

    groups = sorted(set(group_map.values())) if group_map else ["Unknown"]
    family_to_idx = {g: i for i, g in enumerate(groups)}

    class_to_family = [
        family_to_idx.get(group_map.get(lbl, "Unknown"), 0) for lbl in primary_labels
    ]

    return len(groups), class_to_family, family_to_idx


def build_file_families(
    labels_files: np.ndarray,
    class_to_family: list,
    n_families: int,
) -> np.ndarray:
    """Build per-file multi-hot family label matrix.

    Args:
        labels_files: (n_files, n_windows, n_classes)
        class_to_family: list mapping class index to family index
        n_families: total number of families

    Returns:
        file_families: (n_files, n_families) multi-hot family labels
    """
    n_files = labels_files.shape[0]
    # Aggregate labels across windows: any positive window counts
    file_labels = labels_files.max(axis=1)  # (n_files, n_classes)

    file_families = np.zeros((n_files, n_families), dtype=np.float32)
    for cls_idx, fam_idx in enumerate(class_to_family):
        mask = file_labels[:, cls_idx] > 0.5
        file_families[mask, fam_idx] = 1.0

    return file_families


# ===========================================================================
# Section 10: Ensemble (inline from ensemble.py)
# ===========================================================================


def temporal_shift_tta(
    emb_files: np.ndarray,
    logits_files: np.ndarray,
    model: nn.Module,
    site_ids: np.ndarray,
    hours: np.ndarray,
    shifts: list | None = None,
) -> np.ndarray:
    """TTA by circular-shifting the 12-window embedding sequence.

    For each shift: roll embeddings/logits -> forward pass -> unroll -> average.

    Args:
        emb_files: (n_files, 12, 1536)
        logits_files: (n_files, 12, 234)
        model: ProtoSSMv2 instance
        site_ids: (n_files,) site indices
        hours: (n_files,) hour values
        shifts: list of shifts (default [0, 1, -1])

    Returns:
        averaged_preds: (n_files, 12, 234) averaged logits
    """
    if shifts is None:
        shifts = [0, 1, -1]

    all_preds = []
    model.eval()

    for shift in shifts:
        if shift == 0:
            e = emb_files
            lo = logits_files
        else:
            e = np.roll(emb_files, shift, axis=1)
            lo = np.roll(logits_files, shift, axis=1)

        with torch.no_grad():
            out, _, _ = model(
                torch.tensor(e, dtype=torch.float32),
                torch.tensor(lo, dtype=torch.float32),
                site_ids=torch.tensor(site_ids, dtype=torch.long),
                hours=torch.tensor(hours, dtype=torch.long),
            )
            pred = out.numpy()

        if shift != 0:
            pred = np.roll(pred, -shift, axis=1)

        all_preds.append(pred)

    return np.mean(all_preds, axis=0)


def optimize_ensemble_weight(
    oof_proto_flat: np.ndarray,
    oof_mlp_flat: np.ndarray,
    y_true_flat: np.ndarray,
) -> tuple:
    """Grid search over blend weights to find optimal ProtoSSM ensemble weight.

    Blended = w * proto + (1-w) * mlp

    Returns:
        best_w: optimal weight for proto
        best_auc: macro AUC at best weight
        results: list of (weight, auc) tuples
    """
    weights = np.arange(0.0, 1.05, 0.05)
    results = []

    for w in weights:
        blended = w * oof_proto_flat + (1.0 - w) * oof_mlp_flat
        try:
            auc = macro_auc_skip_empty(y_true_flat, blended)
        except Exception:
            auc = 0.0
        results.append((float(w), auc))

    best_w, best_auc = max(results, key=lambda x: x[1])
    return best_w, best_auc, results


def fuse_protossm_mlp(
    proto_scores: np.ndarray,
    mlp_scores: np.ndarray,
    weight_proto: float,
) -> np.ndarray:
    """Weighted ensemble of ProtoSSM and MLP probe scores.

    final = weight_proto * proto + (1 - weight_proto) * mlp
    """
    return (weight_proto * proto_scores + (1.0 - weight_proto) * mlp_scores).astype(np.float32)


# ===========================================================================
# Section 11: Training loops (inline from training.py)
# ===========================================================================


def train_proto_ssm_single(
    model,
    emb_train: np.ndarray,
    logits_train: np.ndarray,
    labels_train: np.ndarray,
    site_ids_train: np.ndarray | None = None,
    hours_train: np.ndarray | None = None,
    emb_val: np.ndarray | None = None,
    logits_val: np.ndarray | None = None,
    labels_val: np.ndarray | None = None,
    site_ids_val: np.ndarray | None = None,
    hours_val: np.ndarray | None = None,
    file_families_train: np.ndarray | None = None,
    file_families_val: np.ndarray | None = None,
    cfg: dict | None = None,
    verbose: bool = True,
) -> tuple:
    """Train a single ProtoSSM model with focal loss, mixup, and SWA.

    Args:
        model: ProtoSSMv2 instance (already initialized)
        emb_train: (n_files, 12, 1536)
        logits_train: (n_files, 12, 234)
        labels_train: (n_files, 12, 234)
        cfg: training config dict

    Returns:
        model: trained model (with SWA or best state)
        history: dict with train_loss, val_loss, val_auc lists
    """
    if cfg is None:
        cfg = {
            "n_epochs": PROTO_SSM_N_EPOCHS,
            "lr": PROTO_SSM_LR,
            "weight_decay": PROTO_SSM_WEIGHT_DECAY,
            "patience": PROTO_SSM_PATIENCE,
            "pos_weight_cap": PROTO_SSM_POS_WEIGHT_CAP,
            "distill_weight": PROTO_SSM_DISTILL_WEIGHT,
            "label_smoothing": PROTO_SSM_LABEL_SMOOTHING,
            "oof_n_splits": PROTO_SSM_OOF_N_SPLITS,
            "mixup_alpha": PROTO_SSM_MIXUP_ALPHA,
            "focal_gamma": PROTO_SSM_FOCAL_GAMMA,
            "swa_start_frac": PROTO_SSM_SWA_START_FRAC,
        }

    label_smoothing = cfg.get("label_smoothing", 0.0)
    mixup_alpha = cfg.get("mixup_alpha", 0.0)
    focal_gamma = cfg.get("focal_gamma", 0.0)
    swa_start_frac = cfg.get("swa_start_frac", 1.0)
    n_epochs = cfg["n_epochs"]
    swa_start_epoch = int(n_epochs * swa_start_frac)

    # Apply label smoothing to base labels
    labels_np = labels_train.copy()
    if label_smoothing > 0:
        labels_np = labels_np * (1.0 - label_smoothing) + label_smoothing / 2.0

    has_val = emb_val is not None
    if has_val:
        emb_v = torch.tensor(emb_val, dtype=torch.float32)
        logits_v = torch.tensor(logits_val, dtype=torch.float32)
        labels_v = torch.tensor(labels_val, dtype=torch.float32)
        site_v = torch.tensor(site_ids_val, dtype=torch.long) if site_ids_val is not None else None
        hour_v = torch.tensor(hours_val, dtype=torch.long) if hours_val is not None else None

    # Note: fam_v not used in val forward pass (family loss only computed on train)
    _ = (
        file_families_val  # kept for reference
        if (has_val and file_families_val is not None)
        else None
    )

    # Class weights for imbalanced data
    labels_tr_t = torch.tensor(labels_np, dtype=torch.float32)
    pos_counts = labels_tr_t.sum(dim=(0, 1))
    total = labels_tr_t.shape[0] * labels_tr_t.shape[1]
    pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        epochs=n_epochs,
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    # SWA state accumulator
    swa_state = None
    swa_count = 0

    for epoch in range(n_epochs):
        # === Mixup augmentation (skip first 5 epochs for warmup) ===
        if mixup_alpha > 0 and epoch > 5:
            emb_mix, logits_mix, labels_mix, _, _, fam_mix = mixup_files(
                emb_train,
                logits_train,
                labels_np,
                site_ids_train,
                hours_train,
                file_families_train,
                alpha=mixup_alpha,
            )
        else:
            emb_mix, logits_mix, labels_mix = emb_train, logits_train, labels_np
            fam_mix = file_families_train

        emb_tr = torch.tensor(emb_mix, dtype=torch.float32)
        logits_tr = torch.tensor(logits_mix, dtype=torch.float32)
        labels_tr = torch.tensor(labels_mix, dtype=torch.float32)
        site_tr = (
            torch.tensor(site_ids_train, dtype=torch.long) if site_ids_train is not None else None
        )
        hour_tr = torch.tensor(hours_train, dtype=torch.long) if hours_train is not None else None
        fam_tr = torch.tensor(fam_mix, dtype=torch.float32) if fam_mix is not None else None

        # === Train ===
        model.train()
        species_out, family_out, _ = model(emb_tr, logits_tr, site_ids=site_tr, hours=hour_tr)

        # Primary loss: focal BCE or weighted BCE
        if focal_gamma > 0:
            loss_main = focal_bce_with_logits(
                species_out,
                labels_tr,
                gamma=focal_gamma,
                pos_weight=pos_weight[None, None, :],
            )
        else:
            loss_main = F.binary_cross_entropy_with_logits(
                species_out, labels_tr, pos_weight=pos_weight[None, None, :]
            )

        # Knowledge distillation loss
        loss_distill = F.mse_loss(species_out, logits_tr)

        # Total loss
        loss = loss_main + cfg["distill_weight"] * loss_distill

        # Taxonomic auxiliary loss
        if family_out is not None and fam_tr is not None:
            loss_family = F.binary_cross_entropy_with_logits(family_out, fam_tr)
            loss = loss + 0.1 * loss_family

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # === SWA accumulation ===
        if epoch >= swa_start_epoch:
            if swa_state is None:
                swa_state = {k: v.clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k in swa_state:
                    swa_state[k] += model.state_dict()[k]
                swa_count += 1

        # === Validate ===
        model.eval()
        with torch.no_grad():
            if has_val:
                val_out, val_fam, _ = model(emb_v, logits_v, site_ids=site_v, hours=hour_v)
                val_loss = F.binary_cross_entropy_with_logits(
                    val_out, labels_v, pos_weight=pos_weight[None, None, :]
                )

                val_pred = val_out.reshape(-1, val_out.shape[-1]).numpy()
                val_true = labels_v.reshape(-1, labels_v.shape[-1]).numpy()
                try:
                    val_auc = macro_auc_skip_empty(val_true, val_pred)
                except Exception:
                    val_auc = 0.0
            else:
                val_loss = loss
                val_auc = 0.0

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_auc"].append(val_auc)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch + 1) % 20 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            swa_info = f" swa={swa_count}" if swa_count > 0 else ""
            print(
                f"  Epoch {epoch + 1:3d}: train={loss.item():.4f} val={val_loss.item():.4f} "
                f"auc={val_auc:.4f} lr={lr_now:.6f} wait={wait}{swa_info}"
            )

        if wait >= cfg["patience"]:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})")
            break

    # Apply SWA if we accumulated enough checkpoints (>=3)
    if swa_state is not None and swa_count >= 3:
        if verbose:
            print(f"  Applying SWA (averaged {swa_count} checkpoints)")
        avg_state = {k: v / swa_count for k, v in swa_state.items()}
        model.load_state_dict(avg_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"  Training complete. Best val_loss={best_val_loss:.4f}")
        with torch.no_grad():
            alphas = torch.sigmoid(model.fusion_alpha).numpy()
            print(
                f"  Fusion alpha: mean={alphas.mean():.3f} min={alphas.min():.3f} "
                f"max={alphas.max():.3f}"
            )
            print(f"  Proto temperature: {F.softplus(model.proto_temp).item():.3f}")

    return model, history


def run_proto_ssm_oof(
    emb_files: np.ndarray,
    logits_files: np.ndarray,
    labels_files: np.ndarray,
    site_ids_all: np.ndarray,
    hours_all: np.ndarray,
    file_families: np.ndarray,
    file_groups: np.ndarray,
    n_families: int,
    class_to_family: list,
    ssm_cfg: dict | None = None,
    train_cfg: dict | None = None,
    tta_shifts: list | None = None,
    n_classes: int = N_SPECIES,
    n_windows: int = N_WINDOWS,
    verbose: bool = True,
) -> tuple:
    """Run GroupKFold OOF cross-validation for ProtoSSM.

    Returns:
        oof_preds: (n_files, 12, 234) OOF predictions
        fold_histories: list of training history dicts
        fold_alphas: list of per-fold fusion alpha arrays
    """
    if ssm_cfg is None:
        ssm_cfg = {
            "d_model": PROTO_SSM_D_MODEL,
            "d_state": PROTO_SSM_D_STATE,
            "n_ssm_layers": PROTO_SSM_N_LAYERS,
            "dropout": PROTO_SSM_DROPOUT,
            "n_sites": PROTO_SSM_N_SITES,
            "meta_dim": PROTO_SSM_META_DIM,
            "use_cross_attn": PROTO_SSM_USE_CROSS_ATTN,
            "cross_attn_heads": PROTO_SSM_CROSS_ATTN_HEADS,
        }
    if train_cfg is None:
        train_cfg = {
            "n_epochs": PROTO_SSM_N_EPOCHS,
            "lr": PROTO_SSM_LR,
            "weight_decay": PROTO_SSM_WEIGHT_DECAY,
            "patience": PROTO_SSM_PATIENCE,
            "pos_weight_cap": PROTO_SSM_POS_WEIGHT_CAP,
            "distill_weight": PROTO_SSM_DISTILL_WEIGHT,
            "label_smoothing": PROTO_SSM_LABEL_SMOOTHING,
            "oof_n_splits": PROTO_SSM_OOF_N_SPLITS,
            "mixup_alpha": PROTO_SSM_MIXUP_ALPHA,
            "focal_gamma": PROTO_SSM_FOCAL_GAMMA,
            "swa_start_frac": PROTO_SSM_SWA_START_FRAC,
        }
    if tta_shifts is None:
        tta_shifts = [0, 1, -1]

    n_splits = train_cfg.get("oof_n_splits", 3)
    n_files = len(emb_files)

    oof_preds = np.zeros((n_files, n_windows, n_classes), dtype=np.float32)
    fold_histories = []
    fold_alphas = []

    n_unique_groups = len(set(file_groups))
    if n_unique_groups < n_splits:
        if verbose:
            print(
                f"  WARNING: Only {n_unique_groups} groups, reducing n_splits "
                f"from {n_splits} to {n_unique_groups}"
            )
        n_splits = n_unique_groups

    gkf = GroupKFold(n_splits=n_splits)
    dummy_y = np.zeros(n_files)

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(dummy_y, dummy_y, file_groups)):
        if verbose:
            print(
                f"\n--- Fold {fold_i + 1}/{n_splits} "
                f"(train={len(train_idx)}, val={len(val_idx)}) ---"
            )

        fold_model = ProtoSSMv2(
            d_input=emb_files.shape[2],
            d_model=ssm_cfg["d_model"],
            d_state=ssm_cfg["d_state"],
            n_ssm_layers=ssm_cfg["n_ssm_layers"],
            n_classes=n_classes,
            n_windows=n_windows,
            dropout=ssm_cfg["dropout"],
            n_sites=ssm_cfg["n_sites"],
            meta_dim=ssm_cfg["meta_dim"],
            use_cross_attn=ssm_cfg.get("use_cross_attn", True),
            cross_attn_heads=ssm_cfg.get("cross_attn_heads", 4),
        )

        # Initialize prototypes from fold training data
        emb_flat_fold = emb_files[train_idx].reshape(-1, emb_files.shape[2])
        labels_flat_fold = labels_files[train_idx].reshape(-1, n_classes)
        fold_model.init_prototypes_from_data(
            torch.tensor(emb_flat_fold, dtype=torch.float32),
            torch.tensor(labels_flat_fold, dtype=torch.float32),
        )
        fold_model.init_family_head(n_families, class_to_family)

        # Train on fold
        fold_model, fold_hist = train_proto_ssm_single(
            fold_model,
            emb_files[train_idx],
            logits_files[train_idx],
            labels_files[train_idx].astype(np.float32),
            site_ids_train=site_ids_all[train_idx],
            hours_train=hours_all[train_idx],
            emb_val=emb_files[val_idx],
            logits_val=logits_files[val_idx],
            labels_val=labels_files[val_idx].astype(np.float32),
            site_ids_val=site_ids_all[val_idx],
            hours_val=hours_all[val_idx],
            file_families_train=file_families[train_idx],
            file_families_val=file_families[val_idx],
            cfg=train_cfg,
            verbose=verbose,
        )

        # OOF predictions with TTA
        fold_model.eval()
        if len(tta_shifts) > 1:
            oof_preds[val_idx] = temporal_shift_tta(
                emb_files[val_idx],
                logits_files[val_idx],
                fold_model,
                site_ids_all[val_idx],
                hours_all[val_idx],
                shifts=tta_shifts,
            )
        else:
            with torch.no_grad():
                val_emb = torch.tensor(emb_files[val_idx], dtype=torch.float32)
                val_logits = torch.tensor(logits_files[val_idx], dtype=torch.float32)
                val_sites = torch.tensor(site_ids_all[val_idx], dtype=torch.long)
                val_hours = torch.tensor(hours_all[val_idx], dtype=torch.long)
                val_out, _, _ = fold_model(val_emb, val_logits, site_ids=val_sites, hours=val_hours)
                oof_preds[val_idx] = val_out.numpy()

        fold_alphas.append(torch.sigmoid(fold_model.fusion_alpha).detach().numpy().copy())
        fold_histories.append(fold_hist)

    return oof_preds, fold_histories, fold_alphas


def train_residual_ssm(
    emb_files: np.ndarray,
    first_pass_files: np.ndarray,
    labels_files: np.ndarray,
    site_ids_all: np.ndarray,
    hours_all: np.ndarray,
    cfg: dict | None = None,
    n_sites: int = 20,
    wall_time_limit: float | None = None,
    start_time: float | None = None,
    verbose: bool = True,
) -> tuple:
    """Train ResidualSSM on first-pass errors.

    Args:
        emb_files: (n_files, 12, 1536) embeddings
        first_pass_files: (n_files, 12, 234) first-pass ensemble logits
        labels_files: (n_files, 12, 234) ground truth
        site_ids_all: (n_files,) site indices
        hours_all: (n_files,) hours
        cfg: residual SSM config
        wall_time_limit: max wall time in minutes (skip if exceeded)
        start_time: wall clock start time

    Returns:
        res_model: trained ResidualSSM or None if skipped
        correction_weight: weight for correction (0.0 if skipped)
    """
    if cfg is None:
        cfg = {
            "d_model": RESIDUAL_SSM_D_MODEL,
            "d_state": RESIDUAL_SSM_D_STATE,
            "dropout": RESIDUAL_SSM_DROPOUT,
            "correction_weight": RESIDUAL_SSM_CORRECTION_WEIGHT,
            "n_epochs": RESIDUAL_SSM_N_EPOCHS,
            "lr": RESIDUAL_SSM_LR,
            "patience": RESIDUAL_SSM_PATIENCE,
        }

    # Wall-time safety gate
    if wall_time_limit is not None and start_time is not None:
        elapsed_min = (time.time() - start_time) / 60.0
        if elapsed_min > wall_time_limit:
            if verbose:
                print(
                    f"  SKIPPED ResidualSSM (wall time {elapsed_min:.1f}min > {wall_time_limit}min)"
                )
            return None, 0.0

    n_classes = first_pass_files.shape[2]

    # Compute residuals: target - sigmoid(first_pass)
    first_pass_probs = 1.0 / (1.0 + np.exp(-np.clip(first_pass_files, -30, 30)))
    residuals = labels_files.astype(np.float32) - first_pass_probs

    if verbose:
        print(
            f"  Residuals: mean={residuals.mean():.4f}, std={residuals.std():.4f}, "
            f"abs_mean={np.abs(residuals).mean():.4f}"
        )

    # Create model
    res_model = ResidualSSM(
        d_input=emb_files.shape[2],
        d_scores=n_classes,
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        n_classes=n_classes,
        n_windows=emb_files.shape[1],
        dropout=cfg["dropout"],
        n_sites=n_sites,
        meta_dim=8,
    )

    if verbose:
        print(f"  ResidualSSM parameters: {res_model.count_parameters():,}")

    # Train/val split (15% val, seed 123)
    n_files = len(emb_files)
    n_val = max(1, int(n_files * 0.15))
    perm = torch.randperm(n_files, generator=torch.Generator().manual_seed(123))
    val_i = perm[:n_val].numpy()
    train_i = perm[n_val:].numpy()

    emb_tr = torch.tensor(emb_files[train_i], dtype=torch.float32)
    fp_tr = torch.tensor(first_pass_files[train_i], dtype=torch.float32)
    res_tr = torch.tensor(residuals[train_i], dtype=torch.float32)
    site_tr = torch.tensor(site_ids_all[train_i], dtype=torch.long)
    hour_tr = torch.tensor(hours_all[train_i], dtype=torch.long)

    emb_va = torch.tensor(emb_files[val_i], dtype=torch.float32)
    fp_va = torch.tensor(first_pass_files[val_i], dtype=torch.float32)
    res_va = torch.tensor(residuals[val_i], dtype=torch.float32)
    site_va = torch.tensor(site_ids_all[val_i], dtype=torch.long)
    hour_va = torch.tensor(hours_all[val_i], dtype=torch.long)

    optimizer = torch.optim.AdamW(res_model.parameters(), lr=cfg["lr"], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        epochs=cfg["n_epochs"],
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    t0 = time.time()
    for epoch in range(cfg["n_epochs"]):
        res_model.train()
        correction = res_model(emb_tr, fp_tr, site_ids=site_tr, hours=hour_tr)
        loss = F.mse_loss(correction, res_tr)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(res_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        res_model.eval()
        with torch.no_grad():
            val_corr = res_model(emb_va, fp_va, site_ids=site_va, hours=hour_va)
            val_loss = F.mse_loss(val_corr, res_va)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in res_model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"  ResidualSSM epoch {epoch + 1}: "
                f"train={loss.item():.6f} val={val_loss.item():.6f} wait={wait}"
            )

        if wait >= cfg["patience"]:
            if verbose:
                print(f"  ResidualSSM early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        res_model.load_state_dict(best_state)

    res_time = time.time() - t0
    if verbose:
        print(f"  ResidualSSM training: {res_time:.1f}s, best val MSE: {best_val_loss:.6f}")

    return res_model, cfg["correction_weight"]


# ===========================================================================
# Section 12: Post-processing (inline from postprocessing.py)
# ===========================================================================


def temperature_scale(
    scores: np.ndarray,
    class_temperatures: np.ndarray,
) -> np.ndarray:
    """Stage 1: Per-taxon temperature scaling on logits, then sigmoid."""
    scaled = scores / class_temperatures[None, :]
    return sigmoid(scaled)


def file_level_confidence_scale(
    probs: np.ndarray,
    n_windows: int = N_WINDOWS,
    top_k: int = 2,
) -> np.ndarray:
    """Stage 2: Scale each window by file's top-K mean confidence."""
    N, C = probs.shape
    assert N % n_windows == 0
    view = probs.reshape(-1, n_windows, C)
    sorted_view = np.sort(view, axis=1)
    top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
    scaled = view * top_k_mean
    return scaled.reshape(N, C)


def rank_aware_scaling(
    scores: np.ndarray,
    n_windows: int = N_WINDOWS,
    power: float = 0.5,
) -> np.ndarray:
    """Stage 3: Scale each window by (file_max)^power.

    Suppresses predictions in uncertain files, boosts confident files.
    """
    N, C = scores.shape
    assert N % n_windows == 0
    n_files = N // n_windows

    view = scores.reshape(n_files, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)  # (F, 1, C)
    scale = np.power(file_max, power)
    scaled = view * scale
    return scaled.reshape(N, C)


def delta_shift_smooth(
    scores: np.ndarray,
    n_windows: int = N_WINDOWS,
    alpha: float = 0.15,
) -> np.ndarray:
    """Stage 4: Temporal smoothing across windows.

    new[t] = (1-alpha)*old[t] + 0.5*alpha*(old[t-1] + old[t+1])
    """
    N, C = scores.shape
    assert N % n_windows == 0
    n_files = N // n_windows

    view = scores.reshape(n_files, n_windows, C)
    prev_view = np.concatenate([view[:, :1, :], view[:, :-1, :]], axis=1)
    next_view = np.concatenate([view[:, 1:, :], view[:, -1:, :]], axis=1)
    smoothed = (1 - alpha) * view + 0.5 * alpha * (prev_view + next_view)
    return smoothed.reshape(N, C)


def optimize_per_class_thresholds(
    oof_scores: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple:
    """Find optimal per-class F1 threshold from OOF predictions.

    Returns:
        best_thresholds: (C,) per-class optimal thresholds
        best_scores: (C,) per-class best F1 scores
    """
    if thresholds is None:
        thresholds = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70], dtype=np.float32)

    n_classes = oof_scores.shape[1]
    best_thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    best_scores = np.zeros(n_classes, dtype=np.float32)

    for c in range(n_classes):
        y_c = y_true[:, c]
        scores_c = oof_scores[:, c]

        if y_c.sum() == 0:
            continue

        best_f1 = 0.0
        best_t = 0.5

        for t in thresholds:
            pred_c = (scores_c > t).astype(int)
            tp = ((pred_c == 1) & (y_c == 1)).sum()
            fp = ((pred_c == 1) & (y_c == 0)).sum()
            fn = ((pred_c == 0) & (y_c == 1)).sum()

            if tp + fp == 0 or tp + fn == 0:
                continue

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds[c] = best_t
        best_scores[c] = best_f1

    return best_thresholds, best_scores


def apply_per_class_thresholds(
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Stage 5: Sharpen probabilities around per-class thresholds.

    Above threshold: pushed toward 1.0
    Below threshold: pushed toward 0.0
    When thresholds=0.5, this is an identity transform (no-op in submit mode).
    """
    N, C = scores.shape
    assert C == len(thresholds)

    scaled = np.copy(scores)
    for c in range(C):
        t = thresholds[c]
        mask_above = scores[:, c] > t
        scaled[mask_above, c] = 0.5 + 0.5 * (scores[mask_above, c] - t) / (1 - t + 1e-8)
        scaled[~mask_above, c] = 0.5 * scores[~mask_above, c] / (t + 1e-8)

    return np.clip(scaled, 0, 1)


def run_postprocessing_pipeline(
    final_scores: np.ndarray,
    class_temperatures: np.ndarray,
    per_class_thresholds: np.ndarray | None = None,
    n_windows: int = N_WINDOWS,
    file_level_top_k: int = 2,
    rank_aware_power: float = 0.5,
    delta_shift_alpha: float = 0.15,
) -> np.ndarray:
    """Run the full 6-stage post-processing pipeline.

    Args:
        final_scores: (N, C) logits
        class_temperatures: (C,) per-class temperature values
        per_class_thresholds: (C,) per-class thresholds (default 0.5 = no-op)
        n_windows: windows per file (12)
        file_level_top_k: K for confidence scaling
        rank_aware_power: power for rank-aware scaling
        delta_shift_alpha: alpha for delta shift smoothing

    Returns:
        probs: (N, C) final probabilities clipped to [0, 1]
    """
    # Stage 1: Temperature scaling (logit space) -> probabilities
    probs = temperature_scale(final_scores, class_temperatures)

    # Stage 2: File-level confidence scaling
    if file_level_top_k > 0:
        probs = file_level_confidence_scale(probs, n_windows=n_windows, top_k=file_level_top_k)

    # Stage 3: Rank-aware scaling
    if rank_aware_power > 0:
        probs = rank_aware_scaling(probs, n_windows=n_windows, power=rank_aware_power)

    # Stage 4: Delta shift smoothing
    if delta_shift_alpha > 0:
        probs = delta_shift_smooth(probs, n_windows=n_windows, alpha=delta_shift_alpha)

    # Stage 5: Per-class threshold sharpening
    if per_class_thresholds is not None:
        probs = apply_per_class_thresholds(probs, per_class_thresholds)

    return np.clip(probs, 0, 1)


# ===========================================================================
# Gaussian smoothing + sigmoid (backward compat, kept from v3)
# ===========================================================================


def gauss_smooth_final(scores, weights=np.array([0.1, 0.2, 0.4, 0.2, 0.1])):
    """Gaussian temporal smoothing on ALL species after probes (matching chiranjithdharma)."""
    from scipy.ndimage import convolve1d

    smoothed = scores.reshape(-1, N_WINDOWS, scores.shape[1]).copy()
    for i in range(smoothed.shape[0]):
        smoothed[i] = convolve1d(smoothed[i], weights, axis=0, mode="nearest")
    return smoothed.reshape(-1, scores.shape[1])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ===========================================================================
# Main
# ===========================================================================


def main():
    print("=" * 70)
    print("BirdCLEF 2026 Perch + ProtoSSM v4 (full ProtoSSM v5 pipeline)")
    print("=" * 70)

    # -------------------------------------------------------------------
    # Resolve paths
    # -------------------------------------------------------------------
    competition_dir = find_competition_dir()
    test_dir = competition_dir / "test_soundscapes"
    sample_sub_path = competition_dir / "sample_submission.csv"
    output_path = Path("/kaggle/working/submission.csv")
    if not output_path.parent.exists():
        output_path = Path("predictions/submission.csv")

    labeled_emb_dir = find_labeled_emb_dir()
    artifacts_dir = find_artifacts_dir()
    perch_model_dir = find_perch_model_dir()

    print(f"\nCompetition: {competition_dir}")
    print(f"Labeled embeddings: {labeled_emb_dir}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Perch model: {perch_model_dir}")

    assert labeled_emb_dir is not None, "Labeled embeddings not found!"
    assert artifacts_dir is not None, "Artifacts (species mapping) not found!"
    assert perch_model_dir is not None, "Perch model not found!"

    # -------------------------------------------------------------------
    # Load species mapping
    # -------------------------------------------------------------------
    for name in ["perch_species_mapping.json", "species_mapping.json"]:
        p = artifacts_dir / name
        if p.exists():
            with open(p) as f:
                species_mapping = json.load(f)
            break
    else:
        raise FileNotFoundError("Species mapping not found!")

    birdclef_species = species_mapping["birdclef_species"]
    taxa_classes = species_mapping.get("taxa_classes", {})
    unmapped_set = set(species_mapping.get("unmapped", []))
    print(f"  Species: {len(birdclef_species)}, unmapped: {len(unmapped_set)}")

    # -------------------------------------------------------------------
    # Load ALL soundscape annotations (for richer prior tables)
    # -------------------------------------------------------------------
    print("\n--- Loading Soundscape Annotations ---")
    sc_labels_raw = pd.read_csv(competition_dir / "train_soundscapes_labels.csv")
    sc_labels = sc_labels_raw.drop_duplicates().reset_index(drop=True)

    def _union_labels(series):
        out = set()
        for x in series:
            if pd.notna(x):
                for t in str(x).split(";"):
                    t = t.strip()
                    if t:
                        out.add(t)
        return sorted(out)

    sc_clean = (
        sc_labels.groupby(["filename", "start", "end"])["primary_label"]
        .apply(_union_labels)
        .reset_index(name="label_list")
    )

    # Parse site/hour from filenames
    sc_sites_list, sc_hours_list = [], []
    for fn in sc_clean["filename"]:
        site, hour = _parse_soundscape_metadata(fn)
        sc_sites_list.append(site)
        sc_hours_list.append(hour)
    sc_clean["site"] = sc_sites_list
    sc_clean["hour_utc"] = sc_hours_list

    # Multi-hot label matrix for ALL soundscape rows
    species_to_idx = {s: i for i, s in enumerate(birdclef_species)}
    Y_SC = np.zeros((len(sc_clean), len(birdclef_species)), dtype=np.float32)
    for i, labels in enumerate(sc_clean["label_list"]):
        for lbl in labels:
            if lbl in species_to_idx:
                Y_SC[i, species_to_idx[lbl]] = 1.0

    # Identify fully-labeled files (all 12 windows annotated)
    windows_per_file = sc_clean.groupby("filename").size()
    full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())

    # Build meta_rows format for fit_prior_tables compatibility
    sc_meta_rows = [
        {"site": s, "hour": h} for s, h in zip(sc_sites_list, sc_hours_list, strict=False)
    ]
    sc_sites_arr = np.array(sc_sites_list, dtype=object)

    print(f"  Soundscape rows: {len(sc_clean)}, files: {sc_clean['filename'].nunique()}")
    print(f"  Fully-labeled: {len(full_files)} files")

    # -------------------------------------------------------------------
    # Species category indices (6-way split matching chiranjithdharma)
    # -------------------------------------------------------------------
    direct_map_set = set(species_mapping.get("direct_map", {}).keys())
    proxy_map_set = set(species_mapping.get("genus_proxy_map", {}).keys())
    active_species = {birdclef_species[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]}

    idx_mapped_event = []
    idx_mapped_texture = []
    idx_proxy_texture = []
    idx_prioronly_event = []
    idx_prioronly_texture = []
    idx_unmapped_inactive = []

    for i, sp in enumerate(birdclef_species):
        is_direct = sp in direct_map_set
        is_proxy = sp in proxy_map_set
        is_active = sp in active_species
        is_texture = taxa_classes.get(sp, "Aves") in TEXTURE_TAXA

        if is_direct and is_active and not is_texture:
            idx_mapped_event.append(i)
        elif is_direct and is_active and is_texture:
            idx_mapped_texture.append(i)
        elif is_proxy and is_active and is_texture:
            idx_proxy_texture.append(i)
        elif not is_direct and is_active and not is_texture:
            idx_prioronly_event.append(i)
        elif not is_direct and is_active and is_texture:
            idx_prioronly_texture.append(i)
        elif not is_direct and not is_active:
            idx_unmapped_inactive.append(i)

    species_idx = {
        "mapped_event": np.array(idx_mapped_event, dtype=np.int32),
        "mapped_texture": np.array(idx_mapped_texture, dtype=np.int32),
        "proxy_texture": np.array(idx_proxy_texture, dtype=np.int32),
        "prioronly_event": np.array(idx_prioronly_event, dtype=np.int32),
        "prioronly_texture": np.array(idx_prioronly_texture, dtype=np.int32),
        "unmapped_inactive": np.array(idx_unmapped_inactive, dtype=np.int32),
        "all_active_texture": np.array(
            idx_mapped_texture + idx_proxy_texture + idx_prioronly_texture, dtype=np.int32
        ),
        "all_active_event": np.array(idx_mapped_event + idx_prioronly_event, dtype=np.int32),
    }

    print(
        f"  Categories: mapped_event={len(idx_mapped_event)}, mapped_texture={len(idx_mapped_texture)}, "
        f"proxy_texture={len(idx_proxy_texture)}, prioronly_event={len(idx_prioronly_event)}, "
        f"prioronly_texture={len(idx_prioronly_texture)}, unmapped_inactive={len(idx_unmapped_inactive)}"
    )

    # ===================================================================
    # Phase 1: Load labeled data + OOF probe training + final fit (~5s)
    # ===================================================================
    print("\n--- Phase 1: Load Labeled Data + Train Probes (OOF) ---")
    t1 = time.time()

    # Load pre-extracted labeled embeddings (RAW LOGITS + embeddings)
    npz = np.load(labeled_emb_dir / "labeled_perch_arrays.npz")
    labeled_scores_raw = npz["scores_raw"]  # (792, 234) RAW LOGITS
    labeled_emb = npz["emb"]  # (792, 1536)
    labeled_meta_df = pd.read_parquet(labeled_emb_dir / "labeled_perch_meta.parquet")

    print(
        f"  Labeled: scores_raw {labeled_scores_raw.shape} (range [{labeled_scores_raw.min():.1f}, {labeled_scores_raw.max():.1f}])"
    )
    print(f"  Labeled: emb {labeled_emb.shape}")

    # Filter to fully-labeled files only (matching chiranjithdharma)
    full_file_mask = labeled_meta_df["filename"].isin(full_files).values
    labeled_scores_raw = labeled_scores_raw[full_file_mask]
    labeled_emb = labeled_emb[full_file_mask]
    labeled_meta_df = labeled_meta_df[full_file_mask].reset_index(drop=True)
    print(
        f"  Filtered to fully-labeled: {labeled_scores_raw.shape[0]} windows "
        f"from {labeled_meta_df['filename'].nunique()} files"
    )

    # Build meta_rows
    meta_rows = []
    for _, row in labeled_meta_df.iterrows():
        meta_rows.append(
            {
                "filename": row["filename"],
                "window_idx": int(row["window_idx"]),
                "window_start": float(row["window_start"]),
                "window_end": float(row["window_end"]),
                "site": row.get("site"),
                "hour": int(row["hour"]) if pd.notna(row.get("hour")) else None,
            }
        )

    # Build labels
    labels_df = pd.read_csv(competition_dir / "train_soundscapes_labels.csv")
    labels_matrix = build_labels_matrix(meta_rows, labels_df, birdclef_species)
    n_positive = labels_matrix.sum(axis=0)
    print(f"  Labels: {labels_matrix.sum():.0f} positives, {(n_positive > 0).sum()} species")

    # -------------------------------------------------------------------
    # Phase 1a: Global StandardScaler + PCA (fit once on all data)
    # -------------------------------------------------------------------
    sites_arr = np.array([r["site"] for r in meta_rows])
    hours_arr = np.array([r["hour"] for r in meta_rows])

    emb_scaler = StandardScaler()
    emb_scaled = emb_scaler.fit_transform(labeled_emb)
    pca = PCA(n_components=min(PCA_DIM, emb_scaled.shape[0] - 1, emb_scaled.shape[1]))
    Z_FULL = pca.fit_transform(emb_scaled).astype(np.float32)
    print(
        f"  PCA: {labeled_emb.shape[1]} -> {pca.n_components_} dims ({pca.explained_variance_ratio_.sum():.3f} var)"
    )

    # -------------------------------------------------------------------
    # Phase 1b: OOF fold-safe priors (fills oof_base and oof_prior)
    # -------------------------------------------------------------------
    # Use filename grouping for OOF (not site)
    groups = labeled_meta_df["filename"].factorize()[0]

    unique_groups = np.unique(groups)
    n_splits = min(OOF_N_SPLITS, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)
    probe_candidate_idx = np.where(n_positive >= PROBE_MIN_POS)[0]

    # These arrays collect fold-safe scores -- used as features for final probes
    oof_base = np.zeros_like(labeled_scores_raw, dtype=np.float32)
    oof_prior = np.zeros_like(labeled_scores_raw, dtype=np.float32)

    print(f"\n  OOF: {n_splits}-fold GroupKFold by filename ({len(unique_groups)} groups)")

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(labeled_emb, groups=groups)):
        val_idx = np.sort(val_idx)
        fold_sites_val = {meta_rows[i]["site"] for i in val_idx}

        # Fold-safe priors from ALL soundscape annotations (minus val sites)
        prior_mask = ~np.isin(sc_sites_arr, list(fold_sites_val))
        fold_sc_meta = [sc_meta_rows[i] for i in np.where(prior_mask)[0]]
        fold_prior_tables = fit_prior_tables(Y_SC[prior_mask], fold_sc_meta, birdclef_species)

        # Fuse val scores with fold-safe priors -> fills oof_base, oof_prior
        fold_val_base, fold_val_prior = fuse_scores_with_priors(
            labeled_scores_raw[val_idx],
            sites_arr[val_idx],
            hours_arr[val_idx],
            fold_prior_tables,
            species_idx,
        )
        oof_base[val_idx] = fold_val_base
        oof_prior[val_idx] = fold_val_prior

        print(
            f"    Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}, "
            f"val_sites={fold_sites_val}"
        )

    # OOF evaluation (AUROC on fold-safe base scores -- before probes)
    oof_probs = sigmoid(oof_base)
    species_with_labels = np.where(n_positive > 0)[0]
    per_species_auroc = []
    for cls_idx in species_with_labels:
        y_true = labels_matrix[:, cls_idx]
        y_pred = oof_probs[:, cls_idx]
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            pos = y_pred[y_true == 1]
            neg = y_pred[y_true == 0]
            auroc = np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(
                pos[:, None] == neg[None, :]
            )
            per_species_auroc.append(auroc)
    oof_mean_auroc = np.mean(per_species_auroc) if per_species_auroc else 0.0
    print(
        f"\n  OOF baseline AUROC (prior fusion only): {oof_mean_auroc:.4f} "
        f"(mean over {len(per_species_auroc)} species)"
    )

    # -------------------------------------------------------------------
    # Phase 1c: Train final probes on ALL data with OOF features
    # -------------------------------------------------------------------
    print("\n  --- Training final probes (OOF features) ---")

    # All-data prior tables from ALL soundscape annotations (used at test time)
    prior_tables = fit_prior_tables(Y_SC, sc_meta_rows, birdclef_species)

    probe_models = {}

    for cls_idx in probe_candidate_idx:
        y = labels_matrix[:, cls_idx]
        if y.sum() < PROBE_MIN_POS or y.sum() == len(y):
            continue

        # Features use OOF base/prior (fold-safe), not all-data base/prior
        X = build_class_features(
            Z_FULL,
            raw_col=labeled_scores_raw[:, cls_idx],
            prior_col=oof_prior[:, cls_idx],
            base_col=oof_base[:, cls_idx],
        )

        # Positive oversampling (MLPClassifier doesn't support class_weight)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos > 0 and n_neg > n_pos:
            repeat = max(1, n_neg // n_pos)
            pos_idx = np.where(y == 1)[0]
            X_bal = np.vstack([X, np.tile(X[pos_idx], (repeat, 1))])
            y_bal = np.concatenate([y, np.ones(len(pos_idx) * repeat)])
        else:
            X_bal, y_bal = X, y

        clf = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN,
            activation=MLP_ACTIVATION,
            max_iter=MLP_MAX_ITER,
            early_stopping=MLP_EARLY_STOPPING,
            validation_fraction=MLP_VALIDATION_FRACTION,
            learning_rate_init=MLP_LR_INIT,
            alpha=MLP_ALPHA,
        )
        clf.fit(X_bal, y_bal)
        probe_models[cls_idx] = clf

    print(
        f"  Final probes: {len(probe_models)} MLP trained, {len(birdclef_species) - len(probe_models)} skipped"
    )
    print(f"  Phase 1: {time.time() - t1:.1f}s")

    # ===================================================================
    # Phase 2: ProtoSSM Training (~30s)
    # ===================================================================
    print("\n--- Phase 2: ProtoSSM Training ---")
    t2 = time.time()

    # 2a. Reshape to file-level
    emb_files, file_list = reshape_to_files(labeled_emb, labeled_meta_df)
    logits_files, _ = reshape_to_files(labeled_scores_raw, labeled_meta_df)
    labels_files, _ = reshape_to_files(labels_matrix, labeled_meta_df)

    # 2b. Build site mapping, taxonomy groups, file metadata
    site_to_idx, n_sites = build_site_mapping(labeled_meta_df, n_sites_max=PROTO_SSM_N_SITES)
    site_ids_all, hours_all = get_file_metadata(
        labeled_meta_df, file_list, site_to_idx, PROTO_SSM_N_SITES
    )

    # Build taxonomy groups using the taxonomy.csv from competition data
    taxonomy_df = (
        pd.read_csv(competition_dir / "taxonomy.csv")
        if (competition_dir / "taxonomy.csv").exists()
        else pd.DataFrame()
    )
    n_families, class_to_family, _ = build_taxonomy_groups(taxonomy_df, birdclef_species)
    file_families = build_file_families(labels_files, class_to_family, n_families)

    # Use site from filename for ProtoSSM OOF groups
    file_sites = np.array(
        [site_to_idx.get(str(_parse_soundscape_metadata(f)[0]), 0) for f in file_list]
    )

    print(f"  Files: {len(file_list)}, emb: {emb_files.shape}, logits: {logits_files.shape}")
    print(f"  Sites: {n_sites}, families: {n_families}")

    # 2c. ProtoSSM OOF (3-fold by site)
    ssm_cfg = {
        "d_model": PROTO_SSM_D_MODEL,
        "d_state": PROTO_SSM_D_STATE,
        "n_ssm_layers": PROTO_SSM_N_LAYERS,
        "dropout": PROTO_SSM_DROPOUT,
        "n_sites": PROTO_SSM_N_SITES,
        "meta_dim": PROTO_SSM_META_DIM,
        "use_cross_attn": PROTO_SSM_USE_CROSS_ATTN,
        "cross_attn_heads": PROTO_SSM_CROSS_ATTN_HEADS,
    }
    train_cfg = {
        "n_epochs": PROTO_SSM_N_EPOCHS,
        "lr": PROTO_SSM_LR,
        "weight_decay": PROTO_SSM_WEIGHT_DECAY,
        "patience": PROTO_SSM_PATIENCE,
        "pos_weight_cap": PROTO_SSM_POS_WEIGHT_CAP,
        "distill_weight": PROTO_SSM_DISTILL_WEIGHT,
        "label_smoothing": PROTO_SSM_LABEL_SMOOTHING,
        "oof_n_splits": PROTO_SSM_OOF_N_SPLITS,
        "mixup_alpha": PROTO_SSM_MIXUP_ALPHA,
        "focal_gamma": PROTO_SSM_FOCAL_GAMMA,
        "swa_start_frac": PROTO_SSM_SWA_START_FRAC,
    }

    oof_proto_preds, fold_histories, fold_alphas = run_proto_ssm_oof(
        emb_files,
        logits_files,
        labels_files,
        site_ids_all,
        hours_all,
        file_families,
        file_sites,
        n_families,
        class_to_family,
        ssm_cfg=ssm_cfg,
        train_cfg=train_cfg,
        tta_shifts=TTA_SHIFTS,
    )

    # 2d. Compute OOF MLP scores for ensemble weight optimization
    # The MLP probes were already trained in Phase 1
    # For OOF MLP scores, we use the oof_base (fold-safe) + probe predictions
    oof_mlp_flat = oof_base.copy()  # start with fold-safe base
    for cls_idx, clf in probe_models.items():
        X = build_class_features(
            Z_FULL,
            raw_col=labeled_scores_raw[:, cls_idx],
            prior_col=oof_prior[:, cls_idx],
            base_col=oof_base[:, cls_idx],
        )
        pred_prob = clf.predict_proba(X)[:, 1].astype(np.float32)
        pred_logit = np.log(pred_prob + 1e-7) - np.log(1.0 - pred_prob + 1e-7)
        oof_mlp_flat[:, cls_idx] = (1.0 - PROBE_ALPHA) * oof_base[
            :, cls_idx
        ] + PROBE_ALPHA * pred_logit

    # Flatten OOF proto predictions for ensemble optimization
    oof_proto_flat = oof_proto_preds.reshape(-1, N_SPECIES)
    y_true_flat = labels_matrix

    best_w, best_auc, weight_results = optimize_ensemble_weight(
        oof_proto_flat, oof_mlp_flat, y_true_flat
    )
    ENSEMBLE_WEIGHT_PROTO = best_w
    print(f"\n  Ensemble weight: ProtoSSM={best_w:.2f}, MLP={1-best_w:.2f}, AUC={best_auc:.4f}")

    # 2e. Train final ProtoSSM on ALL data
    final_model = ProtoSSMv2(
        d_input=1536,
        d_model=PROTO_SSM_D_MODEL,
        d_state=PROTO_SSM_D_STATE,
        n_ssm_layers=PROTO_SSM_N_LAYERS,
        n_classes=N_SPECIES,
        n_windows=N_WINDOWS,
        dropout=PROTO_SSM_DROPOUT,
        n_sites=PROTO_SSM_N_SITES,
        meta_dim=PROTO_SSM_META_DIM,
        use_cross_attn=PROTO_SSM_USE_CROSS_ATTN,
        cross_attn_heads=PROTO_SSM_CROSS_ATTN_HEADS,
    )
    emb_flat_all = emb_files.reshape(-1, 1536)
    labels_flat_all = labels_files.reshape(-1, N_SPECIES)
    final_model.init_prototypes_from_data(
        torch.tensor(emb_flat_all, dtype=torch.float32),
        torch.tensor(labels_flat_all, dtype=torch.float32),
    )
    final_model.init_family_head(n_families, class_to_family)

    # For final training, use 15% val split
    n_val = max(1, int(len(file_list) * 0.15))
    perm_idx = torch.randperm(len(file_list), generator=torch.Generator().manual_seed(42))
    val_file_idx = perm_idx[:n_val].numpy()
    train_file_idx = perm_idx[n_val:].numpy()

    final_model, final_hist = train_proto_ssm_single(
        final_model,
        emb_files[train_file_idx],
        logits_files[train_file_idx],
        labels_files[train_file_idx].astype(np.float32),
        site_ids_train=site_ids_all[train_file_idx],
        hours_train=hours_all[train_file_idx],
        emb_val=emb_files[val_file_idx],
        logits_val=logits_files[val_file_idx],
        labels_val=labels_files[val_file_idx].astype(np.float32),
        site_ids_val=site_ids_all[val_file_idx],
        hours_val=hours_all[val_file_idx],
        file_families_train=file_families[train_file_idx],
        file_families_val=file_families[val_file_idx],
        cfg=train_cfg,
    )

    # 2f. Train ResidualSSM
    # First compute first-pass ensemble scores on training data
    final_model.eval()
    with torch.no_grad():
        proto_train_out, _, _ = final_model(
            torch.tensor(emb_files, dtype=torch.float32),
            torch.tensor(logits_files, dtype=torch.float32),
            site_ids=torch.tensor(site_ids_all, dtype=torch.long),
            hours=torch.tensor(hours_all, dtype=torch.long),
        )
        proto_train_scores = proto_train_out.numpy()

    # Reshape oof_mlp_flat to file-level for ensemble
    mlp_train_files, _ = reshape_to_files(oof_mlp_flat, labeled_meta_df)

    first_pass_files = fuse_protossm_mlp(proto_train_scores, mlp_train_files, ENSEMBLE_WEIGHT_PROTO)

    res_model, correction_weight = train_residual_ssm(
        emb_files,
        first_pass_files,
        labels_files,
        site_ids_all,
        hours_all,
        cfg={
            "d_model": RESIDUAL_SSM_D_MODEL,
            "d_state": RESIDUAL_SSM_D_STATE,
            "dropout": RESIDUAL_SSM_DROPOUT,
            "correction_weight": RESIDUAL_SSM_CORRECTION_WEIGHT,
            "n_epochs": RESIDUAL_SSM_N_EPOCHS,
            "lr": RESIDUAL_SSM_LR,
            "patience": RESIDUAL_SSM_PATIENCE,
        },
        n_sites=PROTO_SSM_N_SITES,
        wall_time_limit=4.0,
        start_time=START_TIME,
    )

    print(f"  Phase 2: {time.time() - t2:.1f}s")

    # ===================================================================
    # Phase 3: Perch Inference on Test Soundscapes (~80 min)
    # ===================================================================
    print("\n--- Phase 3: Perch on Test Soundscapes ---")
    t3 = time.time()

    print("  Loading Perch model...")
    _, infer_fn = load_perch_model(str(perch_model_dir))
    dummy_emb, dummy_logits = run_perch_inference(
        infer_fn, np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
    )
    print(f"  Perch: emb={dummy_emb.shape}, logits={dummy_logits.shape}")

    audio_files = sorted(test_dir.glob("*.ogg"))
    if not audio_files:
        train_dir = competition_dir / "train_soundscapes"
        if train_dir.exists():
            audio_files = sorted(train_dir.glob("*.ogg"))[:3]
            print(f"  No test files, using {len(audio_files)} train files")
    print(f"  Processing {len(audio_files)} soundscapes")

    # Batched inference: process BATCH_FILES files per TF call (matching Yuriy)
    # This is ~10x faster than file-by-file because TF amortizes session overhead.
    BATCH_FILES = 16
    n_files_total = len(audio_files)

    test_emb_list = []
    test_scores_list = []
    test_meta = []
    n_timeout = 0

    for batch_start in range(0, n_files_total, BATCH_FILES):
        if time.time() - START_TIME > TIMEOUT_SECONDS:
            n_timeout += n_files_total - batch_start
            break

        batch_paths = audio_files[batch_start : batch_start + BATCH_FILES]
        batch_n = len(batch_paths)

        # Read all files in batch into a single waveform array
        waveforms = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        batch_meta = []
        for i, audio_path in enumerate(batch_paths):
            waveform = load_and_window_audio(audio_path)  # (12, WINDOW_SAMPLES)
            waveforms[i * N_WINDOWS : (i + 1) * N_WINDOWS] = waveform

            site, hour = _parse_soundscape_metadata(str(audio_path))
            sid = audio_path.stem
            for w in range(N_WINDOWS):
                batch_meta.append(
                    {
                        "soundscape_id": sid,
                        "end_time": (w + 1) * WINDOW_SECONDS,
                        "site": site,
                        "hour": hour,
                    }
                )

        # Single TF call for entire batch
        emb, logits = run_perch_inference(infer_fn, waveforms)
        mapped = compute_perch_mapped_logits(logits, species_mapping)

        test_emb_list.append(emb)
        test_scores_list.append(mapped)
        test_meta.extend(batch_meta)

        n_done = batch_start + batch_n
        if n_done % (BATCH_FILES * 3) == 0 or n_done == batch_n or n_done == n_files_total:
            elapsed = time.time() - START_TIME
            print(f"  [{n_done}/{n_files_total}] {elapsed / 60:.1f}min elapsed")

    if n_timeout:
        print(f"  *** {n_timeout} files timed out ***")

    test_emb_all = (
        np.concatenate(test_emb_list, axis=0)
        if test_emb_list
        else np.zeros((0, 1536), dtype=np.float32)
    )
    test_scores_raw = (
        np.concatenate(test_scores_list, axis=0)
        if test_scores_list
        else np.zeros((0, N_SPECIES), dtype=np.float32)
    )
    print(f"  Test: emb {test_emb_all.shape}, scores_raw {test_scores_raw.shape}")
    print(f"  Phase 3: {(time.time() - t3) / 60:.1f}min")

    # ===================================================================
    # Phase 4: Score Fusion + Post-Processing (~30s)
    # ===================================================================
    print("\n--- Phase 4: Score Fusion + Post-Processing ---")
    t4 = time.time()

    # PCA transform test embeddings
    Z_TEST = pca.transform(emb_scaler.transform(test_emb_all)).astype(np.float32)

    # Fuse test scores with priors (with event smoothing)
    test_sites = np.array([r["site"] for r in test_meta])
    test_hours = np.array([r["hour"] for r in test_meta])
    test_base, test_prior = fuse_scores_with_priors(
        test_scores_raw,
        test_sites,
        test_hours,
        prior_tables,
        species_idx,
    )

    # Apply MLP probes (existing logic)
    mlp_test_scores = test_base.copy()
    for cls_idx, clf in probe_models.items():
        X = build_class_features(
            Z_TEST,
            raw_col=test_scores_raw[:, cls_idx],
            prior_col=test_prior[:, cls_idx],
            base_col=test_base[:, cls_idx],
        )
        pred_prob = clf.predict_proba(X)[:, 1].astype(np.float32)
        pred_logit = np.log(pred_prob + 1e-7) - np.log(1.0 - pred_prob + 1e-7)
        mlp_test_scores[:, cls_idx] = (1.0 - PROBE_ALPHA) * test_base[
            :, cls_idx
        ] + PROBE_ALPHA * pred_logit

    # ProtoSSM inference with TTA
    # Need file-level data for ProtoSSM
    test_meta_df = pd.DataFrame({"filename": [r["soundscape_id"] for r in test_meta]})
    test_emb_files, test_file_list = reshape_to_files(test_emb_all, test_meta_df)
    test_logits_files, _ = reshape_to_files(test_scores_raw, test_meta_df)

    # Get test site/hour metadata per file
    test_site_to_idx = site_to_idx  # reuse from training
    test_site_ids = np.zeros(len(test_file_list), dtype=np.int64)
    test_hour_ids = np.zeros(len(test_file_list), dtype=np.int64)
    # Build from test_meta: one entry per file
    file_meta_map = {}
    for r in test_meta:
        sid = r["soundscape_id"]
        if sid not in file_meta_map:
            file_meta_map[sid] = r
    for fi, fname in enumerate(test_file_list):
        r = file_meta_map.get(fname, {})
        s = r.get("site")
        h = r.get("hour")
        test_site_ids[fi] = (
            min(test_site_to_idx.get(str(s), 0), PROTO_SSM_N_SITES - 1) if s is not None else 0
        )
        test_hour_ids[fi] = int(h) % 24 if h is not None else 0

    # Run ProtoSSM with TTA
    proto_test_preds = temporal_shift_tta(
        test_emb_files,
        test_logits_files,
        final_model,
        test_site_ids,
        test_hour_ids,
        shifts=TTA_SHIFTS,
    )
    proto_test_flat = proto_test_preds.reshape(-1, N_SPECIES)

    # Reshape MLP scores to file-level
    mlp_test_files, _ = reshape_to_files(mlp_test_scores, test_meta_df)

    # Ensemble fusion
    final_scores = fuse_protossm_mlp(proto_test_flat, mlp_test_scores, ENSEMBLE_WEIGHT_PROTO)

    # ResidualSSM correction
    if res_model is not None and correction_weight > 0:
        res_model.eval()
        with torch.no_grad():
            test_first_pass_files = fuse_protossm_mlp(
                proto_test_preds, mlp_test_files, ENSEMBLE_WEIGHT_PROTO
            )
            test_correction = res_model(
                torch.tensor(test_emb_files, dtype=torch.float32),
                torch.tensor(test_first_pass_files, dtype=torch.float32),
                site_ids=torch.tensor(test_site_ids, dtype=torch.long),
                hours=torch.tensor(test_hour_ids, dtype=torch.long),
            ).numpy()
        final_scores = final_scores + correction_weight * test_correction.reshape(-1, N_SPECIES)
        print(f"  ResidualSSM correction applied (weight={correction_weight})")

    # Build class temperatures
    class_temperatures = np.ones(N_SPECIES, dtype=np.float32) * TEMPERATURE_AVES
    for i, sp in enumerate(birdclef_species):
        if taxa_classes.get(sp, "Aves") in TEXTURE_TAXA:
            class_temperatures[i] = TEMPERATURE_TEXTURE

    # Per-class thresholds: default 0.5 in submit mode (no OOF optimization -> identity)
    per_class_thresholds = np.full(N_SPECIES, 0.5, dtype=np.float32)

    # 6-stage post-processing pipeline (replaces Gaussian smoothing)
    final_probs = run_postprocessing_pipeline(
        final_scores,
        class_temperatures,
        per_class_thresholds,
        file_level_top_k=FILE_LEVEL_TOP_K,
        rank_aware_power=RANK_AWARE_POWER,
        delta_shift_alpha=DELTA_SHIFT_ALPHA,
    )

    print(f"  Score range (logits): [{final_scores.min():.2f}, {final_scores.max():.2f}]")
    print(f"  Prob range: [{final_probs.min():.4f}, {final_probs.max():.4f}]")
    print(f"  Phase 4: {time.time() - t4:.1f}s")

    # -------------------------------------------------------------------
    # Build submission
    # -------------------------------------------------------------------
    all_predictions = []
    for i, row in enumerate(test_meta):
        if i >= final_probs.shape[0]:
            break
        pred_row = {"row_id": f"{row['soundscape_id']}_{row['end_time']}"}
        for sp_idx, species in enumerate(birdclef_species):
            pred_row[species] = float(final_probs[i, sp_idx])
        all_predictions.append(pred_row)

    if sample_sub_path.exists():
        sample_sub = pd.read_csv(sample_sub_path)
        expected_columns = list(sample_sub.columns)
        species_columns = [c for c in expected_columns if c != "row_id"]
        if all_predictions:
            pred_df = pd.DataFrame(all_predictions)
            submission = sample_sub[["row_id"]].merge(pred_df, on="row_id", how="left")
            for col in species_columns:
                if col not in submission.columns:
                    submission[col] = 0.0
                else:
                    submission[col] = submission[col].fillna(0.0)
            submission = submission[expected_columns]
        else:
            submission = sample_sub.copy()
            for col in species_columns:
                submission[col] = 0.0
    else:
        submission = pd.DataFrame(all_predictions) if all_predictions else pd.DataFrame()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    total = time.time() - START_TIME
    print(f"\nSubmission: {output_path} ({len(submission)} rows)")
    print(f"Total: {total:.1f}s ({total / 60:.1f}min)")


if __name__ == "__main__":
    main()
