#!/usr/bin/env python3
"""BirdCLEF 2026 Perch + Probes v3 Submission (OOF logit-space pipeline).

Matches chiranjithdharma (0.910) pipeline exactly (verified line-by-line):
  1. Load ALL soundscape annotations for prior tables (richer priors)
  2. Filter labeled embeddings to 59 fully-labeled files (clean labels)
  3. StandardScaler + PCA fit ONCE on all data (global)
  4. 5-fold GroupKFold by site: fold-safe priors from ALL annotations
  5. Train final probes on all data using OOF features (oof_base, oof_prior)
  6. Run Perch v2 live on test soundscapes → raw logits
  7. 6-category fuse_scores (proxy, prior-only replacement, -8.0 inactive)
  8. Gaussian smoothing on all species + sigmoid at submission

Key v3 fixes vs v2 (0.898):
  - K_HOUR fixed: 4 → 8 (matching chiranjithdharma)
  - Prior tables from ALL soundscape annotations, not just labeled subset
  - 6-category fuse_scores: proxy λ=0.8, prior-only replacement, -8.0 inactive
  - Filtered to 59 fully-labeled files (removed 7 noisy partial files)

CRITICAL: Everything stays in LOGIT SPACE until sigmoid at submission.
"""

# ---------------------------------------------------------------------------
# Offline dependency installation
# ---------------------------------------------------------------------------
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
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

# Probe hyperparameters (matching chiranjithdharma)
PCA_DIM = 32
PROBE_MIN_POS = 8
PROBE_C = 0.25
PROBE_ALPHA_BASE = 0.25
PROBE_ALPHA_SCALE = 0.25
PROBE_ALPHA_THRESHOLD = 40

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
    """Returns (embeddings, logits) — logits are RAW, no sigmoid."""
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

    # Mapped active event: additive prior (λ=0.4)
    if len(idx["mapped_event"]):
        scores[:, idx["mapped_event"]] += LAMBDA_EVENT * prior[:, idx["mapped_event"]]
    # Mapped active texture: additive prior (λ=1.0)
    if len(idx["mapped_texture"]):
        scores[:, idx["mapped_texture"]] += LAMBDA_TEXTURE * prior[:, idx["mapped_texture"]]
    # Proxy active texture: additive prior (λ=0.8)
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

    # Temporal smoothing in logit space (all active texture — matching chiranjithdharma)
    scores = smooth_texture_cols(scores, idx["all_active_texture"].tolist(), SMOOTH_TEXTURE)

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
    return prev_v, next_v, mean_v, max_v


def build_class_features(Z, raw_col, prior_col, base_col):
    """Build 39-dim feature vector per class (matching chiranjithdharma)."""
    prev_b, next_b, mean_b, max_b = seq_features_1d(base_col)
    return np.concatenate(
        [
            Z,  # PCA embeddings (32)
            raw_col[:, None],  # raw Perch logit (1)
            prior_col[:, None],  # prior logit (1)
            base_col[:, None],  # fused base logit (1)
            prev_b[:, None],  # prev window base (1)
            next_b[:, None],  # next window base (1)
            mean_b[:, None],  # file mean base (1)
            max_b[:, None],  # file max base (1)
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
# Main
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


def main():
    print("=" * 70)
    print("BirdCLEF 2026 Perch + Probes v3 (OOF logit-space pipeline)")
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
    }

    print(
        f"  Categories: mapped_event={len(idx_mapped_event)}, mapped_texture={len(idx_mapped_texture)}, "
        f"proxy_texture={len(idx_proxy_texture)}, prioronly_event={len(idx_prioronly_event)}, "
        f"prioronly_texture={len(idx_prioronly_texture)}, unmapped_inactive={len(idx_unmapped_inactive)}"
    )

    # -------------------------------------------------------------------
    # Phase 1: Load labeled data + OOF probe training + final fit
    # -------------------------------------------------------------------
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
        f"  PCA: {labeled_emb.shape[1]} → {pca.n_components_} dims ({pca.explained_variance_ratio_.sum():.3f} var)"
    )

    # -------------------------------------------------------------------
    # Phase 1b: OOF fold-safe priors (fills oof_base and oof_prior)
    # -------------------------------------------------------------------
    # Key insight from chiranjithdharma source: PCA is global, but priors
    # are fold-safe. The OOF loop fills oof_base/oof_prior arrays which
    # are then used as FEATURES for the final probe training. This prevents
    # probes from learning to exploit leaked prior signal.

    unique_sites = sorted({s for s in sites_arr if s is not None})
    site_to_group = {s: i for i, s in enumerate(unique_sites)}
    groups = np.array([site_to_group.get(s, 0) for s in sites_arr])

    n_splits = min(OOF_N_SPLITS, len(unique_sites))
    gkf = GroupKFold(n_splits=n_splits)
    probe_candidate_idx = np.where(n_positive >= PROBE_MIN_POS)[0]

    # These arrays collect fold-safe scores — used as features for final probes
    oof_base = np.zeros_like(labeled_scores_raw, dtype=np.float32)
    oof_prior = np.zeros_like(labeled_scores_raw, dtype=np.float32)

    print(f"\n  OOF: {n_splits}-fold GroupKFold by site ({len(unique_sites)} sites)")

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(labeled_emb, groups=groups)):
        val_idx = np.sort(val_idx)
        fold_sites_val = {meta_rows[i]["site"] for i in val_idx}

        # Fold-safe priors from ALL soundscape annotations (minus val sites)
        prior_mask = ~np.isin(sc_sites_arr, list(fold_sites_val))
        fold_sc_meta = [sc_meta_rows[i] for i in np.where(prior_mask)[0]]
        fold_prior_tables = fit_prior_tables(Y_SC[prior_mask], fold_sc_meta, birdclef_species)

        # Fuse val scores with fold-safe priors → fills oof_base, oof_prior
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

    # OOF evaluation (AUROC on fold-safe base scores — before probes)
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
    # CRITICAL: probes see oof_base/oof_prior as features (fold-safe),
    # NOT all-data priors. This is what chiranjithdharma does.
    print("\n  --- Training final probes (OOF features) ---")

    # All-data prior tables from ALL soundscape annotations (used at test time)
    prior_tables = fit_prior_tables(Y_SC, sc_meta_rows, birdclef_species)

    probe_models = {}
    probe_alphas = {}

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

        clf = LogisticRegression(
            C=PROBE_C, max_iter=1000, solver="liblinear", class_weight="balanced"
        )
        clf.fit(X, y)
        probe_models[cls_idx] = clf

        n_pos = int(n_positive[cls_idx])
        alpha = PROBE_ALPHA_BASE + PROBE_ALPHA_SCALE * min(
            1.0, (n_pos - PROBE_MIN_POS) / PROBE_ALPHA_THRESHOLD
        )
        probe_alphas[cls_idx] = alpha

    print(
        f"  Final probes: {len(probe_models)} trained, {len(birdclef_species) - len(probe_models)} skipped"
    )
    print(f"  Phase 1: {time.time() - t1:.1f}s")

    # -------------------------------------------------------------------
    # Phase 2: Perch inference on test soundscapes
    # -------------------------------------------------------------------
    print("\n--- Phase 2: Perch on Test Soundscapes ---")
    t2 = time.time()

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

    def process_file(args):
        file_idx, audio_path = args
        if time.time() - START_TIME > TIMEOUT_SECONDS:
            return (file_idx, audio_path, None, None, True)
        waveform = load_and_window_audio(audio_path)
        emb, logits = run_perch_inference(infer_fn, waveform)
        mapped = compute_perch_mapped_logits(logits, species_mapping)
        return file_idx, audio_path, emb, mapped, False

    test_emb_list = []
    test_scores_list = []
    test_meta = []
    n_timeout = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(process_file, enumerate(audio_files)):
            file_idx, audio_path, emb, mapped, timed_out = result
            if timed_out:
                n_timeout += 1
                continue

            test_emb_list.append(emb)
            test_scores_list.append(mapped)

            site, hour = _parse_soundscape_metadata(str(audio_path))
            sid = audio_path.stem
            for w in range(N_WINDOWS):
                test_meta.append(
                    {
                        "soundscape_id": sid,
                        "end_time": (w + 1) * WINDOW_SECONDS,
                        "site": site,
                        "hour": hour,
                    }
                )

            n_done = len(test_emb_list)
            if n_done % 50 == 0 or n_done == 1 or n_done == len(audio_files):
                elapsed = time.time() - START_TIME
                print(f"  [{n_done}/{len(audio_files)}] {elapsed / 60:.1f}min elapsed")

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
    print(f"  Phase 2: {(time.time() - t2) / 60:.1f}min")

    # -------------------------------------------------------------------
    # Phase 3: Apply probes in logit space
    # -------------------------------------------------------------------
    print("\n--- Phase 3: Apply Probes ---")
    t3 = time.time()

    # PCA transform test embeddings (using scaler + PCA fit on labeled)
    Z_TEST = pca.transform(emb_scaler.transform(test_emb_all)).astype(np.float32)

    # Fuse test scores with priors (logit space)
    test_sites = np.array([r["site"] for r in test_meta])
    test_hours = np.array([r["hour"] for r in test_meta])
    test_base, test_prior = fuse_scores_with_priors(
        test_scores_raw,
        test_sites,
        test_hours,
        prior_tables,
        species_idx,
    )

    # Apply probes (logit blending)
    final_scores = test_base.copy()
    for cls_idx, clf in probe_models.items():
        X = build_class_features(
            Z_TEST,
            raw_col=test_scores_raw[:, cls_idx],
            prior_col=test_prior[:, cls_idx],
            base_col=test_base[:, cls_idx],
        )
        pred = clf.decision_function(X).astype(np.float32)  # LOGIT output
        alpha = probe_alphas[cls_idx]
        final_scores[:, cls_idx] = (1.0 - alpha) * test_base[:, cls_idx] + alpha * pred

    # Gaussian final smoothing on ALL species (matching chiranjithdharma)
    final_scores = gauss_smooth_final(final_scores)

    # Sigmoid ONLY at submission
    final_probs = sigmoid(final_scores)

    print(f"  Score range (logits): [{final_scores.min():.2f}, {final_scores.max():.2f}]")
    print(f"  Prob range: [{final_probs.min():.4f}, {final_probs.max():.4f}]")
    print(f"  Phase 3: {time.time() - t3:.1f}s")

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
