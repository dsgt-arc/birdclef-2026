# ruff: noqa
# Exp 023: Perch ONNX + SED v5j-4fold-pp2 Ensemble
# Combines two orthogonal pipelines:
#   Pipeline 1: Perch ONNX → ProtoSSMv2 + MLP probes → 6-stage PP
#   Pipeline 2: HGNet-B0 SED 4-fold → taxa-aware PP
# Final: weighted average W_PERCH * perch + (1-W_PERCH) * sed

# ============================================================================
# SECTION 0: Dependency Install
# ============================================================================
import os
import subprocess
import sys

KAGGLE_USER = "murilogustineli"


def _find_input(slug):
    """Resolve Kaggle input path, supporting both old and new directory layouts."""
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]  # fallback to old-style


DEPS_DIR = _find_input("birdclef-deps")

if os.path.exists("/kaggle/input"):
    print(f"Kaggle input contents: {os.listdir('/kaggle/input')}")
print(f"DEPS_DIR resolved to: {DEPS_DIR} (exists={os.path.exists(DEPS_DIR)})")

if os.path.exists(DEPS_DIR):
    print(f"Installing onnxruntime from {DEPS_DIR}")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            DEPS_DIR,
            "onnxruntime",
        ]
    )
else:
    print(f"WARNING: deps dir not found at {DEPS_DIR}, onnxruntime may not be available")

# ============================================================================
# SECTION 1: Imports + Constants
# ============================================================================
import gc  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

warnings.filterwarnings("ignore")

# --- Constants ---
W_PERCH = 0.5
SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC  # 160000
FILE_SAMPLES = 60 * SR  # 1920000
N_WINDOWS = 12
N_CLASSES = 234  # will be updated from sample_submission
DEVICE = torch.device("cpu")

# --- Path resolution ---


def find_competition_dir():
    for c in ["/kaggle/input/competitions/birdclef-2026", "/kaggle/input/birdclef-2026"]:
        if os.path.exists(c):
            return Path(c)
    return Path("data/birdclef-2026")


BASE = find_competition_dir()

print(f"Competition dir: {BASE} (exists={BASE.exists()})")
print(f"PyTorch: {torch.__version__}")

# ============================================================================
# SECTION 2: Shared Utilities + Data Loading
# ============================================================================

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {
            "file_id": None,
            "site": None,
            "date": pd.NaT,
            "time_utc": None,
            "hour_utc": -1,
            "month": -1,
        }
    file_id, site, ymd, hms = m.groups()
    dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    return {
        "file_id": file_id,
        "site": site,
        "date": dt,
        "time_utc": hms,
        "hour_utc": int(hms[:2]),
        "month": int(dt.month) if pd.notna(dt) else -1,
    }


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


# --- Load competition data ---
taxonomy = pd.read_csv(BASE / "taxonomy.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY_LABELS)

taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)

label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}


def parse_soundscape_labels(x):
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]


def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))


# Deduplicate and aggregate labels per 5s window
sc_clean = (
    soundscape_labels.groupby(["filename", "start", "end"])["primary_label"]
    .apply(union_labels)
    .reset_index(name="label_list")
)

sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
sc_clean["row_id"] = (
    sc_clean["filename"].str.replace(".ogg", "", regex=False)
    + "_"
    + sc_clean["end_sec"].astype(str)
)

meta = sc_clean["filename"].apply(parse_soundscape_filename).apply(pd.Series)
sc_clean = pd.concat([sc_clean, meta], axis=1)

# Fully-labeled files
windows_per_file = sc_clean.groupby("filename").size()
full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)

# Multi-hot label matrix
Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
for i, labels in enumerate(sc_clean["label_list"]):
    idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
    if idxs:
        Y_SC[i, idxs] = 1

full_truth = (
    sc_clean[sc_clean["file_fully_labeled"]]
    .sort_values(["filename", "end_sec"])
    .reset_index(drop=False)
)

Y_FULL_TRUTH = Y_SC[full_truth["index"].to_numpy()]

CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA = {"Amphibia", "Insecta"}

print(f"N_CLASSES={N_CLASSES}, sc_clean={sc_clean.shape}, full_files={len(full_files)}")

# ============================================================================
# SECTION 3: Perch ONNX Pipeline
# ============================================================================

# --- 3A: Perch mapping + proxies (lazy init) ---
# These module-level variables are populated by _init_perch_mapping()
_PERCH_MAPPING_INITIALIZED = False
BC_INDICES = None
MAPPED_MASK = None
MAPPED_POS = None
UNMAPPED_POS = None
MAPPED_BC_INDICES = None
ACTIVE_CLASSES = None
idx_active_texture = None
idx_active_event = None
idx_mapped_active_texture = None
idx_mapped_active_event = None
idx_unmapped_active_texture = None
idx_unmapped_active_event = None
idx_unmapped_inactive = None
selected_proxy_pos_to_bc = None
idx_selected_proxy_active_texture = None
idx_selected_prioronly_active_texture = None
idx_selected_prioronly_active_event = None
bc_labels = None


def _init_perch_mapping():
    """Lazy-initialize Perch species mapping and proxy structures."""
    global _PERCH_MAPPING_INITIALIZED
    global BC_INDICES, MAPPED_MASK, MAPPED_POS, UNMAPPED_POS, MAPPED_BC_INDICES
    global ACTIVE_CLASSES, idx_active_texture, idx_active_event
    global idx_mapped_active_texture, idx_mapped_active_event
    global idx_unmapped_active_texture, idx_unmapped_active_event
    global idx_unmapped_inactive
    global selected_proxy_pos_to_bc
    global idx_selected_proxy_active_texture
    global idx_selected_prioronly_active_texture, idx_selected_prioronly_active_event
    global bc_labels

    if _PERCH_MAPPING_INITIALIZED:
        return

    MODEL_DIR = Path(
        "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"
    )

    bc_labels = (
        pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
        .reset_index()
        .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
    )

    NO_LABEL_INDEX = len(bc_labels)

    MANUAL_SCIENTIFIC_NAME_MAP = {}

    tax = taxonomy.copy()
    tax["scientific_name_lookup"] = tax["scientific_name"].replace(MANUAL_SCIENTIFIC_NAME_MAP)

    bc_lookup = bc_labels.rename(columns={"scientific_name": "scientific_name_lookup"})

    mapping = tax.merge(
        bc_lookup[["scientific_name_lookup", "bc_index"]], on="scientific_name_lookup", how="left"
    )
    mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL_INDEX).astype(int)

    label_to_bc_index = mapping.set_index("primary_label")["bc_index"]
    BC_INDICES = np.array([int(label_to_bc_index.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)

    MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX
    MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
    UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
    MAPPED_BC_INDICES = BC_INDICES[MAPPED_MASK].astype(np.int32)

    ACTIVE_CLASSES = [PRIMARY_LABELS[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]]

    idx_active_texture = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA],
        dtype=np.int32,
    )
    idx_active_event = np.array(
        [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA],
        dtype=np.int32,
    )

    idx_mapped_active_texture = idx_active_texture[MAPPED_MASK[idx_active_texture]]
    idx_mapped_active_event = idx_active_event[MAPPED_MASK[idx_active_event]]

    idx_unmapped_active_texture = idx_active_texture[~MAPPED_MASK[idx_active_texture]]
    idx_unmapped_active_event = idx_active_event[~MAPPED_MASK[idx_active_event]]

    idx_unmapped_inactive = np.array(
        [i for i in UNMAPPED_POS if PRIMARY_LABELS[i] not in ACTIVE_CLASSES], dtype=np.int32
    )

    # Build automatic genus proxies for unmapped non-sonotypes
    unmapped_df = mapping[mapping["bc_index"] == NO_LABEL_INDEX].copy()
    unmapped_non_sonotype = unmapped_df[
        ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
    ].copy()

    def get_genus_hits(scientific_name):
        genus = str(scientific_name).split()[0]
        hits = bc_labels[
            bc_labels["scientific_name"].astype(str).str.match(rf"^{re.escape(genus)}\s", na=False)
        ].copy()
        return genus, hits

    proxy_map = {}
    for _, row in unmapped_non_sonotype.iterrows():
        target = row["primary_label"]
        sci = row["scientific_name"]
        genus, hits = get_genus_hits(sci)
        if len(hits) > 0:
            proxy_map[target] = {
                "target_scientific_name": sci,
                "genus": genus,
                "bc_indices": hits["bc_index"].astype(int).tolist(),
                "proxy_scientific_names": hits["scientific_name"].tolist(),
            }

    PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
    SELECTED_PROXY_TARGETS = sorted(
        [t for t in proxy_map.keys() if CLASS_NAME_MAP.get(t) in PROXY_TAXA]
    )

    selected_proxy_pos = np.array([label_to_idx[c] for c in SELECTED_PROXY_TARGETS], dtype=np.int32)

    selected_proxy_pos_to_bc = {
        label_to_idx[target]: np.array(proxy_map[target]["bc_indices"], dtype=np.int32)
        for target in SELECTED_PROXY_TARGETS
    }

    idx_selected_proxy_active_texture = np.intersect1d(selected_proxy_pos, idx_active_texture)
    idx_selected_prioronly_active_texture = np.setdiff1d(
        idx_unmapped_active_texture, selected_proxy_pos
    )
    idx_selected_prioronly_active_event = np.setdiff1d(
        idx_unmapped_active_event, selected_proxy_pos
    )

    _PERCH_MAPPING_INITIALIZED = True
    print(f"Perch mapping initialized: {MAPPED_MASK.sum()} mapped, {(~MAPPED_MASK).sum()} unmapped")
    print(
        f"Proxy targets: {len(SELECTED_PROXY_TARGETS)}, active texture: {len(idx_active_texture)}"
    )


# --- Perch config ---
PERCH_CFG = {
    "batch_files": 16,
    "proxy_reduce": "max",
    "dryrun_n_files": 20,
    "proto_ssm": {
        "d_model": 320,
        "d_state": 32,
        "n_ssm_layers": 4,
        "dropout": 0.12,
        "n_prototypes": 2,
        "n_sites": 20,
        "meta_dim": 24,
        "use_cross_attn": True,
        "cross_attn_heads": 8,
    },
    "residual_ssm": {
        "d_model": 128,
        "d_state": 16,
        "n_ssm_layers": 2,
        "dropout": 0.1,
        "correction_weight": 0.35,
        "n_epochs": 40,
        "lr": 1e-3,
        "patience": 8,
    },
    "frozen_best_probe": {"pca_dim": 128, "min_pos": 5, "C": 0.75, "alpha": 0.45},
    "temperature": {"aves": 1.10, "texture": 0.95},
    "file_level_top_k": 2,
    "tta_shifts": [0, 1, -1, 2, -2],
    "rank_aware_scale": True,
    "rank_aware_power": 0.4,
    "delta_shift_alpha": 0.20,
    "best_fusion": {
        "lambda_event": 0.45,
        "lambda_texture": 1.1,
        "lambda_proxy_texture": 0.9,
        "smooth_texture": 0.35,
        "smooth_event": 0.15,
    },
}

BEST = PERCH_CFG["best_fusion"]


# --- 3B: Perch ONNX inference ---


def read_soundscape_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Unexpected sample rate {sr} in {path}; expected {SR}")
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    elif len(y) > FILE_SAMPLES:
        y = y[:FILE_SAMPLES]
    return y


def infer_perch_with_embeddings(
    perch_session, paths, batch_files=16, verbose=True, proxy_reduce="max"
):
    """Run Perch ONNX inference on soundscape files.

    Returns: (meta_df, scores, embeddings)
    """
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)

    scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)

    write_row = 0
    iterator = range(0, n_files, batch_files)
    if verbose:
        iterator = tqdm(
            iterator, total=(n_files + batch_files - 1) // batch_files, desc="Perch batches"
        )

    for start in iterator:
        batch_paths = paths[start : start + batch_files]
        batch_n = len(batch_paths)

        x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        batch_row_start = write_row
        x_pos = 0

        for path in batch_paths:
            y = read_soundscape_60s(path)
            x[x_pos : x_pos + N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)

            meta = parse_soundscape_filename(path.name)
            stem = path.stem

            row_ids[write_row : write_row + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
            filenames[write_row : write_row + N_WINDOWS] = path.name
            sites[write_row : write_row + N_WINDOWS] = meta["site"]
            hours[write_row : write_row + N_WINDOWS] = int(meta["hour_utc"])

            x_pos += N_WINDOWS
            write_row += N_WINDOWS

        # ONNX inference (replaces TF call)
        onnx_out = perch_session.run(None, {"inputs": x.astype(np.float32)})
        emb = onnx_out[0].astype(np.float32)  # (B, 1536) embeddings
        logits = onnx_out[3].astype(np.float32)  # (B, 14795) label logits

        scores[batch_row_start:write_row, MAPPED_POS] = logits[:, MAPPED_BC_INDICES]
        embeddings[batch_row_start:write_row] = emb

        # Selected genus proxies
        for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
            sub = logits[:, bc_idx_arr]
            if proxy_reduce == "max":
                proxy_score = sub.max(axis=1)
            elif proxy_reduce == "mean":
                proxy_score = sub.mean(axis=1)
            else:
                raise ValueError("proxy_reduce must be 'max' or 'mean'")
            scores[batch_row_start:write_row, pos] = proxy_score.astype(np.float32)

        del x, onnx_out, logits, emb
        gc.collect()

    meta_df = pd.DataFrame(
        {
            "row_id": row_ids,
            "filename": filenames,
            "site": sites,
            "hour_utc": hours,
        }
    )

    return meta_df, scores, embeddings


# --- 3C: ProtoSSM model classes ---


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
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

    def forward(self, x):
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
    def __init__(self, d_model, n_heads=4, dropout=0.1):
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

    def forward(self, x):
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class ProtoSSMv2(nn.Module):
    def __init__(
        self,
        d_input=1536,
        d_model=192,
        d_state=16,
        n_ssm_layers=2,
        n_classes=234,
        n_windows=12,
        dropout=0.2,
        n_sites=20,
        meta_dim=16,
        use_cross_attn=True,
        cross_attn_heads=4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

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

        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(
                d_model, n_heads=cross_attn_heads, dropout=dropout
            )

        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))

        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        self.n_families = 0
        self.family_head = None

    def init_family_head(self, n_families, class_to_family):
        self.n_families = n_families
        self.family_head = nn.Linear(self.d_model, n_families)
        self.register_buffer("class_to_family", torch.tensor(class_to_family, dtype=torch.long))

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape

        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            s_emb = self.site_emb(site_ids)
            h_emb = self.hour_emb(hours)
            meta = self.meta_proj(torch.cat([s_emb, h_emb], dim=-1))
            h = h + meta[:, None, :]

        for fwd, bwd, merge, norm in zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)

        if self.use_cross_attn:
            h = self.cross_attn(h)

        h_temporal = h

        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        temp = F.softplus(self.proto_temp)
        sim = torch.matmul(h_norm, p_norm.T) * temp + self.class_bias[None, None, :]

        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim

        family_logits = None
        if self.family_head is not None:
            h_pool = h.mean(dim=1)
            family_logits = self.family_head(h_pool)

        return species_logits, family_logits, h_temporal

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualSSM(nn.Module):
    def __init__(
        self,
        d_input=1536,
        d_scores=234,
        d_model=64,
        d_state=8,
        n_classes=234,
        n_windows=12,
        dropout=0.1,
        n_sites=20,
        meta_dim=8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm_drop = nn.Dropout(dropout)

        self.output_head = nn.Linear(d_model, n_classes)

        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, emb, first_pass_scores, site_ids=None, hours=None):
        B, T, _ = emb.shape

        x = torch.cat([emb, first_pass_scores], dim=-1)
        h = self.input_proj(x)

        if site_ids is not None and hours is not None:
            site_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            hour_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([site_e, hour_e], dim=-1))
            h = h + meta.unsqueeze(1)

        h = h + self.pos_enc[:, :T, :]

        residual = h
        h_f = self.ssm_fwd(h)
        h_b = self.ssm_bwd(h.flip(1)).flip(1)
        h = self.ssm_merge(torch.cat([h_f, h_b], dim=-1))
        h = self.ssm_drop(h)
        h = self.ssm_norm(h + residual)

        correction = self.output_head(h)
        return correction

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_pretrained_artifacts(pretrained_dir):
    """Load all training artifacts for inference-only mode."""
    pretrained_dir = Path(pretrained_dir)

    with open(pretrained_dir / "config.json") as f:
        config = json.load(f)

    emb_scaler = joblib.load(pretrained_dir / "emb_scaler.joblib")
    emb_pca = joblib.load(pretrained_dir / "emb_pca.joblib")
    probe_models = joblib.load(pretrained_dir / "probe_models.joblib")
    prior_tables = joblib.load(pretrained_dir / "prior_tables.joblib")
    taxonomy_data = joblib.load(pretrained_dir / "taxonomy.joblib")
    per_class_thresholds = np.load(pretrained_dir / "per_class_thresholds.npy")

    res_ssm_path = pretrained_dir / "residual_ssm_state.pt"

    return {
        "config": config,
        "emb_scaler": emb_scaler,
        "emb_pca": emb_pca,
        "probe_models": probe_models,
        "prior_tables": prior_tables,
        "per_class_thresholds": per_class_thresholds,
        "taxonomy": taxonomy_data,
        "proto_ssm_path": pretrained_dir / "proto_ssm_state.pt",
        "residual_ssm_path": res_ssm_path if res_ssm_path.exists() else None,
    }


# --- 3D: Perch utility functions ---


def smooth_cols_fixed12(scores, cols, alpha=0.35):
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    assert len(s) % N_WINDOWS == 0, "Expected full-file blocks of 12 windows"
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return s


def smooth_events_fixed12(scores, cols, alpha=0.15):
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    assert len(s) % N_WINDOWS == 0
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev_x = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    next_x = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    local_max = np.maximum(x, np.maximum(prev_x, next_x))
    view[:, :, cols] = (1.0 - alpha) * x + alpha * local_max
    return s


def seq_features_1d(v):
    assert len(v) % N_WINDOWS == 0, "Expected full-file blocks of 12 windows"
    x = v.reshape(-1, N_WINDOWS)
    prev_v = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    next_v = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    mean_v = np.repeat(x.mean(axis=1), N_WINDOWS)
    max_v = np.repeat(x.max(axis=1), N_WINDOWS)
    std_v = np.repeat(x.std(axis=1), N_WINDOWS)
    return prev_v, next_v, mean_v, max_v, std_v


def file_level_confidence_scale(preds, n_windows=12, top_k=2):
    N, C = preds.shape
    assert N % n_windows == 0
    view = preds.reshape(-1, n_windows, C)
    sorted_view = np.sort(view, axis=1)
    top_k_mean = sorted_view[:, -top_k:, :].mean(axis=1, keepdims=True)
    scaled = view * top_k_mean
    return scaled.reshape(N, C)


def temporal_shift_tta(emb_files, logits_files, model, site_ids, hours, shifts=[0, 1, -1]):
    all_preds = []
    model.eval()

    for shift in shifts:
        if shift == 0:
            e = emb_files
            l = logits_files
        else:
            e = np.roll(emb_files, shift, axis=1)
            l = np.roll(logits_files, shift, axis=1)

        with torch.no_grad():
            out, _, _ = model(
                torch.tensor(e, dtype=torch.float32),
                torch.tensor(l, dtype=torch.float32),
                site_ids=torch.tensor(site_ids, dtype=torch.long),
                hours=torch.tensor(hours, dtype=torch.long),
            )
            pred = out.numpy()

        if shift != 0:
            pred = np.roll(pred, -shift, axis=1)

        all_preds.append(pred)

    return np.mean(all_preds, axis=0)


def rank_aware_scaling(scores, n_windows=12, power=0.5):
    N, C = scores.shape
    assert N % n_windows == 0
    n_files = N // n_windows
    view = scores.reshape(n_files, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)
    scale = np.power(file_max, power)
    scaled = view * scale
    return scaled.reshape(N, C)


def apply_per_class_thresholds(scores, thresholds, n_windows=12):
    N, C = scores.shape
    assert C == len(thresholds)
    scaled = np.copy(scores)
    for c in range(C):
        t = thresholds[c]
        mask_above = scores[:, c] > t
        scaled[mask_above, c] = 0.5 + 0.5 * (scores[mask_above, c] - t) / (1 - t + 1e-8)
        scaled[~mask_above, c] = 0.5 * scores[~mask_above, c] / (t + 1e-8)
    return np.clip(scaled, 0, 1)


def _build_v18_thresholds():
    """Return hardcoded V18 per-class thresholds (optimized from V18 OOF logs)."""
    return np.array(
        [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.30000001192092896,
            0.5,
            0.5,
            0.5,
            0.5,
            0.4000000059604645,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.3499999940395355,
            0.5,
            0.6000000238418579,
            0.699999988079071,
            0.5,
            0.5,
            0.25,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.6000000238418579,
            0.5,
            0.699999988079071,
            0.699999988079071,
            0.25,
            0.3499999940395355,
            0.4000000059604645,
            0.5,
            0.5,
            0.3499999940395355,
            0.5,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.30000001192092896,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.699999988079071,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.6499999761581421,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.30000001192092896,
            0.5,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.30000001192092896,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.699999988079071,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        dtype=np.float32,
    )


def prior_logits_from_tables(sites, hours, tables, eps=1e-4):
    n = len(sites)
    p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)

    site_idx = np.fromiter(
        (tables["site_to_i"].get(str(s), -1) for s in sites), dtype=np.int32, count=n
    )
    hour_idx = np.fromiter(
        (tables["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours),
        dtype=np.int32,
        count=n,
    )
    sh_idx = np.fromiter(
        (
            tables["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1
            for s, h in zip(sites, hours)
        ),
        dtype=np.int32,
        count=n,
    )

    valid = hour_idx >= 0
    if valid.any():
        nh = tables["hour_n"][hour_idx[valid]][:, None]
        wh = nh / (nh + 8.0)
        p[valid] = wh * tables["hour_p"][hour_idx[valid]] + (1.0 - wh) * p[valid]

    valid = site_idx >= 0
    if valid.any():
        ns = tables["site_n"][site_idx[valid]][:, None]
        ws = ns / (ns + 8.0)
        p[valid] = ws * tables["site_p"][site_idx[valid]] + (1.0 - ws) * p[valid]

    valid = sh_idx >= 0
    if valid.any():
        nsh = tables["sh_n"][sh_idx[valid]][:, None]
        wsh = nsh / (nsh + 4.0)
        p[valid] = wsh * tables["sh_p"][sh_idx[valid]] + (1.0 - wsh) * p[valid]

    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32, copy=False)


def fuse_scores_with_tables(
    base_scores,
    sites,
    hours,
    tables,
    lambda_event=BEST["lambda_event"],
    lambda_texture=BEST["lambda_texture"],
    lambda_proxy_texture=BEST["lambda_proxy_texture"],
    smooth_texture=BEST["smooth_texture"],
    smooth_event=BEST["smooth_event"],
):
    scores = base_scores.copy()
    prior = prior_logits_from_tables(sites, hours, tables)

    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += lambda_event * prior[:, idx_mapped_active_event]

    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += lambda_texture * prior[:, idx_mapped_active_texture]

    if len(idx_selected_proxy_active_texture):
        scores[:, idx_selected_proxy_active_texture] += (
            lambda_proxy_texture * prior[:, idx_selected_proxy_active_texture]
        )

    if len(idx_selected_prioronly_active_event):
        scores[:, idx_selected_prioronly_active_event] = (
            lambda_event * prior[:, idx_selected_prioronly_active_event]
        )

    if len(idx_selected_prioronly_active_texture):
        scores[:, idx_selected_prioronly_active_texture] = (
            lambda_texture * prior[:, idx_selected_prioronly_active_texture]
        )

    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0

    scores = smooth_cols_fixed12(scores, idx_active_texture, alpha=smooth_texture)
    scores = smooth_events_fixed12(scores, idx_active_event, alpha=smooth_event)
    return scores.astype(np.float32, copy=False), prior


def build_class_features(emb_proj, raw_col, prior_col, base_col):
    prev_base, next_base, mean_base, max_base, std_base = seq_features_1d(base_col)

    diff_mean = base_col - mean_base
    diff_prev = base_col - prev_base
    diff_next = base_col - next_base

    feats = np.concatenate(
        [
            emb_proj,
            raw_col[:, None],
            prior_col[:, None],
            base_col[:, None],
            prev_base[:, None],
            next_base[:, None],
            mean_base[:, None],
            max_base[:, None],
            std_base[:, None],
            diff_mean[:, None],
            diff_prev[:, None],
            diff_next[:, None],
            (raw_col * prior_col)[:, None],
            (raw_col * base_col)[:, None],
            (prior_col * base_col)[:, None],
        ],
        axis=1,
    )

    return feats.astype(np.float32, copy=False)


def build_taxonomy_groups(taxonomy_df, primary_labels):
    for col in ["family", "order", "class_name"]:
        if col in taxonomy_df.columns:
            group_map = taxonomy_df.set_index("primary_label")[col].to_dict()
            break
    else:
        group_map = {label: "Unknown" for label in primary_labels}

    groups = sorted(set(group_map.values()))
    grp_to_idx = {g: i for i, g in enumerate(groups)}
    class_to_group = []
    for label in primary_labels:
        grp = group_map.get(label, "Unknown")
        class_to_group.append(grp_to_idx.get(grp, 0))
    return len(groups), class_to_group, grp_to_idx


def build_site_mapping(meta_df):
    sites = meta_df["site"].unique().tolist()
    site_to_idx = {s: i + 1 for i, s in enumerate(sites)}
    n_sites = len(sites) + 1
    return site_to_idx, n_sites


def reshape_to_files(flat_array, meta_df, n_windows=N_WINDOWS):
    filenames = meta_df["filename"].to_numpy()
    unique_files = []
    seen = set()
    for f in filenames:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)

    n_files = len(unique_files)
    assert (
        len(flat_array) == n_files * n_windows
    ), f"Expected {n_files * n_windows} rows, got {len(flat_array)}"

    new_shape = (n_files, n_windows) + flat_array.shape[1:]
    return flat_array.reshape(new_shape), unique_files


def get_file_metadata(meta_df, file_list, site_to_idx, n_sites_max):
    file_to_row = {}
    filenames = meta_df["filename"].to_numpy()
    sites_arr = meta_df["site"].to_numpy()
    hours_arr = meta_df["hour_utc"].to_numpy()
    for i, f in enumerate(filenames):
        if f not in file_to_row:
            file_to_row[f] = i

    site_ids = np.zeros(len(file_list), dtype=np.int64)
    hour_ids = np.zeros(len(file_list), dtype=np.int64)
    for fi, fname in enumerate(file_list):
        row = file_to_row.get(fname)
        if row is not None:
            sid = site_to_idx.get(sites_arr[row], 0)
            site_ids[fi] = min(sid, n_sites_max - 1)
            hour_ids[fi] = int(hours_arr[row]) % 24
    return site_ids, hour_ids


# --- 3E: Perch orchestrator ---


def run_perch_pipeline(test_paths):
    """Run Perch ONNX + ProtoSSM + MLP + 6-stage PP pipeline.

    Returns: pd.DataFrame with columns ["row_id"] + PRIMARY_LABELS
    """
    t_start = time.time()

    # Initialize mapping globals
    _init_perch_mapping()

    # Find and load Perch ONNX model
    perch_onnx_candidates = [
        Path(
            "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1/saved_model.onnx"
        ),
        Path(_find_input("birdclef-2026-perch-onnx")) / "perch_v2.onnx",
    ]
    # Also search in model dir for any .onnx file
    model_dir = Path(
        "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"
    )
    if model_dir.exists():
        for f in model_dir.glob("*.onnx"):
            perch_onnx_candidates.insert(0, f)

    perch_onnx_path = None
    for p in perch_onnx_candidates:
        if p.exists():
            perch_onnx_path = p
            break

    if perch_onnx_path is None:
        raise FileNotFoundError(
            f"Perch ONNX model not found. Searched: {[str(p) for p in perch_onnx_candidates]}"
        )

    print(f"Loading Perch ONNX model from {perch_onnx_path}")
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = os.cpu_count() or 4
    sess_opts.inter_op_num_threads = 1
    perch_session = ort.InferenceSession(
        str(perch_onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )
    print(f"Perch ONNX session created ({time.time() - t_start:.1f}s)")

    # Step 1: Run Perch inference
    meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings(
        perch_session,
        test_paths,
        batch_files=PERCH_CFG["batch_files"],
        verbose=True,
        proxy_reduce=PERCH_CFG["proxy_reduce"],
    )
    print(f"Perch inference done: {meta_test.shape[0]} windows from {len(test_paths)} files")

    # Free ONNX session
    del perch_session
    gc.collect()

    # Step 2: Load pretrained artifacts
    pretrained_candidates = [
        Path("/kaggle/input/datasets/murilogustineli/birdclef-2026-pretrained-protossm-v18"),
        Path("/kaggle/input/datasets/birdclef-2026-pretrained-protossm-v18"),
        Path("/kaggle/input/birdclef-2026-pretrained-protossm-v18"),
    ]
    pretrained_dir = None
    for _p in pretrained_candidates:
        if _p.exists() and (_p / "config.json").exists():
            pretrained_dir = _p
            break

    if pretrained_dir is None:
        raise FileNotFoundError(
            f"ProtoSSM pretrained artifacts not found. Searched: {[str(p) for p in pretrained_candidates]}"
        )

    print(f"Loading pretrained artifacts from {pretrained_dir}")
    _artifacts = load_pretrained_artifacts(pretrained_dir)
    _cfg_loaded = _artifacts["config"]

    ENSEMBLE_WEIGHT_PROTO = _cfg_loaded["ensemble_weight_proto"]
    CORRECTION_WEIGHT = _cfg_loaded["correction_weight"]
    site_to_idx = _cfg_loaded["site_to_idx"]

    emb_scaler = _artifacts["emb_scaler"]
    emb_pca = _artifacts["emb_pca"]
    probe_models = _artifacts["probe_models"]
    final_prior_tables = _artifacts["prior_tables"]
    PER_CLASS_THRESHOLDS = _artifacts["per_class_thresholds"]

    _tax = _artifacts["taxonomy"]
    class_to_family = _tax["class_to_family"]
    fam_to_idx = _tax["fam_to_idx"]
    n_families = _tax["n_families"]

    # Step 3: Reconstruct ProtoSSM model
    _ssm_cfg = _cfg_loaded.get("proto_ssm", PERCH_CFG["proto_ssm"])
    model = ProtoSSMv2(
        d_input=1536,
        d_model=_ssm_cfg["d_model"],
        d_state=_ssm_cfg["d_state"],
        n_ssm_layers=_ssm_cfg["n_ssm_layers"],
        n_classes=N_CLASSES,
        n_windows=N_WINDOWS,
        dropout=_ssm_cfg["dropout"],
        n_sites=_ssm_cfg["n_sites"],
        meta_dim=_ssm_cfg.get("meta_dim", 16),
        use_cross_attn=_ssm_cfg.get("use_cross_attn", True),
        cross_attn_heads=_ssm_cfg.get("cross_attn_heads", 4),
    ).to(DEVICE)
    model.init_family_head(n_families, class_to_family)
    model.load_state_dict(torch.load(_artifacts["proto_ssm_path"], map_location=DEVICE))
    model.eval()
    print(f"ProtoSSM loaded: {model.count_parameters():,} params")

    # Reconstruct ResidualSSM
    res_model = None
    if _artifacts["residual_ssm_path"] is not None:
        _res_cfg = _cfg_loaded.get("residual_ssm", PERCH_CFG["residual_ssm"])
        res_model = ResidualSSM(
            d_input=1536,
            d_scores=N_CLASSES,
            d_model=_res_cfg["d_model"],
            d_state=_res_cfg["d_state"],
            n_classes=N_CLASSES,
            n_windows=N_WINDOWS,
            dropout=_res_cfg["dropout"],
            n_sites=_ssm_cfg["n_sites"],
            meta_dim=8,
        ).to(DEVICE)
        res_model.load_state_dict(torch.load(_artifacts["residual_ssm_path"], map_location=DEVICE))
        res_model.eval()
        print(f"ResidualSSM loaded: {res_model.count_parameters():,} params")

    # Step 4: ProtoSSM inference with TTA
    emb_test_files, test_file_list = reshape_to_files(emb_test, meta_test)
    logits_test_files, _ = reshape_to_files(scores_test_raw, meta_test)

    test_site_ids, test_hours = get_file_metadata(
        meta_test, test_file_list, site_to_idx, PERCH_CFG["proto_ssm"]["n_sites"]
    )

    tta_shifts = PERCH_CFG.get("tta_shifts", [0])
    if len(tta_shifts) > 1:
        print(f"Running TTA with shifts: {tta_shifts}")
        proto_scores = temporal_shift_tta(
            emb_test_files, logits_test_files, model, test_site_ids, test_hours, shifts=tta_shifts
        )
    else:
        with torch.no_grad():
            proto_out, _, _ = model(
                torch.tensor(emb_test_files, dtype=torch.float32),
                torch.tensor(logits_test_files, dtype=torch.float32),
                site_ids=torch.tensor(test_site_ids, dtype=torch.long),
                hours=torch.tensor(test_hours, dtype=torch.long),
            )
            proto_scores = proto_out.numpy()

    proto_scores_flat = proto_scores.reshape(-1, N_CLASSES).astype(np.float32)

    # Step 5: Prior-fused base scores
    test_base_scores, test_prior_scores = fuse_scores_with_tables(
        scores_test_raw,
        sites=meta_test["site"].to_numpy(),
        hours=meta_test["hour_utc"].to_numpy(),
        tables=final_prior_tables,
    )

    # Step 6: MLP probe scores
    emb_test_scaled = emb_scaler.transform(emb_test)
    Z_TEST = emb_pca.transform(emb_test_scaled).astype(np.float32)

    mlp_scores = test_base_scores.copy()

    for cls_idx, clf in probe_models.items():
        X_cls_test = build_class_features(
            Z_TEST,
            raw_col=scores_test_raw[:, cls_idx],
            prior_col=test_prior_scores[:, cls_idx],
            base_col=test_base_scores[:, cls_idx],
        )

        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X_cls_test)[:, 1].astype(np.float32)
            pred = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        else:
            pred = clf.decision_function(X_cls_test).astype(np.float32)

        alpha = float(PERCH_CFG["frozen_best_probe"]["alpha"])
        mlp_scores[:, cls_idx] = (1.0 - alpha) * test_base_scores[:, cls_idx] + alpha * pred

    # Step 7: Ensemble fusion (proto + mlp)
    print(f"Using ensemble weight (proto): {ENSEMBLE_WEIGHT_PROTO:.2f}")
    final_test_scores = (
        ENSEMBLE_WEIGHT_PROTO * proto_scores_flat + (1.0 - ENSEMBLE_WEIGHT_PROTO) * mlp_scores
    ).astype(np.float32)

    # Step 8: ResidualSSM correction
    if res_model is not None and CORRECTION_WEIGHT > 0:
        first_pass_test_files, _ = reshape_to_files(final_test_scores, meta_test)
        first_pass_test_t = torch.tensor(first_pass_test_files, dtype=torch.float32)

        emb_test_tensor = torch.tensor(emb_test_files, dtype=torch.float32)
        test_site_tensor = torch.tensor(test_site_ids, dtype=torch.long)
        test_hour_tensor = torch.tensor(test_hours, dtype=torch.long)

        res_model.eval()
        with torch.no_grad():
            test_correction = res_model(
                emb_test_tensor,
                first_pass_test_t,
                site_ids=test_site_tensor,
                hours=test_hour_tensor,
            ).numpy()

        test_correction_flat = test_correction.reshape(-1, N_CLASSES).astype(np.float32)
        final_test_scores = final_test_scores + CORRECTION_WEIGHT * test_correction_flat
        print(f"Residual correction applied: mean_abs={np.abs(test_correction_flat).mean():.4f}")
    else:
        print("Residual correction: SKIPPED")

    # Step 9: 6-stage post-processing pipeline

    # PP Stage 1: Per-taxon temperature scaling
    temp_cfg = PERCH_CFG["temperature"]
    T_AVES = temp_cfg["aves"]
    T_TEXTURE = temp_cfg["texture"]

    class_temperatures = np.ones(N_CLASSES, dtype=np.float32) * T_AVES
    for ci, label in enumerate(PRIMARY_LABELS):
        cn = CLASS_NAME_MAP.get(label, "Aves")
        if cn in TEXTURE_TAXA:
            class_temperatures[ci] = T_TEXTURE

    scaled_scores = final_test_scores / class_temperatures[None, :]

    def _sigmoid_perch(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    probs = _sigmoid_perch(scaled_scores)

    # PP Stage 2: File-level confidence scaling
    top_k = PERCH_CFG.get("file_level_top_k", 0)
    if top_k > 0:
        probs = file_level_confidence_scale(probs, n_windows=N_WINDOWS, top_k=top_k)
        probs = np.clip(probs, 0.0, 1.0)

    # PP Stage 3: Rank-aware scaling
    if PERCH_CFG.get("rank_aware_scale", False):
        power = PERCH_CFG.get("rank_aware_power", 0.5)
        probs = rank_aware_scaling(probs, n_windows=N_WINDOWS, power=power)
        probs = np.clip(probs, 0.0, 1.0)

    # PP Stage 4: Adaptive delta shift smoothing
    def adaptive_delta_smooth(probs, n_windows, base_alpha=0.20):
        n_files = probs.shape[0] // n_windows
        result = probs.copy()
        view = result.reshape(n_files, n_windows, -1)
        p_view = probs.reshape(n_files, n_windows, -1)
        for i in range(1, n_windows - 1):
            conf = p_view[:, i, :].max(axis=-1, keepdims=True)
            a = base_alpha * (1.0 - conf)
            neighbor_avg = (p_view[:, i - 1, :] + p_view[:, i + 1, :]) / 2.0
            view[:, i, :] = (1.0 - a) * p_view[:, i, :] + a * neighbor_avg
        return result.reshape(probs.shape)

    alpha = PERCH_CFG.get("delta_shift_alpha", 0.0)
    if alpha > 0:
        probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS, base_alpha=alpha)
        probs = np.clip(probs, 0.0, 1.0)

    # PP Stage 5: Per-class threshold sharpening
    probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS, n_windows=N_WINDOWS)

    # Build output DataFrame
    submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
    submission.insert(0, "row_id", meta_test["row_id"].values)
    submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)

    print(
        f"Perch pipeline complete: {submission.shape}, range=[{probs.min():.6f}, {probs.max():.6f}]"
    )

    # Cleanup
    del model, res_model, emb_test, scores_test_raw, emb_test_files, logits_test_files
    del _artifacts, probe_models, emb_scaler, emb_pca, final_prior_tables
    gc.collect()

    return submission


# ============================================================================
# SECTION 4: SED Pipeline
# ============================================================================

# --- 4A: Mel computation ---


def compute_mel_amplitude_to_db(audio, mel_cfg):
    """Compute mel spectrogram with AmplitudeToDB + standardize + min-max [0,1]."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg.get("fmin", 0),
        fmax=mel_cfg.get("fmax"),
        htk=mel_cfg.get("htk", True),
        power=2.0,
    )
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))

    top_db = mel_cfg.get("top_db", 80.0)
    mel_max = mel_db.max()
    mel_db = np.maximum(mel_db, mel_max - top_db)

    mean = mel_db.mean()
    std = max(mel_db.std(), 1e-6)
    mel_norm = (mel_db - mean) / std

    mel_min = mel_norm.min()
    mel_max = mel_norm.max()
    mel_norm = (mel_norm - mel_min) / (mel_max - mel_min + 1e-6)

    target_frames = mel_cfg.get("target_frames", 1000)
    T = mel_norm.shape[1]
    if T > target_frames:
        start = (T - target_frames) // 2
        mel_norm = mel_norm[:, start : start + target_frames]
    elif T < target_frames:
        mel_norm = np.pad(mel_norm, ((0, 0), (0, target_frames - T)), mode="constant")

    return mel_norm[np.newaxis, np.newaxis, :, :].astype(np.float32)


def compute_mel_ttahara(audio, mel_cfg):
    """Compute mel spectrogram with ttahara's pipeline: torchaudio + resize + min-max."""
    import torchaudio
    from torchvision.transforms import v2 as tvt_v2

    sr = mel_cfg["sr"]
    wave = torch.from_numpy(audio).float().unsqueeze(0)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=mel_cfg.get("n_fft", 2048),
        win_length=mel_cfg.get("win_length", 626),
        hop_length=mel_cfg.get("hop_length", 313),
        f_min=mel_cfg.get("fmin", 20),
        n_mels=mel_cfg.get("n_mels", 256),
        power=2.0,
        center=True,
        pad_mode="reflect",
        norm="slaney",
        mel_scale="htk",
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(
        stype="power", top_db=mel_cfg.get("top_db", 80.0)
    )

    mel_spec = mel_transform(wave)
    lms = db_transform(mel_spec)

    target_shape = mel_cfg.get("target_shape", [256, 256])
    resize = tvt_v2.Resize(size=target_shape)
    lms = resize(lms)

    lms_flat = lms.reshape(1, -1)
    lms_min = lms_flat.min(dim=1)[0][:, None, None]
    lms_max = lms_flat.max(dim=1)[0][:, None, None]
    lms = (lms - lms_min) / (lms_max - lms_min + 1e-7)

    result = lms[:, None, :, :].numpy().astype(np.float32)
    in_chans = mel_cfg.get("in_chans", 1)
    if in_chans > 1:
        result = np.repeat(result, in_chans, axis=1)
    return result


def sed_compute_mel(audio, mel_cfg):
    """Dispatch to model-specific mel computation for SED."""
    mel_type = mel_cfg.get("type", "mel")
    norm_type = mel_cfg.get("norm_type", "audioset")
    if mel_type == "ttahara":
        return compute_mel_ttahara(audio, mel_cfg)
    elif norm_type == "amplitude_to_db":
        return compute_mel_amplitude_to_db(audio, mel_cfg)
    else:
        # Default: amplitude_to_db
        return compute_mel_amplitude_to_db(audio, mel_cfg)


# --- 4B: SED inference + PP ---

# Post-processing flags
SED_ENABLE_TTA = True
SED_ENABLE_TEMPORAL_SMOOTHING = True
SED_ENABLE_TOPN = True


def _sed_run_onnx(session, mel, mel_cfg):
    """Run ONNX inference and return probability vector."""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: mel})[0]
    output = output.flatten()
    if mel_cfg.get("output_type") == "probs":
        return output
    else:
        return sigmoid(output)


def _sed_predict_segment_with_tta(
    session, audio_path, sr, window_start_s, window_duration, mel_cfg, use_tta=True
):
    """Predict one segment, optionally with temporal TTA (2.5s shifts)."""
    audio = sed_load_audio_chunk(audio_path, sr, window_start_s, window_duration)
    mel = sed_compute_mel(audio, mel_cfg)
    probs = _sed_run_onnx(session, mel, mel_cfg)

    if not use_tta:
        return probs

    shift_s = 2.5
    tta_preds = [probs]

    for shift in [-shift_s, shift_s]:
        shifted_start = window_start_s + shift
        audio_shifted = sed_load_audio_chunk(audio_path, sr, shifted_start, window_duration)
        mel_shifted = sed_compute_mel(audio_shifted, mel_cfg)
        tta_preds.append(_sed_run_onnx(session, mel_shifted, mel_cfg))

    return 0.5 * tta_preds[0] + 0.25 * tta_preds[1] + 0.25 * tta_preds[2]


def sed_load_audio_chunk(audio_path, sr, start_s, duration_s):
    """Load a chunk of audio at target sample rate."""
    info = sf.info(audio_path)
    source_sr = info.samplerate
    total_duration = info.frames / source_sr
    target_samples = int(sr * duration_s)

    pre_pad_samples = 0
    read_start_s = start_s
    if start_s < 0:
        pre_pad_s = -start_s
        pre_pad_samples = int(sr * pre_pad_s)
        read_start_s = 0.0
        duration_s = duration_s - pre_pad_s

    actual_duration = min(duration_s, total_duration - read_start_s)
    if actual_duration <= 0:
        return np.zeros(target_samples, dtype=np.float32)

    audio, _ = sf.read(
        audio_path,
        dtype="float32",
        start=int(read_start_s * source_sr),
        frames=int(actual_duration * source_sr) + 2,
    )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if source_sr != sr:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)

    if pre_pad_samples > 0:
        audio = np.concatenate([np.zeros(pre_pad_samples, dtype=np.float32), audio])

    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    return audio.astype(np.float32)


def _sed_apply_temperature(probs_matrix, temperature):
    if temperature is None or temperature == 1.0:
        return probs_matrix
    eps = 1e-7
    p = np.clip(probs_matrix, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    scaled_logits = logits / temperature
    return sigmoid(scaled_logits)


def _sed_apply_temporal_smoothing(probs_matrix, taxa_mapping=None):
    """Taxa-aware temporal smoothing across consecutive segments."""
    n_segments, n_species = probs_matrix.shape
    if n_segments < 3:
        return probs_matrix

    smoothed = np.copy(probs_matrix)

    if taxa_mapping is None:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        for c in range(n_species):
            smoothed[:, c] = np.convolve(probs_matrix[:, c], kernel, mode="same")
        return smoothed

    event_indices = set()
    texture_indices = set()
    for taxa in ["aves", "mammalia", "reptilia"]:
        event_indices.update(taxa_mapping.get(taxa, []))
    for taxa in ["amphibia", "insecta"]:
        texture_indices.update(taxa_mapping.get(taxa, []))

    alpha_event = 0.15
    alpha_texture = 0.35

    for c in range(n_species):
        if c in event_indices:
            for t in range(n_segments):
                prev_val = probs_matrix[max(0, t - 1), c]
                next_val = probs_matrix[min(n_segments - 1, t + 1), c]
                smoothed[t, c] = (1 - alpha_event) * probs_matrix[t, c] + alpha_event * max(
                    prev_val, next_val
                )
        elif c in texture_indices:
            for t in range(n_segments):
                prev_val = probs_matrix[max(0, t - 1), c]
                next_val = probs_matrix[min(n_segments - 1, t + 1), c]
                smoothed[t, c] = (1 - alpha_texture) * probs_matrix[t, c] + alpha_texture * (
                    (prev_val + next_val) / 2
                )
        else:
            kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
            smoothed[:, c] = np.convolve(probs_matrix[:, c], kernel, mode="same")

    return smoothed


def _sed_apply_topn(probs_matrix):
    """TopN: multiply each segment by file-level max per species."""
    file_max = probs_matrix.max(axis=0, keepdims=True)
    return probs_matrix * file_max


def _sed_parse_soundscape_metadata(filename):
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 6:
        return None, None
    site = parts[3]
    try:
        hour = int(parts[5][:2])
    except (ValueError, IndexError):
        return site, None
    return site, hour


def _sed_apply_metadata_priors(probs_matrix, metadata_priors, site, hour, taxa_mapping, idx_map):
    """Apply site/hour metadata priors multiplicatively."""
    if metadata_priors is None:
        return probs_matrix

    n_segments, n_species = probs_matrix.shape
    lambda_map = {
        "aves": 0.3,
        "amphibia": 0.5,
        "insecta": 0.5,
        "mammalia": 0.2,
        "reptilia": 0.2,
    }

    idx_to_taxa = {}
    if taxa_mapping:
        for taxa_class, indices in taxa_mapping.items():
            for idx in indices:
                idx_to_taxa[idx] = taxa_class

    site_hour_priors = metadata_priors.get("site_hour_species", {})
    site_priors = metadata_priors.get("site_species", {})
    hour_priors = metadata_priors.get("hour_species", {})
    global_priors = metadata_priors.get("global_species", {})

    site_str = site if site else None
    hour_str = str(hour).zfill(2) if hour is not None else None

    prior_vector = np.ones(n_species, dtype=np.float32)
    for idx_str, species_code in idx_map.items():
        idx = int(idx_str)
        if idx >= n_species:
            continue

        prior_val = None
        if site_str and hour_str:
            prior_val = site_hour_priors.get(site_str, {}).get(hour_str, {}).get(species_code)
        if prior_val is None and site_str:
            prior_val = site_priors.get(site_str, {}).get(species_code)
        if prior_val is None and hour_str:
            prior_val = hour_priors.get(hour_str, {}).get(species_code)
        if prior_val is None:
            prior_val = global_priors.get(species_code)
        if prior_val is None:
            continue

        taxa = idx_to_taxa.get(idx, "aves")
        lam = lambda_map.get(taxa, 0.3)
        prior_vector[idx] = prior_val**lam

    return probs_matrix * prior_vector[np.newaxis, :]


# --- 4C: 4-fold ensemble inference ---


def sed_predict_soundscape_ensemble(
    sessions, audio_path, mel_cfg, species_mapping, taxa_mapping=None, metadata_priors=None
):
    """4-fold ensemble: compute mel once per segment, run all models, average probs."""
    sr = mel_cfg["sr"]
    segment_duration = 5.0
    inference_window = mel_cfg.get("inference_window", segment_duration)
    context_pad = (inference_window - segment_duration) / 2
    soundscape_id = Path(audio_path).stem

    info = sf.info(audio_path)
    total_duration = info.frames / info.samplerate

    species_2026 = species_mapping["birdclef2026_species"]
    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]
    n_model_outputs = max(int(k) for k in idx_map.keys()) + 1

    segment_mels = []
    segment_times = []
    segment_start_s = 0.0

    while segment_start_s < total_duration:
        audio_start = segment_start_s - context_pad

        if SED_ENABLE_TTA:
            shifts = [
                (audio_start, 0.50),
                (audio_start - 2.5, 0.25),
                (audio_start + 2.5, 0.25),
            ]
        else:
            shifts = [(audio_start, 1.0)]

        mels_with_weights = []
        for shift_start, weight in shifts:
            audio = sed_load_audio_chunk(audio_path, sr, shift_start, inference_window)
            mel = sed_compute_mel(audio, mel_cfg)
            mels_with_weights.append((mel, weight))
        segment_mels.append(mels_with_weights)

        segment_times.append(int(segment_start_s + segment_duration))
        segment_start_s += segment_duration

    n_segments = len(segment_mels)
    n_folds = len(sessions)

    fold_probs = np.zeros((n_folds, n_segments, n_model_outputs), dtype=np.float32)
    for fold_idx, sess in enumerate(sessions):
        for seg_idx, mels_with_weights in enumerate(segment_mels):
            seg_prob = np.zeros(n_model_outputs, dtype=np.float32)
            for mel, weight in mels_with_weights:
                prob = _sed_run_onnx(sess, mel, mel_cfg)
                seg_prob += weight * prob
            fold_probs[fold_idx, seg_idx] = seg_prob

    avg_probs = fold_probs.mean(axis=0)

    temperature = mel_cfg.get("temperature", 1.0)
    if temperature != 1.0:
        avg_probs = _sed_apply_temperature(avg_probs, temperature)

    if SED_ENABLE_TEMPORAL_SMOOTHING:
        avg_probs = _sed_apply_temporal_smoothing(avg_probs, taxa_mapping)

    if SED_ENABLE_TOPN:
        avg_probs = _sed_apply_topn(avg_probs)

    site, hour = _sed_parse_soundscape_metadata(audio_path)
    if metadata_priors is not None and (site is not None or hour is not None):
        avg_probs = _sed_apply_metadata_priors(
            avg_probs, metadata_priors, site, hour, taxa_mapping, idx_map
        )

    predictions = []
    for i, end_time in enumerate(segment_times):
        row_id = f"{soundscape_id}_{end_time}"
        row = {"row_id": row_id}
        for sp in species_2026:
            row[sp] = 0.0
        for idx_str, name_2026 in idx_map.items():
            row[name_2026] = float(avg_probs[i, int(idx_str)])
        predictions.append(row)

    return predictions


# --- 4D: SED orchestrator ---


def run_sed_pipeline(test_paths):
    """Run SED v5j-4fold-pp2 pipeline.

    Returns: pd.DataFrame with columns ["row_id"] + PRIMARY_LABELS
    """
    t_start = time.time()

    # Find SED weights
    sed_weights_dir = _find_input("birdclef-2026-hgnet-b0-sed-v5j-4fold-pp2-weights")
    if sed_weights_dir is None or not os.path.exists(sed_weights_dir):
        raise FileNotFoundError(
            "SED weights not found at birdclef-2026-hgnet-b0-sed-v5j-4fold-pp2-weights"
        )
    sed_weights_dir = Path(sed_weights_dir)
    print(f"SED weights dir: {sed_weights_dir}")

    # Load configs
    with open(sed_weights_dir / "mel_config.json") as f:
        mel_cfg = json.load(f)
    with open(sed_weights_dir / "species_mapping.json") as f:
        species_mapping = json.load(f)

    taxa_mapping = None
    taxa_path = sed_weights_dir / "taxa_mapping.json"
    if taxa_path.exists():
        with open(taxa_path) as f:
            taxa_mapping = json.load(f)

    metadata_priors = None
    priors_path = sed_weights_dir / "metadata_priors.json"
    if priors_path.exists():
        with open(priors_path) as f:
            metadata_priors = json.load(f)

    mel_type = mel_cfg.get("type", "mel")
    print(f"SED mel type: {mel_type}, SR: {mel_cfg['sr']}")

    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]
    print(f"SED species mapping: {len(idx_map)} mapped")

    if mel_cfg.get("temperature", 1.0) != 1.0:
        print(f"SED temperature: T={mel_cfg['temperature']}")

    # Create 4 ONNX sessions
    fold_onnx_paths = sorted(sed_weights_dir.glob("model_fold*.onnx"))
    if len(fold_onnx_paths) == 0:
        # Fallback to single model
        single_model = sed_weights_dir / "model.onnx"
        if single_model.exists():
            fold_onnx_paths = [single_model]
        else:
            raise FileNotFoundError(f"No ONNX models found in {sed_weights_dir}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = os.cpu_count() or 4
    sess_opts.inter_op_num_threads = 1

    sessions = []
    for fold_path in fold_onnx_paths:
        s = ort.InferenceSession(
            str(fold_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        sessions.append(s)
    print(f"SED: {len(sessions)} ONNX sessions created ({time.time() - t_start:.1f}s)")

    # Run inference
    all_predictions = []
    for i, audio_path in enumerate(test_paths):
        t0 = time.time()
        preds = sed_predict_soundscape_ensemble(
            sessions,
            str(audio_path),
            mel_cfg,
            species_mapping,
            taxa_mapping=taxa_mapping,
            metadata_priors=metadata_priors,
        )
        elapsed = time.time() - t0
        if i < 3 or i % 10 == 0:
            print(
                f"  SED [{i+1}/{len(test_paths)}] {audio_path.name}: {len(preds)} rows ({elapsed:.1f}s)"
            )
        all_predictions.extend(preds)

    print(f"SED total predictions: {len(all_predictions)} rows")

    # Build submission DataFrame
    species_2026 = species_mapping["birdclef2026_species"]

    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
    else:
        pred_df = pd.DataFrame(columns=["row_id"] + species_2026)

    # Ensure all PRIMARY_LABELS columns exist
    for col in PRIMARY_LABELS:
        if col not in pred_df.columns:
            pred_df[col] = 0.0

    # Reorder to match expected format
    submission = pred_df[["row_id"] + PRIMARY_LABELS].copy()
    for col in PRIMARY_LABELS:
        submission[col] = submission[col].fillna(0.0).astype(np.float32)

    print(f"SED pipeline complete: {submission.shape}")

    # Cleanup
    del sessions
    gc.collect()

    return submission


# ============================================================================
# SECTION 5: Ensemble + Submission
# ============================================================================


def main():
    wall_start = time.time()

    # Discover test files
    test_dir = BASE / "test_soundscapes"
    test_paths = sorted(test_dir.glob("*.ogg"))
    if len(test_paths) == 0:
        train_dir = BASE / "train_soundscapes"
        test_paths = sorted(train_dir.glob("*.ogg"))[:20]
        print(f"No test soundscapes, dry-run on {len(test_paths)} train files")
    else:
        print(f"Processing {len(test_paths)} test soundscapes")

    # Pipeline 1: Perch ONNX + ProtoSSM
    t0 = time.time()
    perch_df = run_perch_pipeline(test_paths)
    print(f"Perch pipeline: {time.time()-t0:.1f}s")

    # Pipeline 2: SED v5j-4fold
    t0 = time.time()
    sed_df = run_sed_pipeline(test_paths)
    print(f"SED pipeline: {time.time()-t0:.1f}s")

    # Ensemble: weighted average
    assert set(perch_df["row_id"]) == set(sed_df["row_id"]), "Row ID mismatch!"
    sed_aligned = sed_df.set_index("row_id").loc[perch_df["row_id"]].reset_index()

    perch_probs = perch_df[PRIMARY_LABELS].values.astype(np.float32)
    sed_probs = sed_aligned[PRIMARY_LABELS].values.astype(np.float32)
    final_probs = np.clip(W_PERCH * perch_probs + (1.0 - W_PERCH) * sed_probs, 0.0, 1.0)

    # Build submission
    submission = pd.DataFrame(final_probs, columns=PRIMARY_LABELS)
    submission.insert(0, "row_id", perch_df["row_id"].values)

    sample_sub_full = pd.read_csv(BASE / "sample_submission.csv")
    submission = sample_sub_full[["row_id"]].merge(submission, on="row_id", how="left")
    for col in PRIMARY_LABELS:
        submission[col] = submission[col].fillna(0.0)

    submission.to_csv("/kaggle/working/submission.csv", index=False)

    print(f"\nEnsemble complete: W_PERCH={W_PERCH}")
    print(f"Shape: {submission.shape}, range: [{final_probs.min():.6f}, {final_probs.max():.6f}]")
    print(f"Total: {time.time()-wall_start:.1f}s ({(time.time()-wall_start)/60:.1f}min)")


if __name__ == "__main__":
    main()
