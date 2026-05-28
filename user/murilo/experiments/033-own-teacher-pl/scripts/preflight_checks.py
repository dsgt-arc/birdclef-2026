"""Pre-flight diagnostic checks for exp 033 (own-teacher PL on Tucker recipe).

Runs read-only checks before any training. Surfaces issues so we don't
discover them mid-run.

Defaults to validating the v1 PL parquet (T1, v5j-4fold teacher). Use
`--parquet-path` to validate a different parquet (e.g., v2 T2 after
Phase 0.2 generation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml

REPO = Path("/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026")
DATA = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
DEFAULT_PL_PATH = Path(
    "/storage/project/ps-clef2026_birdclef-0/shared/pseudo-labels/birdclef-2026/v1/sed_pseudo_labels.parquet"
)
INF_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/inference/birdclef-2026")
CKPT_BASE = Path(
    "/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/032-tucker-distilled-sed/checkpoints"
)
SR = 32000
WINDOW_SAMPLES = 160000

# ----------------------------------------------------------------------
# Argparse — supports overriding the PL parquet for v2 validation
# ----------------------------------------------------------------------
_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--parquet-path",
    type=Path,
    default=DEFAULT_PL_PATH,
    help=f"Path to the PL parquet to validate. Default: {DEFAULT_PL_PATH}",
)
_args = _parser.parse_args()
PL_PATH = _args.parquet_path
print(f"Validating PL parquet: {PL_PATH}")
if not PL_PATH.exists():
    print(f"FATAL: parquet not found at {PL_PATH}")
    sys.exit(1)


def banner(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\nCHECK {n}: {title}\n{'=' * 78}")


def report(passed: bool, msg: str) -> None:
    print(f"  {'PASS ✅' if passed else 'FAIL ❌'}  {msg}")


# ----------------------------------------------------------------------
# CHECK 1: PL parquet filename format matches train_soundscapes/*.ogg
# ----------------------------------------------------------------------
banner(1, "PL parquet filenames match train_soundscapes/ files on disk")
pl_df = pd.read_parquet(PL_PATH)
ss_dir = DATA / "train_soundscapes"
on_disk = {f.name for f in ss_dir.glob("*.ogg")}
in_parquet = set(pl_df["filename"].unique())
extra_parquet = in_parquet - on_disk
extra_disk = on_disk - in_parquet
print(f"  PL parquet unique files: {len(in_parquet):,}")
print(f"  train_soundscapes/ .ogg files: {len(on_disk):,}")
print(f"  In parquet but missing on disk: {len(extra_parquet)}")
print(f"  On disk but missing from parquet: {len(extra_disk)}")
if extra_parquet:
    print(f"  Examples (in parquet, missing on disk): {sorted(extra_parquet)[:3]}")
report(len(extra_parquet) == 0, "all PL parquet filenames resolvable to .ogg files on disk")

# ----------------------------------------------------------------------
# CHECK 2: Load 5 random unlabeled SS audio chunks; verify length
# ----------------------------------------------------------------------
banner(2, "Audio chunks decode correctly + match 160000-sample window")
rng = np.random.default_rng(seed=42)
sample = pl_df.sample(n=5, random_state=42)
all_ok = True
for _, row in sample.iterrows():
    audio_path = ss_dir / row["filename"]
    chunk_idx = int(row["chunk_idx"])
    start_sec = chunk_idx * 5.0
    try:
        audio, src_sr = sf.read(
            str(audio_path),
            dtype="float32",
            start=int(start_sec * SR),
            frames=WINDOW_SAMPLES,
        )
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        ok = src_sr == SR and len(audio) >= WINDOW_SAMPLES - 2
        if not ok:
            all_ok = False
            print(f"  ISSUE {audio_path.name} chunk={chunk_idx}: sr={src_sr} len={len(audio)}")
        else:
            print(f"  OK  {audio_path.name[:50]} chunk={chunk_idx} sr={src_sr} len={len(audio)}")
    except Exception as e:
        all_ok = False
        print(f"  ERROR {audio_path.name} chunk={chunk_idx}: {e}")
report(all_ok, "5 random PL audio chunks decode at 32kHz with full window length")

# ----------------------------------------------------------------------
# CHECK 3: TuckerMelSpectrogram is the SAME for focal/SS/unlabeled-SS paths
# ----------------------------------------------------------------------
banner(3, "TuckerMelSpectrogram is path-invariant (focal vs SS audio waveform)")
sys.path.insert(0, str(REPO / "user/murilo/src"))
from birdclef.preprocessing.tucker_mel import TuckerMelSpectrogram  # noqa: E402

# Read training mel config from Tucker recipe
with open(
    REPO / "user/murilo/experiments/032-tucker-distilled-sed/configs/distilled_sed_kd.yaml"
) as f:
    cfg = yaml.safe_load(f)
mel_cfg = cfg["mel"]
data_cfg = cfg["data"]
sample_rate = data_cfg["sample_rate"]

mel = TuckerMelSpectrogram(
    sample_rate=sample_rate,
    n_fft=mel_cfg["n_fft"],
    hop_length=mel_cfg["hop_length"],
    n_mels=mel_cfg["n_mels"],
    fmin=mel_cfg["fmin"],
    fmax=mel_cfg["fmax"],
    top_db=mel_cfg["top_db"],
    power=mel_cfg.get("power", 2.0),
)
mel.eval()

# Same waveform fed through twice → should be deterministic
audio_path = ss_dir / sample.iloc[0]["filename"]
audio_np, _ = sf.read(str(audio_path), dtype="float32", start=0, frames=WINDOW_SAMPLES)
if audio_np.ndim == 2:
    audio_np = audio_np.mean(axis=1)
wave = torch.from_numpy(audio_np).float().unsqueeze(0)
with torch.no_grad():
    m1 = mel(wave)
    m2 = mel(wave)
diff = (m1 - m2).abs().max().item()
print(f"  Same waveform → mel diff: {diff:.2e} (expected 0)")
report(diff < 1e-7, "TuckerMelSpec is deterministic for identical input")

# Mel shape should match what AttHead expects: (B, 1, n_mels, T)
print(f"  Mel output shape: {tuple(m1.shape)} (expected (B=1, C=1, n_mels=256, T))")
expected_t = WINDOW_SAMPLES // mel_cfg["hop_length"] + 1
shape_ok = (
    m1.ndim == 4
    and m1.shape[0] == 1
    and m1.shape[1] == 1
    and m1.shape[2] == mel_cfg["n_mels"]
    and abs(m1.shape[3] - expected_t) <= 2
)
report(shape_ok, f"Mel shape matches (B=1, C=1, n_mels={mel_cfg['n_mels']}, T≈{expected_t})")

# ----------------------------------------------------------------------
# CHECK 4: PL parquet values ∈ [0,1] + (filename, chunk_idx) uniqueness
# ----------------------------------------------------------------------
banner(4, "PL parquet schema integrity (value range + key uniqueness)")
species_cols = [c for c in pl_df.columns if c not in ("filename", "chunk_idx")]
vals = pl_df[species_cols].values
min_v = vals.min()
max_v = vals.max()
print(f"  species columns: {len(species_cols)}")
print(f"  value range: [{min_v:.4f}, {max_v:.4f}] (expected [0, 1])")
report(min_v >= 0.0 and max_v <= 1.0, "all PL values in [0, 1]")

# Uniqueness of (filename, chunk_idx)
dup_count = pl_df.duplicated(subset=["filename", "chunk_idx"]).sum()
report(dup_count == 0, f"(filename, chunk_idx) is unique (duplicates: {dup_count})")

# Chunk_idx range check (should be 0..11)
ch_min = pl_df["chunk_idx"].min()
ch_max = pl_df["chunk_idx"].max()
print(f"  chunk_idx range: [{ch_min}, {ch_max}] (expected [0, 11])")
report(ch_min == 0 and ch_max == 11, "chunk_idx covers 12 windows per file")

# ----------------------------------------------------------------------
# CHECK 5: Fold-0 SS files appear in PL parquet (can be filtered);
#          inspect labeled vs unlabeled SS coverage
# ----------------------------------------------------------------------
banner(5, "Fold leakage prevention is structurally possible")
# Recompute exp 032 fold 0 SS file set
from birdclef.data.distilled_sed_datamodule import (  # noqa: E402
    build_sc_cache_meta,
    build_sc_folds,
)

labels_csv = DATA / "train_soundscapes_labels.csv"
sc_labels = pd.read_csv(labels_csv)
labeled_files = sorted(sc_labels["filename"].unique().tolist())
print(f"  labeled SS files (have ground-truth labels): {len(labeled_files)}")

sc_meta = build_sc_cache_meta(labeled_files)
sc_folded = build_sc_folds(sc_meta, n_folds=5)
fold0_files = set(sc_folded[sc_folded["fold"] == 0]["filename"].unique())
fold1_4_files = set(sc_folded[sc_folded["fold"] != 0]["filename"].unique())
print(f"  fold 0 SS files (val for distilled-sed-kd Phase D): {len(fold0_files)}")
print(f"  fold 1-4 SS files (train): {len(fold1_4_files)}")

# Find these files in PL parquet
pl_fold0 = pl_df[pl_df["filename"].isin(fold0_files)]
pl_fold1_4 = pl_df[pl_df["filename"].isin(fold1_4_files)]
print(
    f"  fold-0 SS in PL parquet: {len(pl_fold0):,} segments ({pl_fold0['filename'].nunique()} files)"
)
print(f"  fold-1-4 SS in PL parquet: {len(pl_fold1_4):,} segments")
report(
    len(pl_fold0) == len(fold0_files) * 12 and len(pl_fold1_4) == len(fold1_4_files) * 12,
    "labeled SS files have full 12-window PL coverage in parquet",
)

# Unlabeled SS in PL parquet
unlabeled_in_pl = pl_df[~pl_df["filename"].isin(set(labeled_files))]
print(
    f"  unlabeled SS in PL parquet: {len(unlabeled_in_pl):,} segments ({unlabeled_in_pl['filename'].nunique()} files)"
)
ss_files_on_disk = on_disk
unlabeled_on_disk = ss_files_on_disk - set(labeled_files)
print(f"  unlabeled SS .ogg files on disk: {len(unlabeled_on_disk)}")
report(
    unlabeled_in_pl["filename"].nunique() == len(unlabeled_on_disk),
    "all unlabeled SS files are in the PL parquet",
)

# ----------------------------------------------------------------------
# CHECK 6: PL parquet species cols match taxonomy.csv ordering
# ----------------------------------------------------------------------
banner(6, "PL species column ordering matches taxonomy.csv")
tax = pd.read_csv(DATA / "taxonomy.csv")
tax_species = tax["primary_label"].astype(str).tolist()
print(f"  taxonomy.csv species: {len(tax_species)}")
print(f"  PL parquet species cols: {len(species_cols)}")

# Both lists should contain the same set
tax_set = set(tax_species)
pl_set = set(species_cols)
only_tax = tax_set - pl_set
only_pl = pl_set - tax_set
print(f"  In taxonomy only: {len(only_tax)} {(sorted(only_tax)[:5] if only_tax else '')}")
print(f"  In PL only: {len(only_pl)} {(sorted(only_pl)[:5] if only_pl else '')}")
report(len(only_tax) == 0 and len(only_pl) == 0, "PL species set == taxonomy.csv species set")

# Order match
same_order = tax_species == species_cols
print(f"  Column orderings identical: {same_order}")
report(
    same_order or len(only_tax) == 0,
    "PL columns can be re-indexed to taxonomy order (set match is sufficient)",
)

# ----------------------------------------------------------------------
# CHECK 7: Sanity inference — Phase D ckpt's top-k ≈ PL parquet's top-k
# ----------------------------------------------------------------------
banner(7, "Phase D ckpt inference agrees with cached PL (sanity check)")
print("  Loading Phase D distilled-sed-kd ckpt...")
ckpt_path = CKPT_BASE / "2026-exp032-distilled-sed-kd/last-v2.ckpt"
report(ckpt_path.exists(), f"checkpoint exists at {ckpt_path}")

if ckpt_path.exists():
    from birdclef.encoders.efficientnet import EfficientNetEncoder
    from birdclef.heads.sed_head import AttHeadLogitSpace
    from birdclef.lightning.distilled_sed_finetune import DistilledSEDFinetuneModule

    encoder = EfficientNetEncoder(
        model_name=cfg["encoder"]["model_name"],
        pretrained=False,
        drop_path_rate=0.0,
        in_chans=cfg["encoder"].get("in_chans", 1),
        use_post_head=cfg["encoder"].get("use_post_head", True),
    )
    head = AttHeadLogitSpace(
        in_channels=encoder.embed_dim,
        num_classes=234,
        hidden_dim=cfg["sed_head"]["hidden_dim"],
        dropout=cfg["sed_head"]["dropout"],
        gem_p=cfg["sed_head"].get("gem_p", 3.0),
    )
    module = DistilledSEDFinetuneModule(
        encoder=encoder,
        sed_head=head,
        num_classes=234,
        mel_transform=mel,
        spec_augment=None,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        max_epochs=cfg["training"]["max_epochs"],
        warmup_epochs=cfg["training"]["warmup_epochs"],
        use_perch_distill=False,
        perch_onnx_path=None,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    missing, unexpected = module.load_state_dict(sd, strict=False)
    missing_real = [k for k in missing if not any(m in k for m in ("cmap", "auroc", "f1", "top"))]
    unexpected_real = [k for k in unexpected if not k.startswith("distill_head")]
    print(f"  missing keys: {len(missing_real)}, unexpected keys: {len(unexpected_real)}")
    report(
        len(missing_real) == 0 and len(unexpected_real) == 0,
        "Phase D ckpt loads cleanly into Tucker module",
    )

    # Pick a labeled SS file with strong positive labels for a clear signal
    fn = labeled_files[0]
    chunk_idx = 0
    file_pl = pl_df[(pl_df["filename"] == fn) & (pl_df["chunk_idx"] == chunk_idx)]
    if len(file_pl) == 0:
        print(f"  SKIP — no PL row for {fn} chunk {chunk_idx}")
    else:
        pl_row = file_pl.iloc[0]
        pl_top5 = pl_row[species_cols].sort_values(ascending=False).head(5)
        print(f"\n  Sanity file: {fn} chunk={chunk_idx} (PL teacher = v5j-4fold)")
        print("  Cached PL top-5 species:")
        for sp, prob in pl_top5.items():
            print(f"    {sp:15s}  {prob:.4f}")

        # Run Phase D ckpt forward
        audio_path = ss_dir / fn
        audio_np, _ = sf.read(str(audio_path), dtype="float32", start=0, frames=WINDOW_SAMPLES)
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)
        wave = torch.from_numpy(audio_np).float().unsqueeze(0)
        module.eval()
        with torch.no_grad():
            mel_t = module._compute_mel(wave, training=False)
            features = module.encoder.forward_features(mel_t)
            clip_logits, frame_logits = module.sed_head(features)
            frame_max_logits = frame_logits.max(dim=-1).values
            blended = 0.5 * clip_logits + 0.5 * frame_max_logits
            probs = torch.sigmoid(blended).squeeze(0).numpy()

        # Phase D ckpt top-5
        sp_arr = np.array(species_cols)
        top5_idx = np.argsort(probs)[::-1][:5]
        print("  Phase D ckpt top-5 species (single-window, no smoothing):")
        for i in top5_idx:
            print(f"    {sp_arr[i]:15s}  {probs[i]:.4f}")

        # Compare: do top-1s agree?
        pl_top1 = pl_top5.index[0]
        ckpt_top1 = sp_arr[top5_idx[0]]
        agree = pl_top1 == ckpt_top1
        print(f"\n  Top-1 PL ({pl_top1}) vs Phase D ckpt ({ckpt_top1}) → agree: {agree}")
        # PL was generated by v5j-4fold, Phase D ckpt is EffNet+KD; agreement not required
        # but extreme disagreement would suggest a wiring issue.
        pl_top1_idx = list(species_cols).index(pl_top1)
        ckpt_rank_of_pl_top1 = np.argsort(probs)[::-1].tolist().index(pl_top1_idx)
        print(f"  Phase D ckpt's rank for PL's top-1 species: #{ckpt_rank_of_pl_top1 + 1}")
        report(
            ckpt_rank_of_pl_top1 < 20,
            "Phase D ckpt ranks PL's top species in top-20 (semantic agreement, no exact match required)",
        )


print(f"\n{'=' * 78}")
print("PRE-FLIGHT CHECKS COMPLETE")
print("=" * 78)
