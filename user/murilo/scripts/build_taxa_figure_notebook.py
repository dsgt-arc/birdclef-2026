"""Scaffold the taxa spectrogram/band-occupancy notebook (provenance record).

This script *generated* `user/murilo/notebooks/006-taxa-spectrogram-figure.ipynb` (the
BirdCLEF-2026 working-note figures). It is kept for reproducibility: it documents exactly
how the notebook was built.

IMPORTANT: the **notebook is the canonical, editable source going forward**. Re-running
this script OVERWRITES the notebook in place, discarding any edits made directly in Jupyter.
Only re-run it to rebuild the notebook from scratch.

Usage (from anywhere in the repo, with the project venv active):
    python user/murilo/scripts/build_taxa_figure_notebook.py
Then execute the notebook to (re)generate assets + figures:
    jupyter nbconvert --to notebook --execute --inplace \\
        user/murilo/notebooks/006-taxa-spectrogram-figure.ipynb
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md(r"""# Taxa frequency & structure figures — BirdCLEF 2026

**Finding (data-driven):** the three vocalizing taxa in BirdCLEF 2026 do **not** occupy
distinct frequency bands — they overlap heavily in **1-4 kHz** (per-recording peak-freq
medians: frogs 1.8, birds 2.9, insects 1.8 kHz; insects are low-mid **cicadas**, not
high-pitched crickets). What separates them is **spectro-temporal structure**: frogs =
pulsed trains, birds = FM whistles / harmonic sweeps, insects = sustained broadband drone.
This motivates spectrogram CNNs over hand-crafted frequency-band features.

- **Figure 1** (`figures/taxa_spectrograms.{pdf,png}`, main): 3 representative spectrograms,
  side-by-side, shared linear-Hz axis (0-16 kHz = 32 kHz Nyquist) — the *structure* contrast.
- **Figure 2** (`figures/band_occupancy.{pdf,png}`, supporting): per-taxon mean Welch
  spectrum + per-recording peak-frequency violins — the *band overlap*.

**Notebook structure:** the *asset-extraction* cell is the only part that touches the raw
dataset (PACE-only). It persists clips + PSD/peak-freq arrays under `notebooks/assets/`, and
**auto-skips when those assets already exist** -> the plotting cells re-run off-cluster from
the committed assets, so styling/framing can be refined without the cluster.""")

code(r"""import json
from pathlib import Path

import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch

# ---- reproducibility & print style ----
RNG_SEED = 51
SR  = 32000          # all BirdCLEF-2026 audio is 32 kHz mono OGG
NYQ = SR // 2        # 16 kHz
CLIP_SEC = 5.0       # spectrogram panel length

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 400, "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ---- repo-relative paths (robust to cwd) ----
def find_repo_root(start: Path) -> Path:
    for q in [start.resolve(), *start.resolve().parents]:
        if (q / "CLAUDE.md").exists() and (q / "user").exists():
            return q
    return start.resolve()

REPO   = find_repo_root(Path.cwd())
ASSETS = REPO / "user/murilo/notebooks/assets"
CLIPS  = ASSETS / "clips"
FIGDIR = REPO / "user/murilo/figures"
for d in (ASSETS, CLIPS, FIGDIR):
    d.mkdir(parents=True, exist_ok=True)

MANIFEST = ASSETS / "clip_manifest.json"
PSD_NPZ  = ASSETS / "band_occupancy.npz"
FREQ_NPZ = ASSETS / "freq_stats.npz"

# Raw dataset = PACE-only. Off-cluster this path won't exist -> extraction auto-skips.
DATA_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")

# figure order (matches taxon group columns left->right)
TAXA = ["Amphibia", "Aves", "Insecta"]
TAXON_LABEL = {"Amphibia": "Frogs (Amphibia)", "Aves": "Birds (Aves)", "Insecta": "Insects (Insecta)"}
TAXON_COLOR = {"Amphibia": "#1f77b4", "Aves": "#2ca02c", "Insecta": "#d62728"}
# spectro-temporal structure descriptor shown in the Fig 1 panel titles
STRUCTURE   = {"Amphibia": "pulsed train", "Aves": "FM whistle", "Insecta": "sustained drone"}

print("REPO        :", REPO)
print("DATA on disk:", DATA_DIR.exists())""")

md(r"""## 1. Asset extraction (PACE-only; auto-skips if assets already present)

Reads `train.csv` directly (it already carries `class_name`, `scientific_name`,
`common_name`, `rating`, `filename`). For each taxon it saves the top `N_CANDIDATES`
distinct-species, highest-rated recordings, each trimmed to the loudest `CLIP_SEC` window
(an RMS-based proxy for "the vocalization is here, not silence"). It also computes a
per-taxon mean Welch PSD over `N_PSD_FILES` randomly sampled recordings.""")

code(r'''N_CANDIDATES  = 6      # candidate clips saved per taxon (pick the best one below)
N_PSD_FILES   = 80     # files sampled per taxon for the PSD curve
PSD_MAX_SEC   = 30.0   # seconds analysed per file for the PSD
FORCE_EXTRACT = False  # set True to re-extract even if assets exist


def _load(fp, max_sec=None):
    y, _ = librosa.load(fp, sr=SR, mono=True, duration=max_sec)
    return y


def _best_window(y, sr=SR, clip_sec=CLIP_SEC, hop_sec=0.5):
    """Return the clip_sec window with the highest mean RMS (loudest vocalization)."""
    n = int(clip_sec * sr)
    if len(y) <= n:
        return np.pad(y, (0, n - len(y)))
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_per_sec = sr / 512.0
    win = max(1, int(clip_sec * rms_per_sec))
    hop = max(1, int(hop_sec * rms_per_sec))
    best_i, best_e = 0, -1.0
    for i in range(0, len(rms) - win + 1, hop):
        e = float(rms[i:i + win].mean())
        if e > best_e:
            best_e, best_i = e, i
    start = min(int(best_i * 512), len(y) - n)
    return y[start:start + n]


def extract_clips(train):
    manifest = {}
    for taxon in TAXA:
        sub = train[train.class_name == taxon].copy()
        sub["rating"] = pd.to_numeric(sub.get("rating", 0), errors="coerce").fillna(0.0)
        sub = sub.sort_values("rating", ascending=False)
        picks, seen = [], set()
        for _, r in sub.iterrows():
            key = str(r.primary_label)
            if key in seen:
                continue
            seen.add(key)
            picks.append(r)
            if len(picks) >= N_CANDIDATES:
                break
        outdir = CLIPS / taxon
        outdir.mkdir(parents=True, exist_ok=True)
        cand = []
        for k, r in enumerate(picks):
            fp = DATA_DIR / "train_audio" / r.filename
            try:
                clip = _best_window(_load(fp, max_sec=60.0))
                outfp = outdir / f"cand{k}_{r.primary_label}.flac"
                try:
                    sf.write(outfp, clip, SR, format="FLAC")
                except Exception:
                    outfp = outfp.with_suffix(".wav")
                    sf.write(outfp, clip, SR)
                cand.append({
                    "path": str(outfp.relative_to(REPO)),
                    "primary_label": str(r.primary_label),
                    "scientific_name": str(r.scientific_name),
                    "common_name": str(r.common_name),
                    "source_file": str(r.filename),
                    "rating": float(r.rating),
                })
            except Exception as e:
                print("  skip clip", fp, "->", e)
        manifest[taxon] = cand
        print(f"{taxon:9s}: {len(cand)} candidate clips")
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def compute_band_occupancy(train):
    rng = np.random.default_rng(RNG_SEED)
    nperseg = 2048
    out, freq = {}, None
    for taxon in TAXA:
        files = train[train.class_name == taxon].filename.tolist()
        rng.shuffle(files)
        acc, used = None, 0
        for fn in files[: N_PSD_FILES * 2]:   # oversample; some may fail/short
            if used >= N_PSD_FILES:
                break
            try:
                y = _load(DATA_DIR / "train_audio" / fn, max_sec=PSD_MAX_SEC)
                if len(y) < nperseg:
                    continue
                f, p = welch(y, fs=SR, nperseg=nperseg)
                p = p / (p.sum() + 1e-12)      # normalize per file: compare SHAPE, not loudness
                acc = p if acc is None else acc + p
                freq = f
                used += 1
            except Exception as e:
                print("  psd skip", fn, "->", e)
        out[taxon] = acc / max(used, 1)
        print(f"{taxon:9s}: PSD over {used} files")
    np.savez(PSD_NPZ, freq=freq, **{t: out[t] for t in TAXA})


def compute_freq_stats(train):
    """Per-recording peak frequency + spectral centroid (above 300 Hz) per taxon."""
    rng = np.random.default_rng(RNG_SEED + 1)
    nperseg, fmin = 4096, 300.0
    stats = {}
    for taxon in TAXA:
        files = train[train.class_name == taxon].filename.tolist()
        rng.shuffle(files)
        peaks, cents = [], []
        for fn in files[: N_PSD_FILES * 2]:
            if len(peaks) >= N_PSD_FILES:
                break
            try:
                y = _load(DATA_DIR / "train_audio" / fn, max_sec=PSD_MAX_SEC)
                if len(y) < nperseg:
                    continue
                f, p = welch(y, fs=SR, nperseg=nperseg)
                band = f >= fmin
                fb, pb = f[band], p[band]
                if pb.sum() <= 0:
                    continue
                peaks.append(float(fb[np.argmax(pb)]))
                cents.append(float((fb * pb).sum() / pb.sum()))
            except Exception as e:
                print("  stat skip", fn, "->", e)
        stats[f"peak_{taxon}"] = np.asarray(peaks)
        stats[f"cent_{taxon}"] = np.asarray(cents)
        print(f"{taxon:9s}: freq-stats over {len(peaks)} files "
              f"(peak med {np.median(peaks)/1000:.1f} kHz, centroid med {np.median(cents)/1000:.1f} kHz)")
    np.savez(FREQ_NPZ, **stats)


need_clips = FORCE_EXTRACT or not MANIFEST.exists()
need_psd   = FORCE_EXTRACT or not PSD_NPZ.exists()
need_stats = FORCE_EXTRACT or not FREQ_NPZ.exists()
if (need_clips or need_psd or need_stats) and DATA_DIR.exists():
    train = pd.read_csv(DATA_DIR / "train.csv")
    if need_clips:
        print("extracting clips ...")
        extract_clips(train)
    if need_psd:
        print("computing band occupancy ...")
        compute_band_occupancy(train)
    if need_stats:
        print("computing freq stats ...")
        compute_freq_stats(train)
    print("DONE -> assets persisted under", ASSETS)
elif (need_clips or need_psd or need_stats):
    print("Assets missing AND raw dataset unavailable (off-cluster). "
          "Run this notebook once on PACE and commit notebooks/assets/.")
else:
    print("All assets present -> extraction skipped (off-cluster refinement mode).")''')

md(r"""## 2. Load persisted assets + plotting helpers""")

code(r"""manifest = json.loads(MANIFEST.read_text())
psd = np.load(PSD_NPZ)

# Chosen exemplar index per taxon (picked from the contact sheets below for clear structure:
# pulsed frog train / FM bird whistle / sustained cicada drone).
# Override e.g. CHOSEN["Insecta"] = 0 and re-run Fig 1 to swap exemplars.
CHOSEN = {"Amphibia": 2, "Aves": 1, "Insecta": 1}

N_FFT, HOP, DB_FLOOR = 1024, 256, -80

def spec_db(y):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    return librosa.amplitude_to_db(S, ref=np.max)

def load_clip(rec):
    y, _ = librosa.load(REPO / rec["path"], sr=SR, mono=True)
    return y

print({t: len(manifest[t]) for t in TAXA}, "candidates/taxon")""")

md(r"""## 3. Contact sheets — inspect candidates, then set `CHOSEN`

Each row shows the `N_CANDIDATES` clips for one taxon. Note the index `[k]` of the
cleanest / most representative clip and set `CHOSEN[taxon] = k` in the cell above.""")

code(r"""for taxon in TAXA:
    cs = manifest[taxon]
    n = len(cs)
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 2.1), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for k, (ax, c) in enumerate(zip(axes, cs, strict=False)):
        y = load_clip(c)
        ax.imshow(spec_db(y), origin="lower", aspect="auto", cmap="magma",
                  vmin=DB_FLOOR, vmax=0, extent=[0, len(y) / SR, 0, NYQ / 1000])
        ax.set_title(f"[{k}] {c['scientific_name']}\n(r={c['rating']:.0f})", fontsize=7)
        ax.set_xlabel("s")
        if k == 0:
            ax.set_ylabel("kHz")
    fig.suptitle(f"{taxon} candidates (set CHOSEN['{taxon}'], currently {CHOSEN[taxon]})", fontsize=9)
    fig.savefig(FIGDIR / f"contactsheet_{taxon}.png", dpi=150)
    plt.show()""")

md(r"""## Figure 1 (main) — taxa spectrograms: the structure contrast

Shared linear-Hz axis (0-16 kHz). Panel titles name the spectro-temporal structure that
discriminates the taxa (frequency band alone does not — see Fig 2).""")

code(r"""fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6), sharey=True, constrained_layout=True)
im = None
for ax, taxon in zip(axes, TAXA, strict=False):
    c = manifest[taxon][CHOSEN[taxon]]
    y = load_clip(c)
    dur = len(y) / SR
    im = ax.imshow(spec_db(y), origin="lower", aspect="auto", cmap="magma",
                   vmin=DB_FLOOR, vmax=0, extent=[0, dur, 0, NYQ / 1000])
    sci = c["scientific_name"].replace(" ", r"\ ")        # mathtext space
    ax.set_title(f"{TAXON_LABEL[taxon]} · {STRUCTURE[taxon]}\n" + r"$\mathit{" + sci + r"}$",
                 fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, dur)
    ax.set_ylim(0, NYQ / 1000)
axes[0].set_ylabel("Frequency (kHz)")
cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.015)
cbar.set_label("Power (dB)")
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"taxa_spectrograms.{ext}")
print("saved", FIGDIR / "taxa_spectrograms.pdf")
plt.show()""")

md(r"""## Figure 2 — band occupancy (quantitative)

Two complementary views over many recordings (random sample per taxon):
- **Left:** mean Welch spectrum, per-file energy-normalized (sum=1) so curves compare
  *where* energy sits, not loudness; lightly smoothed.
- **Right:** per-recording **peak frequency** distribution (violin) — the clearest view of
  the central tendency + overlap of each taxon's band.""")

code(r"""SHORT = {"Amphibia": "Frogs", "Aves": "Birds", "Insecta": "Insects"}
fs = np.load(FREQ_NPZ)

def smooth(x, k=9):
    return np.convolve(x, np.ones(k) / k, mode="same")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.8, 2.7), constrained_layout=True)

# left: smoothed mean spectrum
freq = psd["freq"] / 1000.0
for taxon in TAXA:
    pdb = 10.0 * np.log10(psd[taxon] + 1e-12)
    axL.plot(freq, smooth(pdb), color=TAXON_COLOR[taxon], lw=1.7, label=TAXON_LABEL[taxon])
axL.set_xlim(0, NYQ / 1000)
axL.set_xlabel("Frequency (kHz)")
axL.set_ylabel("Relative power (dB)")
axL.set_title("Mean spectrum (energy-normalized)", fontsize=8)
axL.grid(alpha=0.25)
axL.legend(frameon=False, fontsize=7)

# right: per-recording peak-frequency violin
data = [fs[f"peak_{t}"] / 1000.0 for t in TAXA]
parts = axR.violinplot(data, showmedians=True, widths=0.85)
for pc, t in zip(parts["bodies"], TAXA, strict=False):
    pc.set_facecolor(TAXON_COLOR[t])
    pc.set_edgecolor("0.3")
    pc.set_alpha(0.65)
for key in ("cmedians", "cmins", "cmaxes", "cbars"):
    if key in parts:
        parts[key].set_color("0.3")
        parts[key].set_linewidth(1.0)
axR.set_xticks([1, 2, 3])
axR.set_xticklabels([SHORT[t] for t in TAXA])
axR.set_ylim(0, NYQ / 1000)
axR.set_ylabel("Peak frequency (kHz)")
axR.set_title("Per-recording peak frequency", fontsize=8)
axR.grid(alpha=0.25, axis="y")

for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"band_occupancy.{ext}")
print("saved", FIGDIR / "band_occupancy.pdf")
plt.show()""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
# repo-relative: scripts/ -> ../notebooks/  (robust regardless of cwd)
out = Path(__file__).resolve().parents[1] / "notebooks" / "006-taxa-spectrogram-figure.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out, "with", len(cells), "cells")
