"""Interactive model attention/attribution explorer.

Marimo app for visualizing gradient saliency, per-frame logits, and LSE
effective attention on individual audio samples. Browse by loss rank to
understand WHERE and WHY the model fails.

Usage:
    # Edit mode (see and modify cells):
    marimo edit model_explorer.py --host 0.0.0.0 --port 8080

    # App mode (clean dashboard):
    marimo run model_explorer.py --host 0.0.0.0 --port 8080

    # Then SSH port forward from local machine:
    ssh -L 8080:localhost:8080 user@pace-cluster
"""

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="full")


@app.cell
def imports():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Add src and analysis dirs to path
    _analysis_dir = Path(__file__).parent
    _user_dir = _analysis_dir.parent
    sys.path.insert(0, str(_user_dir / "src"))
    sys.path.insert(0, str(_analysis_dir))

    return Path, mo, np, pd, plt, sys


@app.cell
def config(mo, Path):
    """Configuration — edit these paths for your setup."""
    _analysis_dir = Path(__file__).parent
    _user_dir = _analysis_dir.parent
    _project_dir = Path("/storage/project/ps-clef2026_birdclef-0/mgustineli3")

    CONFIG_PATH = str(_user_dir / "experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml")
    CHECKPOINT_PATH = str(
        _project_dir / "experiments/017-mel-overhaul/checkpoints"
        "/2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt"
    )
    LOSSES_PATH = str(_analysis_dir / "losses_v5j.parquet")
    SAMPLE_RATE = 32000
    WINDOW_DURATION = 5.0

    mo.md(
        f"""
        ## Model Explorer — v5j (HGNetV2-B0 + LSEHead)

        **Config**: `{Path(CONFIG_PATH).name}`
        | **Checkpoint**: `{Path(CHECKPOINT_PATH).name}`
        | **Losses**: `{Path(LOSSES_PATH).name}`
        """
    )

    return CHECKPOINT_PATH, CONFIG_PATH, LOSSES_PATH, SAMPLE_RATE, WINDOW_DURATION


@app.cell
def load_data(mo, pd, Path, LOSSES_PATH):
    """Load pre-computed losses parquet."""
    _losses_path = Path(LOSSES_PATH)
    if not _losses_path.exists():
        mo.stop(
            True,
            mo.md(
                f"**Losses file not found**: `{LOSSES_PATH}`\n\n"
                "Run `precompute_losses.py` first. See README.md for instructions."
            ),
        )

    df = pd.read_parquet(LOSSES_PATH)
    all_species = sorted(df["primary_label"].unique().tolist())
    all_sources = sorted(df["source"].unique().tolist())

    mo.md(f"Loaded **{len(df)}** samples across **{len(all_species)}** species.")

    return all_sources, all_species, df


@app.cell
def controls(mo, all_species, all_sources):
    """Filter and sort controls."""
    species_dropdown = mo.ui.dropdown(
        options=["(all)"] + all_species,
        value="(all)",
        label="Species",
    )
    sort_order = mo.ui.dropdown(
        options=[
            "Loss (high → low)",
            "Loss (low → high)",
            "GT prob (low → high)",
            "GT rank (worst first)",
        ],
        value="Loss (high → low)",
        label="Sort by",
    )
    source_filter = mo.ui.dropdown(
        options=["(all)"] + all_sources,
        value="(all)",
        label="Source",
    )

    mo.hstack([species_dropdown, sort_order, source_filter], justify="start", gap=1)

    return source_filter, sort_order, species_dropdown


@app.cell
def sample_table(mo, df, species_dropdown, sort_order, source_filter):
    """Filtered and sorted sample table."""
    filtered = df.copy()

    if species_dropdown.value != "(all)":
        filtered = filtered[filtered["primary_label"] == species_dropdown.value]
    if source_filter.value != "(all)":
        filtered = filtered[filtered["source"] == source_filter.value]

    # Sort
    sort_val = sort_order.value
    if "high" in sort_val and "Loss" in sort_val:
        filtered = filtered.sort_values("loss", ascending=False)
    elif "low" in sort_val and "Loss" in sort_val:
        filtered = filtered.sort_values("loss", ascending=True)
    elif "GT prob" in sort_val:
        filtered = filtered.sort_values("gt_prob", ascending=True)
    elif "GT rank" in sort_val:
        filtered = filtered.sort_values("gt_rank", ascending=False)

    display_cols = [
        "idx",
        "primary_label",
        "source",
        "loss",
        "gt_prob",
        "gt_rank",
        "top1_species",
        "top1_prob",
    ]
    table = mo.ui.table(
        filtered[display_cols].head(200).reset_index(drop=True),
        selection="single",
        label=f"Select a sample ({len(filtered)} matches)",
    )
    _ = table  # display in Marimo

    return filtered, table


@app.cell
def load_model_cell(mo, CONFIG_PATH, CHECKPOINT_PATH):
    """Load model (runs once, cached by Marimo's reactivity)."""
    from model_loader import load_config, load_model

    _config = load_config(CONFIG_PATH)
    _lse_temp = _config.get("sed_head", {}).get("lse_temperature", 1.0)

    model, mel_transform, full_species_list = load_model(CONFIG_PATH, CHECKPOINT_PATH, device="cpu")

    mo.md(f"Model loaded: **{len(full_species_list)}** classes, LSE temp={_lse_temp}")

    return _config, _lse_temp, full_species_list, mel_transform, model


@app.cell
def attribution_controls(mo, table, full_species_list):
    """Species selector for attribution (defaults to GT species of selected sample)."""
    mo.stop(not table.value, mo.md("*Select a sample from the table above.*"))

    selected = table.value[0]
    default_species = selected["primary_label"]

    attr_species = mo.ui.dropdown(
        options=full_species_list,
        value=default_species if default_species in full_species_list else full_species_list[0],
        label="Attribution target species",
    )

    mo.hstack(
        [
            mo.md(
                f"**Selected**: idx={selected['idx']}, "
                f"species=`{selected['primary_label']}`, "
                f"source={selected['source']}, "
                f"loss={selected['loss']:.4f}, "
                f"gt_prob={selected['gt_prob']:.4f}, "
                f"gt_rank={selected['gt_rank']}"
            ),
            attr_species,
        ],
        justify="start",
        gap=1,
    )

    return attr_species, selected


@app.cell
def compute_cell(
    mo,
    selected,
    attr_species,
    model,
    mel_transform,
    full_species_list,
    _lse_temp,
    df,
    SAMPLE_RATE,
    WINDOW_DURATION,
):
    """Compute attribution for selected sample."""
    from attribution import compute_attribution, load_audio

    # Get the full filename from the original dataframe
    sample_row = df[df["idx"] == selected["idx"]].iloc[0]
    filepath = sample_row["filename"]

    waveform = load_audio(filepath, sample_rate=SAMPLE_RATE, duration=WINDOW_DURATION)
    species_idx = full_species_list.index(attr_species.value)

    result = compute_attribution(
        model=model,
        mel_transform=mel_transform,
        waveform=waveform,
        species_idx=species_idx,
        species_name=attr_species.value,
        lse_temperature=_lse_temp,
        device="cpu",
    )

    mo.md(
        f"Attribution computed for **{attr_species.value}** "
        f"(idx {species_idx}): "
        f"clipwise_prob={result.clipwise_probs[species_idx]:.4f}"
    )

    return filepath, result, waveform


@app.cell
def audio_player(mo, waveform, SAMPLE_RATE, filepath, Path):
    """Audio playback."""
    mo.md(f"**Audio**: `{Path(filepath).name}`")
    mo.audio(src=waveform.numpy(), rate=SAMPLE_RATE)
    return


@app.cell
def visualize(mo, result, full_species_list, np, plt):
    """4-panel visualization of model behavior on the selected sample."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ---- Panel 1: Mel spectrogram ----
    ax = axes[0, 0]
    ax.imshow(result.mel, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Mel Spectrogram")
    ax.set_ylabel("Mel bin")
    ax.set_xlabel("Time frame")

    # ---- Panel 2: Gradient saliency overlay ----
    ax = axes[0, 1]
    ax.imshow(result.mel, aspect="auto", origin="lower", cmap="gray", alpha=0.4)
    im = ax.imshow(
        result.gradient_saliency,
        aspect="auto",
        origin="lower",
        cmap="hot",
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"Gradient Saliency: {result.species_name}")
    ax.set_ylabel("Mel bin")
    ax.set_xlabel("Time frame")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- Panel 3: Top-10 species bar chart ----
    ax = axes[1, 0]
    top_k = 10
    top_indices = np.argsort(result.clipwise_probs)[::-1][:top_k]
    top_names = [full_species_list[i] for i in top_indices]
    top_probs = [result.clipwise_probs[i] for i in top_indices]

    colors = [
        "#d32f2f" if full_species_list[i] == result.species_name else "#1976d2" for i in top_indices
    ]
    bars = ax.barh(range(top_k), top_probs, color=colors)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("Probability")
    ax.set_title("Top-10 Predicted Species")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)

    # Add prob labels on bars
    for bar, prob in zip(bars, top_probs, strict=False):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.3f}",
            va="center",
            fontsize=7,
        )

    # ---- Panel 4: Per-frame analysis ----
    ax = axes[1, 1]
    n_frames = result.frame_probs.shape[1]
    frame_x = np.arange(n_frames)

    # Frame probs for attribution target species
    target_frame_probs = result.frame_probs[result.species_idx]
    ax.bar(
        frame_x - 0.15,
        target_frame_probs,
        width=0.3,
        color="#d32f2f",
        alpha=0.8,
        label=f"Frame prob: {result.species_name}",
    )

    # LSE effective attention
    ax.bar(
        frame_x + 0.15,
        result.effective_attention,
        width=0.3,
        color="#1976d2",
        alpha=0.8,
        label="LSE attention",
    )

    ax.set_xlabel("Frame")
    ax.set_ylabel("Value")
    ax.set_title(f"Per-Frame Analysis: {result.species_name}")
    ax.legend(fontsize=8)
    ax.set_xticks(frame_x)
    ax.set_xticklabels([f"F{i}" for i in frame_x])

    plt.tight_layout()
    _ = fig  # display in Marimo

    return


if __name__ == "__main__":
    app.run()
