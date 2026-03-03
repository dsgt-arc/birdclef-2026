"""Phase 2 diagnostics: evaluate fused label quality.

Reports:
- Per-species LF accuracy table
- LF correlation matrix (verify independence assumption)
- Soft label distribution (should be bimodal)
- Coverage: fraction of tokens with non-abstaining LFs
- Calibration: compare soft labels vs recording-level ground truth (focal only)

CLI:
    weaksup-evaluate --split focal \\
        --fused-dir ~/scratch/birdclef-2026/outputs/exp013/fused \\
        --data-dir ~/scratch/birdclef-2026/data/birdclef-2025 \\
        --output-dir ~/scratch/birdclef-2026/outputs/exp013/eval
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from weaksup.fuse import FOCAL_LF_NAMES, SOUNDSCAPE_LF_NAMES

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

DEFAULT_FUSED_DIR = Path("~/scratch/birdclef-2026/outputs/exp013/fused").expanduser()
DEFAULT_DATA_DIR = Path("~/scratch/birdclef-2026/data/birdclef-2025").expanduser()
DEFAULT_OUTPUT_DIR = Path("~/scratch/birdclef-2026/outputs/exp013/eval").expanduser()


@app.command()
def evaluate(
    split: Annotated[
        str, typer.Option(help="Data split: focal or soundscape")
    ] = "focal",
    fused_dir: Annotated[
        Path, typer.Option(help="Dir with soft_labels.parquet and lf_accuracies.parquet")
    ] = DEFAULT_FUSED_DIR,
    data_dir: Annotated[
        Path, typer.Option(help="BirdCLEF data dir")
    ] = DEFAULT_DATA_DIR,
    output_dir: Annotated[
        Path, typer.Option(help="Output dir for reports")
    ] = DEFAULT_OUTPUT_DIR,
) -> None:
    """Evaluate fused soft labels and LF accuracy estimates."""
    if split not in ("focal", "soundscape"):
        raise typer.BadParameter(f"split must be 'focal' or 'soundscape', got {split!r}")

    is_focal = split == "focal"
    lf_names = FOCAL_LF_NAMES if is_focal else SOUNDSCAPE_LF_NAMES
    n_species_lfs = len(lf_names)

    split_fused_dir = fused_dir / split
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    console.rule(f"[bold]Loading fused outputs — {split}")

    acc_path = split_fused_dir / "lf_accuracies.parquet"
    acc_df = pl.read_parquet(acc_path)
    console.print(f"LF accuracies: {acc_df.shape}")

    soft_path = split_fused_dir / "soft_labels.parquet"
    console.print(f"Loading soft labels from {soft_path}...")
    soft_df = pl.read_parquet(soft_path)
    console.print(f"Soft labels: {soft_df.shape}")

    species_list = acc_df["species"].to_list()
    n_species = len(species_list)
    n_tokens = len(soft_df)

    # Extract soft label values (skip file, start_time columns)
    soft_labels = soft_df.select(species_list).to_numpy()

    # ── 1. LF Accuracy Summary ──
    console.rule("[bold]1. LF Accuracy Summary")

    acc_table = Table(title=f"Mean LF Accuracy Across Species ({split})")
    acc_table.add_column("LF", style="cyan")
    acc_table.add_column("Mean", justify="right")
    acc_table.add_column("Std", justify="right")
    acc_table.add_column("Min", justify="right")
    acc_table.add_column("Max", justify="right")
    acc_table.add_column(">0.6", justify="right", style="green")
    acc_table.add_column("=0.5", justify="right", style="yellow")

    for lf_name in lf_names:
        vals = acc_df[lf_name].to_numpy()
        n_good = (vals > 0.6).sum()
        n_uninf = (np.abs(vals - 0.5) < 0.01).sum()
        acc_table.add_row(
            lf_name,
            f"{vals.mean():.3f}",
            f"{vals.std():.3f}",
            f"{vals.min():.3f}",
            f"{vals.max():.3f}",
            f"{n_good}/{n_species}",
            f"{n_uninf}/{n_species}",
        )
    console.print(acc_table)

    # Save accuracy CSV
    acc_csv_path = split_output_dir / "lf_accuracies.csv"
    acc_df.write_csv(acc_csv_path)
    console.print(f"Saved: {acc_csv_path}")

    # ── 2. LF Correlation Matrix ──
    console.rule("[bold]2. LF Correlation (Independence Check)")

    acc_matrix = acc_df.select(lf_names).to_numpy()
    corr = np.corrcoef(acc_matrix.T)

    corr_table = Table(title="LF Accuracy Correlation Matrix")
    corr_table.add_column("", style="cyan")
    for name in lf_names:
        corr_table.add_column(name[:10], justify="right")

    for i, name in enumerate(lf_names):
        vals = []
        for j in range(len(lf_names)):
            c = corr[i, j]
            if i == j:
                vals.append("[dim]1.000[/dim]")
            elif abs(c) > 0.7:
                vals.append(f"[bold red]{c:.3f}[/bold red]")
            elif abs(c) > 0.4:
                vals.append(f"[yellow]{c:.3f}[/yellow]")
            else:
                vals.append(f"{c:.3f}")
        corr_table.add_row(name[:10], *vals)
    console.print(corr_table)

    corr_df = pl.DataFrame(
        {"lf": lf_names, **{lf_names[j]: corr[:, j] for j in range(len(lf_names))}}
    )
    corr_csv_path = split_output_dir / "lf_correlation.csv"
    corr_df.write_csv(corr_csv_path)
    console.print(f"Saved: {corr_csv_path}")

    # ── 3. Soft Label Distribution ──
    console.rule("[bold]3. Soft Label Distribution")

    flat = soft_labels.ravel()
    bins = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    hist, _ = np.histogram(flat, bins=bins)

    dist_table = Table(title="Soft Label Distribution (all species x all tokens)")
    dist_table.add_column("Bin", style="cyan")
    dist_table.add_column("Count", justify="right")
    dist_table.add_column("%", justify="right")

    for i in range(len(bins) - 1):
        pct = 100 * hist[i] / len(flat)
        bar = "#" * int(pct * 2)
        dist_table.add_row(
            f"[{bins[i]:.2f}, {bins[i+1]:.2f})",
            f"{hist[i]:,}",
            f"{pct:.2f}% {bar}",
        )
    console.print(dist_table)

    near_0 = (flat < 0.1).sum() / len(flat) * 100
    near_1 = (flat > 0.9).sum() / len(flat) * 100
    ambig = ((flat > 0.3) & (flat < 0.7)).sum() / len(flat) * 100
    console.print(f"Near 0 (<0.1): {near_0:.1f}%")
    console.print(f"Near 1 (>0.9): {near_1:.1f}%")
    console.print(f"Ambiguous (0.3-0.7): {ambig:.1f}%")
    bimodal = near_0 + near_1 > 80
    console.print(
        f"Bimodality: {'[green]PASS[/green]' if bimodal else '[yellow]WARN[/yellow]'} "
        f"({near_0 + near_1:.1f}% near extremes)"
    )

    # ── 4. Coverage ──
    console.rule("[bold]4. Coverage")

    n_active_species = (soft_labels > 0.1).sum(axis=1)
    console.print(f"Tokens with >=1 species > 0.1: {(n_active_species >= 1).sum()} "
                  f"({100*(n_active_species >= 1).mean():.1f}%)")
    console.print(f"Mean active species per token: {n_active_species.mean():.1f}")
    console.print(f"Max active species per token: {n_active_species.max()}")

    species_with_signal = (acc_df.select(lf_names).to_numpy().mean(axis=1) > 0.52).sum()
    console.print(f"Species with LF accuracy > 0.52: {species_with_signal}/{n_species}")

    # ── 5. Calibration vs Recording Labels (focal only) ──
    total = 0
    correct_high = 0
    if is_focal:
        console.rule("[bold]5. Calibration vs Recording Labels")

        import csv
        file_to_species: dict[str, str] = {}
        with open(data_dir / "train.csv") as f:
            for row in csv.DictReader(f):
                file_to_species[row["filename"]] = row["primary_label"]

        species_to_idx = {s: i for i, s in enumerate(species_list)}
        files = soft_df["file"].to_list()

        correct_high = 0
        correct_low = 0
        total = 0

        for i in range(0, n_tokens, 100):
            f = files[i]
            primary = file_to_species.get(f)
            if primary is None or primary not in species_to_idx:
                continue
            s_idx = species_to_idx[primary]
            soft = soft_labels[i, s_idx]
            total += 1
            if soft > 0.5:
                correct_high += 1
            else:
                correct_low += 1

        if total > 0:
            console.print(
                f"Primary species soft label > 0.5: {correct_high}/{total} "
                f"({100*correct_high/total:.1f}%)"
            )
            console.print(
                f"Primary species soft label <= 0.5: {correct_low}/{total} "
                f"({100*correct_low/total:.1f}%)"
            )

        primary_means = []
        nonprimary_means = []
        for s_idx, s_code in enumerate(species_list):
            is_primary = np.array(
                [file_to_species.get(f) == s_code for f in files], dtype=bool
            )
            if is_primary.sum() > 0:
                primary_means.append(soft_labels[is_primary, s_idx].mean())
            if (~is_primary).sum() > 0:
                nonprimary_means.append(soft_labels[~is_primary, s_idx].mean())

        if primary_means:
            console.print(
                f"Mean soft label on primary tokens: {np.mean(primary_means):.4f}"
            )
        if nonprimary_means:
            console.print(
                f"Mean soft label on non-primary tokens: {np.mean(nonprimary_means):.6f}"
            )
    else:
        console.rule("[bold]5. Calibration")
        console.print("[dim]Soundscape split: no recording-level ground truth for calibration[/dim]")

        # Basic sanity: soft labels should not be degenerate
        per_species_mean = soft_labels.mean(axis=0)
        n_degenerate_high = (per_species_mean > 0.4).sum()
        n_degenerate_low = (per_species_mean < 1e-6).sum()
        console.print(f"Species with mean soft label > 0.4: {n_degenerate_high} (should be ~0)")
        console.print(f"Species with mean soft label < 1e-6: {n_degenerate_low}")
        console.print(f"Per-species mean soft label: mean={per_species_mean.mean():.6f}, "
                      f"max={per_species_mean.max():.4f}")

    # ── 6. Save report ──
    console.rule("[bold]Report saved")
    report_path = split_output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(f"Phase 2 Evaluation Report — {split}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Tokens: {n_tokens}\n")
        f.write(f"Species: {n_species}\n")
        f.write(f"LFs: {n_species_lfs} ({', '.join(lf_names)})\n")
        f.write(f"Near 0 (<0.1): {near_0:.1f}%\n")
        f.write(f"Near 1 (>0.9): {near_1:.1f}%\n")
        f.write(f"Ambiguous (0.3-0.7): {ambig:.1f}%\n")
        f.write(f"Bimodal: {bimodal}\n")
        f.write(f"Species with signal (acc > 0.52): {species_with_signal}/{n_species}\n")
        if is_focal and total > 0:
            f.write(f"Primary species > 0.5: {100*correct_high/total:.1f}%\n")
        f.write(f"\nLF accuracies (mean across species):\n")
        for lf_name in lf_names:
            vals = acc_df[lf_name].to_numpy()
            f.write(f"  {lf_name}: {vals.mean():.3f} +/- {vals.std():.3f}\n")
    console.print(f"Report: {report_path}")


if __name__ == "__main__":
    app()
