"""Luigi tasks: per-encoder ridge LOSO probe + a combined summary.

One `RidgeProbe(encoder, variant)` per encoder makes the sweep resumable — a
failed/added encoder reruns alone. `AllProbes` fans out over the roster and
writes a single comparison table (`probe_metrics.json`).
"""

from __future__ import annotations

import json
import logging

import luigi
import numpy as np

from birdclef_300.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    embeddings_to_numpy,
    extract_site,
    load_encoder_labeled,
    load_encoder_patches,
)
from birdclef_300.probe.base import ProbeTask
from birdclef_300.probe.proto_probe import proto_loso
from birdclef_300.probe.ridge_probe import ridge_loso

logger = logging.getLogger(__name__)


class RidgeProbe(ProbeTask):
    """Ridge LOSO probe for one (encoder, variant) against the labeled windows."""

    encoder = luigi.Parameter()
    variant = luigi.Parameter(default="frozen")
    standardize = luigi.BoolParameter(default=True)

    def output(self):
        return self.artifact(f"probe_{self.encoder}_{self.variant}.json")

    def run(self):
        self.ensure_parent(self.output())
        df = load_encoder_labeled(self.features_dir, self.encoder, self.variant)
        species_list = build_species_list(df)
        X = embeddings_to_numpy(df)
        Y = build_binary_label_matrix(df, species_list)
        sites = np.array([extract_site(f) for f in df["filename"].to_list()])

        logger.info("RidgeProbe[%s/%s]: %d windows, %d species, dim=%d",
                    self.encoder, self.variant, len(df), len(species_list), X.shape[1])
        result = ridge_loso(X, Y, sites, species_list=species_list, standardize=self.standardize)
        result["encoder"] = self.encoder
        result["variant"] = self.variant
        logger.info("RidgeProbe[%s/%s]: macro AUROC=%.4f (λ=%.3g)",
                    self.encoder, self.variant, result["macro_auroc"], result["best_lambda"])

        with open(self.output().path, "w") as f:
            json.dump(result, f, indent=2)


class ProtoProbe(ProbeTask):
    """Prototypical LOSO probe for one (encoder, variant) — MAE-appropriate readout.

    Loads the pooled cache (Strategy A) and, if a `patches.parquet` exists for this
    (encoder, variant), the per-patch cache (Strategy B, frame-level max).
    """

    encoder = luigi.Parameter()
    variant = luigi.Parameter(default="frozen")
    standardize = luigi.BoolParameter(default=True)  # accepted for parity; proto uses L2-cosine

    def output(self):
        return self.artifact(f"proto_{self.encoder}_{self.variant}.json")

    def run(self):
        self.ensure_parent(self.output())
        df = load_encoder_labeled(self.features_dir, self.encoder, self.variant)
        species_list = build_species_list(df)
        X = embeddings_to_numpy(df)
        Y = build_binary_label_matrix(df, species_list)
        sites = np.array([extract_site(f) for f in df["filename"].to_list()])
        patches = load_encoder_patches(self.features_dir, self.encoder, self.variant)

        logger.info("ProtoProbe[%s/%s]: %d windows, %d species, dim=%d, patches=%s",
                    self.encoder, self.variant, len(df), len(species_list), X.shape[1],
                    "yes" if patches is not None else "no")
        result = proto_loso(X, Y, sites, patches=patches, species_list=species_list)
        result["encoder"] = self.encoder
        result["variant"] = self.variant
        logger.info("ProtoProbe[%s/%s]: clip AUROC=%.4f%s", self.encoder, self.variant,
                    result["macro_auroc"],
                    f", frame-max AUROC={result['macro_auroc_patch']:.4f}"
                    if "macro_auroc_patch" in result else "")
        with open(self.output().path, "w") as f:
            json.dump(result, f, indent=2)


class AllProbes(ProbeTask):
    """Run the chosen readout over the roster and emit a combined comparison table.

    `readout` selects ridge (linear, default) or proto (prototypical). For proto the
    table surfaces both clip-cosine and frame-max (if patches were cached).
    """

    encoders = luigi.Parameter(default="beats,ast,eat,ssamba_hf,perch,birdnet")
    variant = luigi.Parameter(default="frozen")
    readout = luigi.Parameter(default="ridge")
    standardize = luigi.BoolParameter(default=True)

    def _encoder_list(self) -> list[str]:
        return [e.strip() for e in str(self.encoders).split(",") if e.strip()]

    def _probe(self, e: str):
        cls = ProtoProbe if str(self.readout) == "proto" else RidgeProbe
        return cls(encoder=e, variant=self.variant, standardize=self.standardize,
                   features_dir=self.features_dir, artifacts_dir=self.artifacts_dir)

    def requires(self):
        return {e: self._probe(e) for e in self._encoder_list()}

    def output(self):
        # Keep the historical name for ridge (exp 300 frozen); suffix other readouts.
        name = (f"probe_metrics_{self.variant}.json" if str(self.readout) == "ridge"
                else f"probe_metrics_{self.variant}_{self.readout}.json")
        return self.artifact(name)

    def run(self):
        self.ensure_parent(self.output())
        summary = []
        for enc, target in self.requires().items():
            with open(target.output().path) as f:
                r = json.load(f)
            row = {
                "encoder": enc,
                "variant": r["variant"],
                "embed_dim": r["embed_dim"],
                "macro_auroc": r["macro_auroc"],          # pooled-OOF, LB-aligned
                "macro_ap": r["macro_ap"],
                "n_species_evaluated": r["n_species_evaluated"],
            }
            if str(self.readout) == "ridge":
                row["fold_avg_auroc"] = r.get("fold_avg_auroc")
                row["fold_avg_ap"] = r.get("fold_avg_ap")
                row["best_lambda"] = r["best_lambda"]
            else:  # proto
                row["macro_auroc_patch"] = r.get("macro_auroc_patch")
                row["macro_ap_patch"] = r.get("macro_ap_patch")
            summary.append(row)
        # Rank/report by the most informative available metric (frame-max for proto if present).
        def primary(s):
            return s.get("macro_auroc_patch") or s["macro_auroc"]
        summary.sort(key=lambda s: -primary(s))
        with open(self.output().path, "w") as f:
            json.dump({"variant": str(self.variant), "readout": str(self.readout),
                       "ranking": summary}, f, indent=2)
        logger.info("AllProbes[%s/%s] ranking: %s", self.variant, self.readout,
                    ", ".join(f"{s['encoder']}={primary(s):.4f}" for s in summary))
