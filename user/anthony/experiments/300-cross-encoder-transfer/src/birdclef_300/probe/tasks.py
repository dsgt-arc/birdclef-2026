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
)
from birdclef_300.probe.base import ProbeTask
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


class AllProbes(ProbeTask):
    """Run RidgeProbe over the roster and emit a combined comparison table."""

    encoders = luigi.Parameter(default="beats,ast,eat,ssamba_hf,perch,birdnet")
    variant = luigi.Parameter(default="frozen")
    standardize = luigi.BoolParameter(default=True)

    def _encoder_list(self) -> list[str]:
        return [e.strip() for e in str(self.encoders).split(",") if e.strip()]

    def requires(self):
        return {
            e: RidgeProbe(
                encoder=e, variant=self.variant, standardize=self.standardize,
                features_dir=self.features_dir, artifacts_dir=self.artifacts_dir,
            )
            for e in self._encoder_list()
        }

    def output(self):
        return self.artifact(f"probe_metrics_{self.variant}.json")

    def run(self):
        self.ensure_parent(self.output())
        summary = []
        for enc, target in self.requires().items():
            with open(target.output().path) as f:
                r = json.load(f)
            summary.append({
                "encoder": enc,
                "variant": r["variant"],
                "embed_dim": r["embed_dim"],
                "macro_auroc": r["macro_auroc"],
                "macro_ap": r["macro_ap"],
                "best_lambda": r["best_lambda"],
                "n_species_evaluated": r["n_species_evaluated"],
            })
        summary.sort(key=lambda x: -x["macro_auroc"])
        with open(self.output().path, "w") as f:
            json.dump({"variant": str(self.variant), "ranking": summary}, f, indent=2)
        logger.info("AllProbes[%s] ranking: %s", self.variant,
                    ", ".join(f"{s['encoder']}={s['macro_auroc']:.4f}" for s in summary))
