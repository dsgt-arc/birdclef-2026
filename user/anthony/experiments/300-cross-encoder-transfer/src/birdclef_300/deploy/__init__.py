"""Deploy: turn a frozen-encoder ridge probe into a Kaggle-submittable head.

Exp 300 ranks encoders on a LOSO yardstick but never fits a final model. This
package adds the two missing pieces for a competition baseline:

    fit_head.py   — fit one ridge head on ALL training embeddings (focal +
                    soundscape), over the full 234-species roster, and export a
                    portable `head.npz` (W + standardizer + species list).
    species_map.py — the 234-species roster + the Perch-only native-logit
                     fallback maps for species the head cannot cover.

The head is backbone-agnostic (`embeddings -> standardize -> W -> clip`); only the
fallback is Perch-specific. See `kaggle_kernel/submission.py` for inference.
"""
