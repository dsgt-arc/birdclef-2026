#!/bin/bash
# Phase 4.2 — D.5 end-to-end validation launcher for all 10 exp 033 ablations.
#
# What it does:
#   1) Iterates over the 10 ablations: a{1..5}_{v5j,kd}.
#   2) Picks the correct final checkpoint per ablation:
#        - T2 (`*-kd`)  → last.ckpt        (single clean launch)
#        - T1 (`*-v5j`) → last-v1.ckpt     (relaunched after num_workers=0→8 fix;
#                                           the bare `last.ckpt` in those dirs is
#                                           a stale ~epoch-5 snapshot from the
#                                           cancelled job — do NOT use it.)
#   3) Reads `train_auroc` from artifacts/training/<slug>/fold0/metrics.json.
#   4) sbatch's exp 032's `validate_end_to_end.sh` per ablation, passing
#      CKPT / CONFIG / TRAIN_AUROC / FOLD env vars.
#
# Reuses (does NOT duplicate) exp 032's validate_end_to_end.sh + .py. The .py
# was patched 2026-05-27 to deep-merge a sibling `_common.yaml` (no-op for
# exp 032 which has none), letting exp 033's diff-only configs load.
#
# Usage:
#     cd user/murilo/experiments/033-own-teacher-pl
#     bash sbatch/launch_d5_validation.sh
#
# Output per job: logs/slurm_e2e_val_<jobid>.log (lands in exp 032's logs dir
# because validate_end_to_end.sh cd's into its own EXPERIMENT_DIR — see notes
# at end of this script).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXP033_DIR="${REPO_ROOT}/user/murilo/experiments/033-own-teacher-pl"
EXP032_DIR="${REPO_ROOT}/user/murilo/experiments/032-tucker-distilled-sed"
CKPT_BASE="/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/033-own-teacher-pl/checkpoints"

if [ ! -d "${EXP033_DIR}/configs" ]; then
    echo "ERROR: configs dir not found at ${EXP033_DIR}/configs" >&2
    exit 1
fi
if [ ! -f "${EXP032_DIR}/sbatch/validate_end_to_end.sh" ]; then
    echo "ERROR: validate_end_to_end.sh not found at ${EXP032_DIR}/sbatch/" >&2
    exit 1
fi

cd "${EXP032_DIR}"

FOLD=0
SUBMITTED=()
SKIPPED=()

for AB in a1 a2 a3 a4 a5; do
    for T in v5j kd; do
        SLUG="2026-exp033-${AB}-${T}"
        CONFIG="${EXP033_DIR}/configs/${AB}_${T}.yaml"
        METRICS_JSON="${EXP033_DIR}/artifacts/training/${AB}_${T}/fold${FOLD}/metrics.json"

        # T1 (v5j) uses last-v1.ckpt; T2 (kd) uses last.ckpt.
        if [ "${T}" = "v5j" ]; then
            CKPT="${CKPT_BASE}/${SLUG}/fold${FOLD}/last-v1.ckpt"
        else
            CKPT="${CKPT_BASE}/${SLUG}/fold${FOLD}/last.ckpt"
        fi

        # Sanity checks before sbatch.
        if [ ! -f "${CKPT}" ]; then
            echo "[SKIP] ${SLUG} — ckpt missing: ${CKPT}"
            SKIPPED+=("${SLUG}: ckpt missing")
            continue
        fi
        if [ ! -f "${CONFIG}" ]; then
            echo "[SKIP] ${SLUG} — config missing: ${CONFIG}"
            SKIPPED+=("${SLUG}: config missing")
            continue
        fi
        if [ ! -f "${METRICS_JSON}" ]; then
            echo "[SKIP] ${SLUG} — metrics.json missing: ${METRICS_JSON}"
            SKIPPED+=("${SLUG}: metrics.json missing")
            continue
        fi

        TRAIN_AUROC=$(jq -r '.auroc' "${METRICS_JSON}")
        if [ "${TRAIN_AUROC}" = "null" ] || [ -z "${TRAIN_AUROC}" ]; then
            echo "[SKIP] ${SLUG} — could not read .auroc from ${METRICS_JSON}"
            SKIPPED+=("${SLUG}: auroc null")
            continue
        fi

        echo ""
        echo "==> Submitting ${SLUG}"
        echo "    config:       ${CONFIG}"
        echo "    ckpt:         ${CKPT}"
        echo "    train_auroc:  ${TRAIN_AUROC}"
        echo "    fold:         ${FOLD}"

        JOB_OUT=$(
            CKPT="${CKPT}" \
            CONFIG="${CONFIG}" \
            TRAIN_AUROC="${TRAIN_AUROC}" \
            FOLD="${FOLD}" \
            sbatch --parsable \
                --job-name="exp033-e2e-${AB}-${T}" \
                "${EXP032_DIR}/sbatch/validate_end_to_end.sh"
        )
        JOB_ID="${JOB_OUT}"
        echo "    job_id:       ${JOB_ID}"
        SUBMITTED+=("${SLUG} job=${JOB_ID} ckpt=$(basename "${CKPT}") auroc=${TRAIN_AUROC}")
    done
done

echo ""
echo "=============================================="
echo "Summary"
echo "=============================================="
echo "Submitted (${#SUBMITTED[@]}):"
for s in "${SUBMITTED[@]}"; do echo "  ${s}"; done
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "Skipped (${#SKIPPED[@]}):"
    for s in "${SKIPPED[@]}"; do echo "  ${s}"; done
fi
echo ""
echo "Watch:    squeue -u \$USER"
echo "Logs in:  ${EXP032_DIR}/logs/slurm_e2e_val_<jobid>.log"
echo ""
echo "After all 10 jobs complete, parse PASS/FAIL + e2e val/auroc from logs."

# Note on log location:
# validate_end_to_end.sh sets `cd \"$EXPERIMENT_DIR\"` early (line 86 of that
# file). EXPERIMENT_DIR resolves to exp 032 because the .sh lives there. Logs
# land in 032's logs/. That's fine for D.5 — the .tsv it writes to outputs/
# also lives under 032. If we want exp 033-local logs in a follow-up, we'd
# copy validate_end_to_end.sh + .py into exp 033 with its EXPERIMENT_DIR
# pointing at 033. Not worth the duplication for one pre-flight run.
