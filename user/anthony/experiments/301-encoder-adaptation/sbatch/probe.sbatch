#!/bin/bash
#SBATCH --job-name=301-probe
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 60 -q inferno
#SBATCH --output=logs/probe-%j.out

# LOSO probe over the ADAPTED variants (delegates to the exp 300 probe).
# Compare its macro AUROC against the frozen baseline (exp 300 probe_metrics_frozen.json).
#
# READOUT selects the readout (default ridge):
#   ridge — linear LOSO probe (the exp 300 yardstick)
#   proto — prototypical LOSO (MAE-appropriate; clip-cosine + frame-max if patches cached)
#
# Usage:
#   VARIANT=dapt-native-ep1 ENCODERS=eat,ssamba_hf                sbatch sbatch/probe.sbatch
#   VARIANT=dapt-native-ep1 ENCODERS=eat,ssamba_hf READOUT=proto  sbatch sbatch/probe.sbatch
#   VARIANT=frozen          ENCODERS=eat,ssamba_hf READOUT=proto  sbatch sbatch/probe.sbatch

set -euo pipefail
mkdir -p logs

source sbatch/_setup-venv.sh

EXP300_CONFIG=${EXP300_CONFIG:-../300-cross-encoder-transfer/configs/exp300.yaml}

# AllProbes' combined target (probe_metrics_<variant>[_<readout>].json) is keyed by
# variant ONLY, not encoder. When two encoders share a variant string (e.g. the ssamba
# and eat 10s ladders both use dapt-native-10s-ep<N>), the first run's combined file makes
# Luigi skip the second encoder. FORCE=1 deletes the combined target so the per-encoder
# probes (probe_<enc>_*, proto_<enc>_* — kept independently) actually run for this encoder.
if [ "${FORCE:-0}" != "0" ]; then
    R="${READOUT:-ridge}"
    if [ "$R" = "ridge" ]; then rm -f "artifacts/probe_metrics_${VARIANT}.json"
    else rm -f "artifacts/probe_metrics_${VARIANT}_${R}.json"; fi
fi

birdclef-300 probe \
    --config "$EXP300_CONFIG" \
    --encoders "${ENCODERS:?set ENCODERS=beats,eat,ssamba_hf}" \
    --variant "${VARIANT:?set VARIANT=lora-r8|dapt-native-ep1}" \
    --readout "${READOUT:-ridge}" \
    --artifacts-dir artifacts \
    --workers "${WORKERS:-4}"
