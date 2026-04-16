#!/bin/bash
# Submit the 3-stage exp107 cache build dependency chain:
#   bootstrap -> shard array -> finalize
set -euo pipefail

exp107_dir="$HOME/arc/birdclef-2026/user/anthony/experiments/107-hopfield-prototypes"
cd "$exp107_dir"

FEATURES_DIR=${FEATURES_DIR:-$HOME/scratch/birdclef-2026/features/exp106}
ARRAY_MAX_CONCURRENT=${ARRAY_MAX_CONCURRENT:-8}
export FEATURES_DIR

if [[ "$ARRAY_MAX_CONCURRENT" -lt 1 ]]; then
  echo "ARRAY_MAX_CONCURRENT must be >= 1" >&2
  exit 2
fi
if [[ "$ARRAY_MAX_CONCURRENT" -ge 16 ]]; then
  echo "ARRAY_MAX_CONCURRENT must stay under 16; got $ARRAY_MAX_CONCURRENT" >&2
  exit 2
fi

N_FOCAL=$(python - <<'PY'
from pathlib import Path
import os
features = Path(os.path.expanduser(os.environ["FEATURES_DIR"]))
print(len(list((features / "focal").glob("shard-*.parquet"))))
PY
)
N_UNSUP=$(python - <<'PY'
from pathlib import Path
import os
features = Path(os.path.expanduser(os.environ["FEATURES_DIR"]))
print(len(list((features / "soundscape_unlabeled").glob("shard-*.parquet"))))
PY
)
N_UNITS=$((1 + N_FOCAL + N_UNSUP))
ARRAY_SPEC="0-$((N_UNITS - 1))%${ARRAY_MAX_CONCURRENT}"

echo "Submitting exp107 cache build chain"
echo "  features_dir=$FEATURES_DIR"
echo "  work_units=$N_UNITS (labeled=1 focal=$N_FOCAL unsup=$N_UNSUP)"
echo "  array_spec=$ARRAY_SPEC"

BOOTSTRAP_JOBID=$(sbatch --parsable --export=ALL sbatch/cache_bootstrap.sbatch)
SHARDS_JOBID=$(sbatch --parsable --export=ALL --dependency=afterok:${BOOTSTRAP_JOBID} --array="$ARRAY_SPEC" sbatch/cache_shards.sbatch)
FINALIZE_JOBID=$(sbatch --parsable --export=ALL --dependency=afterok:${SHARDS_JOBID} sbatch/cache_finalize.sbatch)

echo "bootstrap: $BOOTSTRAP_JOBID"
echo "shards:    $SHARDS_JOBID"
echo "finalize:  $FINALIZE_JOBID"
