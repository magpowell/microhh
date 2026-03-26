#!/bin/bash
# submit_build_raw_flux_cache.sh
#
# Computes raw_flux_profile_cache.nc for all sweep experiment reps.
# Already-cached reps are skipped (safe to rerun).
#
# Usage:
#   bash submit_build_raw_flux_cache.sh [--dry-run]

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LOG_DIR="$SCRATCH/CASS_LES/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

mkdir -p "$LOG_DIR"

if $DRY_RUN; then
    python3 "$SCRIPT_DIR/build_raw_flux_cache.py" --dry-run
    exit 0
fi

tmp=$(mktemp /tmp/raw_flux_cache_XXXXXX.sh)
cat > "$tmp" <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=460G
#SBATCH --time=06:00:00
#SBATCH --job-name=raw_flux_cache
#SBATCH --output=${LOG_DIR}/raw_flux_cache-%j.out
#SBATCH --error=${LOG_DIR}/raw_flux_cache-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env

echo "=== build_raw_flux_cache  Start: \$(date) ==="
python "${SCRIPT_DIR}/build_raw_flux_cache.py" --n-par 8
echo "=== Done: \$(date) ==="
EOF

jid=$(sbatch "$tmp" | awk '{print $NF}')
rm -f "$tmp"
echo "Submitted: $jid   log: $LOG_DIR/raw_flux_cache-${jid}.out"
