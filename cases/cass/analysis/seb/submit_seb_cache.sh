#!/bin/bash
# submit_seb_cache.sh
#
# Submit compute_seb_cache.py to Perlmutter.
# Skips RTs whose pickle is already up-to-date (variable-set check).
#
# Usage:
#   bash submit_seb_cache.sh                       # default expt, both RTs
#   bash submit_seb_cache.sh --expt no_aerosols    # alternate expt
#   bash submit_seb_cache.sh --rt raytracer        # single RT
#   bash submit_seb_cache.sh --force               # ignore existing cache

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
LOG_DIR="$LES_ROOT/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPT="no_aerosols_zero_wind"
N_REPS=4
RT="both"
FORCE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --expt)   EXPT="$2";   shift 2 ;;
        --n-reps) N_REPS="$2"; shift 2 ;;
        --rt)     RT="$2";     shift 2 ;;
        --force)  FORCE="--force"; shift ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

tmp=$(mktemp /tmp/seb_cache_XXXXXX.sh)
cat > "$tmp" <<HEADER
#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=seb_cache_${EXPT}
#SBATCH --output=${LOG_DIR}/seb_cache_${EXPT}-%j.out
#SBATCH --error=${LOG_DIR}/seb_cache_${EXPT}-%j.err
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env

echo "=== seb_cache: ${EXPT} / ${RT} ==="
echo "Start: \$(date)"
cd ${SCRIPT_DIR}

python compute_seb_cache.py \\
    --expt   ${EXPT} \\
    --n-reps ${N_REPS} \\
    --rt     ${RT} \\
    ${FORCE}

echo "Done: \$(date)"
HEADER

jid=$(sbatch "$tmp" | awk '{print $NF}')
echo "Submitted seb_cache (${EXPT}, rt=${RT}): $jid"
echo "Logs: $LOG_DIR/seb_cache_${EXPT}-${jid}.{out,err}"
rm -f "$tmp"
