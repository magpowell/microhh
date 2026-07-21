#!/bin/bash
# submit_couvreux_columns.sh
# Submit compute_couvreux_columns.py to Perlmutter.
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
LOG_DIR="$LES_ROOT/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$LOG_DIR"

tmp=$(mktemp /tmp/couvreux_cols_XXXXXX.sh)
cat > "$tmp" <<HEADER
#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=couvreux_cols
#SBATCH --output=${LOG_DIR}/couvreux_cols-%j.out
#SBATCH --error=${LOG_DIR}/couvreux_cols-%j.err
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env

echo "=== couvreux_cols ==="
echo "Start: \$(date)"
cd ${SCRIPT_DIR}

python -u compute_couvreux_columns.py

echo "Done: \$(date)"
HEADER

jid=$(sbatch "$tmp" | awk '{print $NF}')
echo "Submitted couvreux_cols: $jid"
echo "Logs: $LOG_DIR/couvreux_cols-${jid}.{out,err}"
rm -f "$tmp"
