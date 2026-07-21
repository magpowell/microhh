#!/bin/bash
# submit_autocorr.sh
# Submit compute_autocorr.py to Perlmutter.
set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
LOG_DIR="$LES_ROOT/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$LOG_DIR"

tmp=$(mktemp /tmp/autocorr_XXXXXX.sh)
cat > "$tmp" <<HEADER
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=autocorr
#SBATCH --output=${LOG_DIR}/autocorr-%j.out
#SBATCH --error=${LOG_DIR}/autocorr-%j.err
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env

echo "=== autocorr ==="
echo "Start: \$(date)"
cd ${SCRIPT_DIR}

python -u compute_autocorr.py

echo "Done: \$(date)"
HEADER

jid=$(sbatch "$tmp" | awk '{print $NF}')
echo "Submitted autocorr: $jid"
echo "Logs: $LOG_DIR/autocorr-${jid}.{out,err}"
rm -f "$tmp"
