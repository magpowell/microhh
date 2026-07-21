#!/bin/bash
# submit_shell.sh — Heus & Jonker 2008 region diagnostics for one CASS expt.

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$LES_ROOT/logs"

EXPT=""
DRY_RUN=false
FORCE=""
LST_LO="11.0"
LST_HI="17.0"
QOS="regular"
TIME="02:00:00"
MAX_PAR=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --expt)    EXPT="$2";  shift 2 ;;
        --lst-lo)  LST_LO="$2"; shift 2 ;;
        --lst-hi)  LST_HI="$2"; shift 2 ;;
        --debug)   QOS="debug"; TIME="00:29:00"; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --force)   FORCE="--force"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXPT" ]]; then
    echo "Usage: bash submit_shell.sh --expt <experiment> [--debug] [--dry-run] [--force]"
    exit 1
fi

mkdir -p "$LOG_DIR"

RT_LIST=(2stream raytracer)
REP_LIST=(01 02 03 04)
if [[ "$EXPT" == "base" ]]; then
    EXPT_ROOT="$LES_ROOT/base"
else
    EXPT_ROOT="$LES_ROOT/experiments/$EXPT"
fi

PAIRS=()
for rt in "${RT_LIST[@]}"; do
    for rep in "${REP_LIST[@]}"; do
        thl="$EXPT_ROOT/$rt/rep_$rep/thl.nc"
        if [[ -f "$thl" ]]; then
            PAIRS+=("$rt $rep")
        fi
    done
done

echo "Experiment : $EXPT"
echo "QOS        : $QOS  (time $TIME)"
echo "LST window : [$LST_LO, $LST_HI]"
echo "Pairs      : ${#PAIRS[@]}"

if $DRY_RUN; then
    echo "[dry-run] Would submit 1 job for ${#PAIRS[@]} pairs."
    exit 0
fi

_pairs_lines=""
for pair in "${PAIRS[@]}"; do
    _pairs_lines+="    '${pair}'"$'\n'
done

tmp=$(mktemp /tmp/shell_XXXXXX.sh)
cat > "$tmp" <<HEADER
#!/bin/bash
#SBATCH --qos=${QOS}
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=${TIME}
#SBATCH --job-name=shell_${EXPT}
#SBATCH --output=${LOG_DIR}/shell_${EXPT}-%j.out
#SBATCH --error=${LOG_DIR}/shell_${EXPT}-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env

MAX_PAR=${MAX_PAR}
EXPT="${EXPT}"
FORCE="${FORCE}"
LST_LO="${LST_LO}"
LST_HI="${LST_HI}"
SCRIPT="${SCRIPT_DIR}/compute_shell.py"

declare -a PAIRS=(
${_pairs_lines})

HEADER

cat >> "$tmp" <<'LOGIC'
echo "=== shell  expt=${EXPT}  Start: $(date) ==="
PIDS=()
FAIL=0

throttle() {
    while [[ $(jobs -rp | wc -l) -ge $MAX_PAR ]]; do sleep 2; done
}

for pair in "${PAIRS[@]}"; do
    rt="${pair%% *}"
    rep="${pair##* }"
    log="${SCRATCH}/CASS_LES/logs/shell_${EXPT}_${rt}_rep${rep}.log"
    echo "START: $rt rep_$rep"
    throttle
    (python "$SCRIPT" --expt "$EXPT" --rt "$rt" --rep "${rep#0}" \
        --lst-lo "$LST_LO" --lst-hi "$LST_HI" $FORCE \
        > "$log" 2>&1 \
        && echo "DONE: $rt rep_$rep" \
        || echo "FAIL: $rt rep_$rep (see $log)") &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
echo "=== Done: $(date) ==="
[[ $FAIL -eq 0 ]] || exit 1
LOGIC

jid=$(sbatch "$tmp" | awk '{print $NF}')
rm -f "$tmp"
echo ""
echo "Submitted: $jid   log: $LOG_DIR/shell_${EXPT}-${jid}.out"
