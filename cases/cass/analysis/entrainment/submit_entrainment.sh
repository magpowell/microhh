#!/bin/bash
# submit_entrainment.sh
#
# Per-rep lateral entrainment diagnostic (Gentine et al. 2016) for one CASS
# experiment, 2stream + raytracer, on a single CPU node.
#
# Each (rt, rep) pair runs as a background process; up to MAX_PAR at once.
# Reps whose entrainment.nc already exists are skipped (safe to re-run).
#
# Usage:
#   bash submit_entrainment.sh --expt no_aerosols_zero_wind
#   bash submit_entrainment.sh --expt no_aerosols_zero_wind --dry-run
#   bash submit_entrainment.sh --expt no_aerosols_zero_wind --force

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
MAX_PAR=4    # 4 reps × ~5 GB each = ~20 GB — well within 512 GB node

while [[ $# -gt 0 ]]; do
    case "$1" in
        --expt)    EXPT="$2";  shift 2 ;;
        --lst-lo)  LST_LO="$2"; shift 2 ;;
        --lst-hi)  LST_HI="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --force)   FORCE="--force"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXPT" ]]; then
    echo "Usage: bash submit_entrainment.sh --expt <experiment_name> [--dry-run] [--force]"
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
        qt="$EXPT_ROOT/$rt/rep_$rep/qt.nc"
        ql="$EXPT_ROOT/$rt/rep_$rep/ql.nc"
        w="$EXPT_ROOT/$rt/rep_$rep/w.nc"
        if [[ -f "$thl" && -f "$qt" && -f "$ql" && -f "$w" ]]; then
            PAIRS+=("$rt $rep")
        else
            echo "  SKIP $rt rep_$rep (missing 3D dump files)"
        fi
    done
done

echo "Experiment : $EXPT"
echo "LST window : [$LST_LO, $LST_HI]"
echo "Reps found : ${#PAIRS[@]}"
for pair in "${PAIRS[@]}"; do
    rt="${pair%% *}"
    rep="${pair##* }"
    out="$LES_ROOT/analysis/entrainment/$EXPT/$rt/rep_$rep/entrainment.nc"
    status="compute"
    [[ -f "$out" && -z "$FORCE" ]] && status="skip (exists)"
    printf "  %-12s rep_%s  →  %s\n" "$rt" "$rep" "$status"
done

if $DRY_RUN; then
    echo ""
    echo "[dry-run] Would submit 1 job for ${#PAIRS[@]} (rt, rep) pairs."
    exit 0
fi

_pairs_lines=""
for pair in "${PAIRS[@]}"; do
    _pairs_lines+="    '${pair}'"$'\n'
done

tmp=$(mktemp /tmp/entrainment_XXXXXX.sh)
cat > "$tmp" <<HEADER
#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=00:29:00
#SBATCH --job-name=entr_${EXPT}
#SBATCH --output=${LOG_DIR}/entr_${EXPT}-%j.out
#SBATCH --error=${LOG_DIR}/entr_${EXPT}-%j.err
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
SCRIPT="${SCRIPT_DIR}/compute_entrainment.py"

declare -a PAIRS=(
${_pairs_lines})

HEADER

cat >> "$tmp" <<'LOGIC'
echo "=== entrainment  expt=${EXPT}  Start: $(date) ==="
echo "Total (rt, rep) pairs: ${#PAIRS[@]}"

PIDS=()
FAIL=0

throttle() {
    while [[ $(jobs -rp | wc -l) -ge $MAX_PAR ]]; do sleep 2; done
}

for pair in "${PAIRS[@]}"; do
    rt="${pair%% *}"
    rep="${pair##* }"
    log="${SCRATCH}/CASS_LES/logs/entrainment_${EXPT}_${rt}_rep${rep}.log"
    echo "START: $rt  rep_$rep"
    throttle
    (python "$SCRIPT" --expt "$EXPT" --rt "$rt" --rep "${rep#0}" \
        --lst-lo "$LST_LO" --lst-hi "$LST_HI" $FORCE \
        > "$log" 2>&1 \
        && echo "DONE:  $rt  rep_$rep" \
        || echo "FAIL:  $rt  rep_$rep  (see $log)") &
    PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} background job(s)..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done

echo "=== Done: $(date) ==="
[[ $FAIL -eq 0 ]] || { echo "One or more reps failed — check per-rep logs in $SCRATCH/CASS_LES/logs/"; exit 1; }
LOGIC

jid=$(sbatch "$tmp" | awk '{print $NF}')
rm -f "$tmp"
echo ""
echo "Submitted: $jid   log: $LOG_DIR/entr_${EXPT}-${jid}.out"
