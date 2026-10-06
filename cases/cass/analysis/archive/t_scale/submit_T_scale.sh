#!/bin/bash
# submit_T_scale.sh
#
# Compute the sliding-window integral time scale T_τ of LWP for all reps
# of one experiment (2stream + raytracer) on a single CPU node.
#
# Each (rt, rep) pair runs as a background process; up to MAX_PAR at once.
# Reps whose T_scale.nc already exists are skipped (safe to re-run).
#
# Usage:
#   bash submit_T_scale.sh --expt no_aerosols_zero_wind
#   bash submit_T_scale.sh --expt base
#   bash submit_T_scale.sh --expt no_aerosols_zero_wind --dry-run
#   bash submit_T_scale.sh --expt no_aerosols_zero_wind --force    # overwrite cache

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$LES_ROOT/logs"

# ── Defaults ──────────────────────────────────────────────────────────────────
EXPT=""
DRY_RUN=false
FORCE=""
MAX_PAR=8    # 8 reps × ~2 GB each = ~16 GB — well within 512 GB node

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --expt)    EXPT="$2";  shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --force)   FORCE="--force"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXPT" ]]; then
    echo "Usage: bash submit_T_scale.sh --expt <experiment_name> [--dry-run] [--force]"
    exit 1
fi

mkdir -p "$LOG_DIR"

# ── Build (rt, rep) list for this experiment ──────────────────────────────────
RT_LIST=(2stream raytracer)
REP_LIST=(01 02 03 04)

if [[ "$EXPT" == "base" ]]; then
    EXPT_ROOT="$LES_ROOT/base"
else
    EXPT_ROOT="$LES_ROOT/experiments/$EXPT"
fi

PAIRS=()   # "rt rep" strings for reps that have qlqi_path.xy.nc
for rt in "${RT_LIST[@]}"; do
    for rep in "${REP_LIST[@]}"; do
        nc="$EXPT_ROOT/$rt/rep_$rep/qlqi_path.xy.nc"
        if [[ -f "$nc" ]]; then
            PAIRS+=("$rt $rep")
        fi
    done
done

echo "Experiment : $EXPT"
echo "Reps found : ${#PAIRS[@]}"
for pair in "${PAIRS[@]}"; do
    rt="${pair%% *}"
    rep="${pair##* }"
    out="$LES_ROOT/analysis/timescale/$EXPT/$rt/rep_$rep/T_scale.nc"
    status="compute"
    [[ -f "$out" ]] && status="skip (exists)"
    printf "  %-12s rep_%s  →  %s\n" "$rt" "$rep" "$status"
done

if $DRY_RUN; then
    echo ""
    echo "[dry-run] Would submit 1 job for ${#PAIRS[@]} (rt, rep) pairs."
    exit 0
fi

# ── Generate pairs string for embedding in the batch script ──────────────────
_pairs_lines=""
for pair in "${PAIRS[@]}"; do
    _pairs_lines+="    '${pair}'"$'\n'
done

# ── Generate and submit the batch script ─────────────────────────────────────
tmp=$(mktemp /tmp/T_scale_XXXXXX.sh)
cat > "$tmp" <<HEADER
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --job-name=T_scale_${EXPT}
#SBATCH --output=${LOG_DIR}/T_scale_${EXPT}-%j.out
#SBATCH --error=${LOG_DIR}/T_scale_${EXPT}-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env

MAX_PAR=${MAX_PAR}
EXPT="${EXPT}"
FORCE="${FORCE}"
SCRIPT="${SCRIPT_DIR}/compute_T_scale.py"

declare -a PAIRS=(
${_pairs_lines})

HEADER

cat >> "$tmp" <<'LOGIC'
echo "=== T_scale  expt=${EXPT}  Start: $(date) ==="
echo "Total (rt, rep) pairs: ${#PAIRS[@]}"

PIDS=()
FAIL=0

throttle() {
    while [[ $(jobs -rp | wc -l) -ge $MAX_PAR ]]; do sleep 2; done
}

for pair in "${PAIRS[@]}"; do
    rt="${pair%% *}"
    rep="${pair##* }"
    log_dir="$(dirname "$SCRIPT")/../../.."   # just use LOG_DIR from env
    log="${SCRATCH}/CASS_LES/logs/T_scale_${EXPT}_${rt}_rep${rep}.log"
    echo "START: $rt  rep_$rep"
    throttle
    (python "$SCRIPT" --expt "$EXPT" --rt "$rt" --rep "${rep#0}" $FORCE \
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
echo "Submitted: $jid   log: $LOG_DIR/T_scale_${EXPT}-${jid}.out"
