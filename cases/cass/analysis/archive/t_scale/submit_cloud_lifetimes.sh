#!/bin/bash
# submit_cloud_lifetimes.sh
#
# Run tobac-based Lagrangian cloud-lifetime tracking for all reps of one
# experiment (2stream + raytracer) on a single CPU node, in parallel.
#
# Each (rt, rep) pair runs as a background process; up to MAX_PAR at once.
# Reps with existing output are skipped (safe to re-run; pass --force to redo).
#
# Usage:
#   bash submit_cloud_lifetimes.sh --expt wind_sun
#   bash submit_cloud_lifetimes.sh --expt no_aerosols_zero_wind --threshold 5
#   bash submit_cloud_lifetimes.sh --expt base --dry-run
#   bash submit_cloud_lifetimes.sh --expt no_aerosols_zero_wind --force

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$LES_ROOT/logs"

# ── Defaults ──────────────────────────────────────────────────────────────────
EXPT=""
DRY_RUN=false
FORCE=""
THRESHOLD="1.0"
MIN_AREA="2500"
SHADOW=""
MAX_PAR=8

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --expt)      EXPT="$2";        shift 2 ;;
        --threshold) THRESHOLD="$2";   shift 2 ;;
        --min-area)  MIN_AREA="$2";    shift 2 ;;
        --shadow-check) SHADOW="--shadow-check"; shift ;;
        --dry-run)   DRY_RUN=true;     shift ;;
        --force)     FORCE="--force";  shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXPT" ]]; then
    echo "Usage: bash submit_cloud_lifetimes.sh --expt <name> [--threshold G/M2] [--min-area M2] [--dry-run] [--force]"
    exit 1
fi

# Sanitised tag for SLURM job-name / log filenames (EXPT may contain '/',
# e.g. wind_geo/u_5p0, which is valid as a path but not as a filename).
EXPT_TAG="${EXPT//\//_}"

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
        nc="$EXPT_ROOT/$rt/rep_$rep/qlqi_path.xy.nc"
        if [[ -f "$nc" ]]; then
            PAIRS+=("$rt $rep")
        fi
    done
done

echo "Experiment : $EXPT"
echo "Threshold  : $THRESHOLD g/m²"
echo "Min area   : $MIN_AREA m²"
echo "Reps found : ${#PAIRS[@]}"
for pair in "${PAIRS[@]}"; do
    rt="${pair%% *}"
    rep="${pair##* }"
    out="$LES_ROOT/analysis/lifetime/$EXPT/$rt/rep_$rep/cloud_tracks.nc"
    out2="$LES_ROOT/analysis/lifetime/$EXPT/$rt/rep_$rep/cloud_lifetime_ts.nc"
    status="compute"
    if [[ -f "$out" && -f "$out2" && -z "$FORCE" ]]; then status="skip (exists)"; fi
    printf "  %-10s rep_%s  →  %s\n" "$rt" "$rep" "$status"
done

if $DRY_RUN; then
    echo
    echo "[dry-run] Would submit 1 job for ${#PAIRS[@]} (rt, rep) pairs."
    exit 0
fi

_pairs_lines=""
for pair in "${PAIRS[@]}"; do
    _pairs_lines+="    '${pair}'"$'\n'
done

tmp=$(mktemp /tmp/cloud_lifetimes_XXXXXX.sh)
cat > "$tmp" <<HEADER
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --job-name=cloud_life_${EXPT_TAG}
#SBATCH --output=${LOG_DIR}/cloud_life_${EXPT_TAG}-%j.out
#SBATCH --error=${LOG_DIR}/cloud_life_${EXPT_TAG}-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env

MAX_PAR=${MAX_PAR}
EXPT="${EXPT}"
EXPT_TAG="${EXPT_TAG}"
FORCE="${FORCE}"
THRESHOLD="${THRESHOLD}"
MIN_AREA="${MIN_AREA}"
SCRIPT="${SCRIPT_DIR}/compute_cloud_lifetimes.py"

declare -a PAIRS=(
${_pairs_lines})

HEADER

cat >> "$tmp" <<'LOGIC'
echo "=== cloud_lifetimes  expt=${EXPT}  Start: $(date) ==="
echo "Total (rt, rep) pairs: ${#PAIRS[@]}"

PIDS=()
FAIL=0

throttle() {
    while [[ $(jobs -rp | wc -l) -ge $MAX_PAR ]]; do sleep 2; done
}

for pair in "${PAIRS[@]}"; do
    rt="${pair%% *}"
    rep="${pair##* }"
    log="${SCRATCH}/CASS_LES/logs/cloud_life_${EXPT_TAG}_${rt}_rep${rep}.log"
    echo "START: $rt  rep_$rep"
    throttle
    (python "$SCRIPT" --expt "$EXPT" --rt "$rt" --rep "${rep#0}" \
            --threshold "$THRESHOLD" --min-area "$MIN_AREA" $FORCE \
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
echo
echo "Submitted: $jid   log: $LOG_DIR/cloud_life_${EXPT_TAG}-${jid}.out"
