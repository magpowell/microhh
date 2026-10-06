#!/bin/bash
# Submit one sbatch job that builds the H1/H2 mechanism cache for all
# (rt, rep) tuples of an experiment in parallel on a single CPU node.
#
# Usage:
#   bash submit_h1h2.sh                                     # default expt
#   bash submit_h1h2.sh --expt no_aerosols_zero_wind_v2 --force
#   bash submit_h1h2.sh --rts "2stream" --reps "rep_01 rep_02"
#   bash submit_h1h2.sh --dry-run

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$LES_ROOT/logs"

EXPT="no_aerosols_zero_wind_v2"
RTS="2stream raytracer"
REPS="rep_01 rep_02 rep_03 rep_04"
TAU="900.0"
DRY_RUN=false
FORCE=""
MAX_PAR=8     # 8 reps × ~3 GB peak = ~24 GB — well within 256 GB node

while [[ $# -gt 0 ]]; do
    case "$1" in
        --expt)    EXPT="$2";    shift 2 ;;
        --rts)     RTS="$2";     shift 2 ;;
        --reps)    REPS="$2";    shift 2 ;;
        --tau)     TAU="$2";     shift 2 ;;
        --max-par) MAX_PAR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --force)   FORCE="--force"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EXPT_ROOT="$LES_ROOT/experiments/$EXPT"

PAIRS=()
for rt in $RTS; do
    for rep in $REPS; do
        rd="$EXPT_ROOT/$rt/$rep"
        if [[ -f "$rd/couvreux.nc" && -f "$rd/ql.nc" && -f "$rd/w.nc" ]]; then
            PAIRS+=("$rt $rep")
        else
            echo "  SKIP $rt/$rep (missing 3D dump nc files at $rd)"
        fi
    done
done

echo "Experiment : $EXPT"
echo "Pairs      : ${#PAIRS[@]}   max_par=$MAX_PAR"
echo "TAU        : $TAU s"
for pair in "${PAIRS[@]}"; do
    rt="${pair%% *}"; rep="${pair##* }"
    out="$LES_ROOT/analysis/cache/h1_h2/$EXPT/${rt}_${rep}.nc"
    status="compute"
    if [[ -f "$out" && -z "$FORCE" ]]; then
        # Quick header check: skip only if eps_sc already cached
        if python -c "import xarray as xr; ds = xr.open_dataset('$out'); import sys; sys.exit(0 if 'eps_sc' in ds.variables else 1)" 2>/dev/null; then
            status="skip (cache OK)"
        fi
    fi
    printf "  %-12s %-8s  →  %s\n" "$rt" "$rep" "$status"
done

if $DRY_RUN; then
    echo "[dry-run] Would submit 1 sbatch job."
    exit 0
fi

mkdir -p "$LOG_DIR"

_pairs_lines=""
for pair in "${PAIRS[@]}"; do
    _pairs_lines+="    '${pair}'"$'\n'
done

tmp=$(mktemp /tmp/h1h2_XXXXXX.sh)
cat > "$tmp" <<HEADER
#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=00:29:00
#SBATCH --job-name=h1h2_${EXPT}
#SBATCH --output=${LOG_DIR}/h1h2_${EXPT}-%j.out
#SBATCH --error=${LOG_DIR}/h1h2_${EXPT}-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env

MAX_PAR=${MAX_PAR}
EXPT="${EXPT}"
TAU="${TAU}"
FORCE="${FORCE}"
SCRIPT="${SCRIPT_DIR}/compute_h1h2.py"

declare -a PAIRS=(
${_pairs_lines})

HEADER

cat >> "$tmp" <<'LOGIC'
echo "=== h1h2 cache build  expt=${EXPT}  Start: $(date) ==="
echo "Pairs: ${#PAIRS[@]}"

throttle() {
    while [[ $(jobs -rp | wc -l) -ge $MAX_PAR ]]; do sleep 2; done
}

PIDS=()
for pair in "${PAIRS[@]}"; do
    rt="${pair%% *}"; rep="${pair##* }"
    log="${SCRATCH}/CASS_LES/logs/h1h2_${EXPT}_${rt}_${rep}.log"
    echo "START: $rt $rep"
    throttle
    (python "$SCRIPT" --expt "$EXPT" --rt "$rt" --rep "$rep" --tau "$TAU" $FORCE \
        > "$log" 2>&1 \
        && echo "DONE:  $rt $rep" \
        || echo "FAIL:  $rt $rep  (see $log)") &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done
echo "=== h1h2 cache build Done: $(date)   failures=$FAIL ==="
exit $FAIL
LOGIC

chmod +x "$tmp"
echo "Generated sbatch script: $tmp"
echo "Submitting ..."
sbatch "$tmp"
