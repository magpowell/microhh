#!/bin/bash
# Submit a single sbatch job that consolidates raw MicroHH 3D dumps into
# per-variable netCDF files (ql.nc, w.nc, couvreux.nc) for the requested
# (experiment, rt, rep) tuples in parallel.
#
# Usage:
#   bash submit_3d_to_nc.sh                                     # default: no_aerosols_zero_wind_v2 raytracer rep_03 rep_04
#   bash submit_3d_to_nc.sh --expt EXPT --rt RT --reps "rep_03 rep_04"
#   bash submit_3d_to_nc.sh --vars "ql w couvreux" ...
#   bash submit_3d_to_nc.sh --dry-run

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
LOG_DIR="$LES_ROOT/logs"

EXPT="no_aerosols_zero_wind_v2"
RT="raytracer"
REPS="rep_03 rep_04"
VARS="ql w couvreux"
DRY_RUN=false
MAX_PAR=2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --expt)    EXPT="$2";    shift 2 ;;
        --rt)      RT="$2";      shift 2 ;;
        --reps)    REPS="$2";    shift 2 ;;
        --vars)    VARS="$2";    shift 2 ;;
        --max-par) MAX_PAR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EXPT_ROOT="$LES_ROOT/experiments/$EXPT"

# Sanity: each rep dir must exist with cass.ini and 3d_to_nc.py
PAIRS=()
for rep in $REPS; do
    rd="$EXPT_ROOT/$RT/$rep"
    if [[ -f "$rd/cass.ini" && -f "$rd/3d_to_nc.py" ]]; then
        PAIRS+=("$rep")
    else
        echo "  SKIP $RT/$rep (missing cass.ini or 3d_to_nc.py at $rd)"
    fi
done

echo "Experiment : $EXPT"
echo "RT         : $RT"
echo "Reps       : ${PAIRS[*]:-NONE}"
echo "Vars       : $VARS"
echo "MAX_PAR    : $MAX_PAR"

if [[ ${#PAIRS[@]} -eq 0 ]]; then
    echo "Nothing to do."; exit 0
fi

if $DRY_RUN; then
    echo "[dry-run] would submit 1 sbatch job covering ${#PAIRS[@]} reps."
    exit 0
fi

mkdir -p "$LOG_DIR"

_pairs_lines=""
for rep in "${PAIRS[@]}"; do
    _pairs_lines+="    '${rep}'"$'\n'
done

tmp=$(mktemp /tmp/3d_to_nc_XXXXXX.sh)
cat > "$tmp" <<HEADER
#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=00:29:00
#SBATCH --job-name=3dnc_${EXPT}_${RT}
#SBATCH --output=${LOG_DIR}/3dnc_${EXPT}_${RT}-%j.out
#SBATCH --error=${LOG_DIR}/3dnc_${EXPT}_${RT}-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env

MAX_PAR=${MAX_PAR}
EXPT="${EXPT}"
RT="${RT}"
EXPT_ROOT="${EXPT_ROOT}"
VARS="${VARS}"

declare -a PAIRS=(
${_pairs_lines})

HEADER

cat >> "$tmp" <<'LOGIC'
echo "=== 3d_to_nc  expt=${EXPT}  rt=${RT}  Start: $(date) ==="
echo "Reps: ${PAIRS[*]}"
echo "Vars: ${VARS}"

throttle() {
    while [[ $(jobs -rp | wc -l) -ge $MAX_PAR ]]; do sleep 2; done
}

PIDS=()
for rep in "${PAIRS[@]}"; do
    rd="${EXPT_ROOT}/${RT}/${rep}"
    log="${SCRATCH}/CASS_LES/logs/3dnc_${EXPT}_${RT}_${rep}.log"
    echo "START: ${RT}/${rep}  → log: $log"
    throttle
    (
        cd "$rd"
        # -t0 0 forces conversion from t=0 even if cass.ini was patched with
        # starttime > 0 on a restart (otherwise only post-restart steps are
        # converted — see CLAUDE.md project notes on restarts).
        python 3d_to_nc.py -v $VARS -f cass.ini -t0 0 > "$log" 2>&1 \
            && echo "DONE:  ${RT}/${rep}" \
            || echo "FAIL:  ${RT}/${rep}  (see $log)"
    ) &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done
echo "=== 3d_to_nc Done: $(date)   failures=$FAIL ==="
exit $FAIL
LOGIC

chmod +x "$tmp"
echo "Generated sbatch script: $tmp"
echo "Submitting ..."
sbatch "$tmp"
