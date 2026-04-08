#!/bin/bash
# submit_nazw_circ.sh
#
# Targeted submission for no_aerosols_zero_wind only.
# Job 1: 3d_to_nc -v thl qt ql w b u v  (skips if b.nc u.nc v.nc all exist)
# Job 2: cloud_root_composite_prep       (depends on Job 1)
#
# Usage:
#   bash submit_nazw_circ.sh [--dry-run]

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
COMPOSITE_ROOT="$LES_ROOT/analysis/cloud_root_composite"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$LES_ROOT/logs"
EXPT="no_aerosols_zero_wind"

DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR" "$COMPOSITE_ROOT"

RUN_DIRS=()
OUT_DIRS=()

for rt in 2stream raytracer; do
    for rep in 01 02 03 04; do
        rd="$LES_ROOT/experiments/$EXPT/$rt/rep_$rep"
        if [[ -d "$rd" ]]; then
            RUN_DIRS+=("$rd")
            OUT_DIRS+=("$COMPOSITE_ROOT/$EXPT/$rt/rep_$rep")
        fi
    done
done

N=${#RUN_DIRS[@]}
echo "Reps to process: $N"
for rd in "${RUN_DIRS[@]}"; do printf "  %s\n" "$rd"; done

if $DRY_RUN; then
    echo "[dry-run] Would submit 2 jobs for $N reps."
    exit 0
fi

_run_array_lines=""
_out_array_lines=""
for rd in "${RUN_DIRS[@]}"; do _run_array_lines+="    '${rd}'"$'\n'; done
for od in "${OUT_DIRS[@]}"; do _out_array_lines+="    '${od}'"$'\n'; done

# ── Job 1: 3d_to_nc ───────────────────────────────────────────────────────────
tmp_3d=$(mktemp /tmp/nazw_3d_to_nc_XXXXXX.sh)
cat > "$tmp_3d" <<HEADER
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --job-name=nazw_3d_to_nc
#SBATCH --output=${LOG_DIR}/nazw_3d_to_nc-%j.out
#SBATCH --error=${LOG_DIR}/nazw_3d_to_nc-%j.err
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env

declare -a RUN_DIRS=(
${_run_array_lines})

HEADER

cat >> "$tmp_3d" <<'LOGIC'
echo "=== 3d_to_nc: no_aerosols_zero_wind ==="
echo "Start: $(date)"

PIDS=(); FAIL=0
throttle() { while [[ $(jobs -rp | wc -l) -ge 8 ]]; do sleep 2; done; }

for rd in "${RUN_DIRS[@]}"; do
    if [[ -f "$rd/b.nc" && -f "$rd/u.nc" && -f "$rd/v.nc" ]]; then
        echo "SKIP (b.nc u.nc v.nc exist): $rd"
        continue
    fi
    echo "START: $rd"
    throttle
    (cd "$rd" && python 3d_to_nc.py -v thl qt ql w b u v \
        >> "${rd}/3d_to_nc.log" 2>&1 \
        && echo "DONE: $rd" || echo "FAIL: $rd") &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
echo "Done: $(date)"
[[ $FAIL -eq 0 ]] || { echo "One or more 3d_to_nc calls failed."; exit 1; }
LOGIC

# ── Job 2: composite prep ─────────────────────────────────────────────────────
tmp_cr=$(mktemp /tmp/nazw_composite_XXXXXX.sh)
cat > "$tmp_cr" <<HEADER
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=460G
#SBATCH --time=04:00:00
#SBATCH --job-name=nazw_cr_circ
#SBATCH --output=${LOG_DIR}/nazw_cr_circ-%j.out
#SBATCH --error=${LOG_DIR}/nazw_cr_circ-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env
export OMP_NUM_THREADS=4

COMPOSITE_PREP='${SCRIPT_DIR}/cloud_root_composite_prep.py'

declare -a RUN_DIRS=(
${_run_array_lines})
declare -a OUT_DIRS=(
${_out_array_lines})

HEADER

cat >> "$tmp_cr" <<'LOGIC'
echo "=== composite prep: no_aerosols_zero_wind ==="
echo "Start: $(date)"

PIDS=(); FAIL=0
throttle() { while [[ $(jobs -rp | wc -l) -ge 8 ]]; do sleep 5; done; }

for i in "${!RUN_DIRS[@]}"; do
    rd="${RUN_DIRS[$i]}"
    od="${OUT_DIRS[$i]}"
    if [[ -f "$od/events_xz.nc" ]] && \
       ncdump -h "$od/events_xz.nc" 2>/dev/null | grep -q 'b_prime'; then
        echo "SKIP (events_xz.nc has circ fields): $od"
        continue
    fi
    echo "START: $rd"
    mkdir -p "$od"
    throttle
    (python "$COMPOSITE_PREP" --run-dir "$rd" --output-dir "$od" \
        >> "${od}/composite.log" 2>&1 \
        && echo "DONE: $rd" || echo "FAIL: $rd") &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
echo "Done: $(date)"
[[ $FAIL -eq 0 ]] || { echo "One or more composite preps failed."; exit 1; }
LOGIC

# ── Submit ────────────────────────────────────────────────────────────────────
jid_3d=$(sbatch "$tmp_3d" | awk '{print $NF}')
echo "Submitted 3d_to_nc:  $jid_3d"

jid_cr=$(sbatch --dependency=afterok:"$jid_3d" "$tmp_cr" | awk '{print $NF}')
echo "Submitted composite: $jid_cr  (after $jid_3d)"

rm -f "$tmp_3d" "$tmp_cr"
echo ""
echo "2 jobs queued for $N reps. Logs: $LOG_DIR"
