#!/bin/bash
# submit_cloud_root_v2.sh
#
# Regenerate cloud-root composites for no_aerosols_zero_wind_v2 ONLY, after the
# local-solar-time fix in cass_analysis.py (dump_t / lst now in solar time).
#
# Single regular-QOS job, three phases over all 8 reps (4x 2stream, 4x raytracer):
#   1. ensure_inputs   3d_to_nc for any missing field .nc (raytracer rep_03/04
#                      have the raw thl/qt dumps but never made thl.nc/qt.nc)
#   2. composite_prep  events_*.nc        (binned in solar time now)
#   3. sw_composite    sw_composite.nc
#
# Usage: bash submit_cloud_root_v2.sh

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
RUN_ROOT="$LES_ROOT/experiments/no_aerosols_zero_wind_v2"
OUT_ROOT="$LES_ROOT/analysis/cloud_root_composite/no_aerosols_zero_wind_v2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$LES_ROOT/logs"
mkdir -p "$LOG_DIR"

REPS=(
  "2stream/rep_01" "2stream/rep_02" "2stream/rep_03" "2stream/rep_04"
  "raytracer/rep_01" "raytracer/rep_02" "raytracer/rep_03" "raytracer/rep_04"
)

_rep_lines=""
for r in "${REPS[@]}"; do _rep_lines+="    '${r}'"$'\n'; done

tmp=$(mktemp /tmp/cr_v2_XXXXXX.sh)
cat > "$tmp" <<HEADER
#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=460G
#SBATCH --time=00:30:00
#SBATCH --job-name=cr_v2_solar
#SBATCH --output=${LOG_DIR}/cr_v2_solar-%j.out
#SBATCH --error=${LOG_DIR}/cr_v2_solar-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266

set -euo pipefail
module load conda
conda activate xr_env
export OMP_NUM_THREADS=4

PREP='${SCRIPT_DIR}/cloud_root_composite_prep.py'
SW='${SCRIPT_DIR}/sw_surface_composite.py'
RUN_ROOT='${RUN_ROOT}'
OUT_ROOT='${OUT_ROOT}'
MAX_PAR=8

declare -a REPS=(
${_rep_lines})
HEADER

cat >> "$tmp" <<'LOGIC'
throttle() { while [[ $(jobs -rp | wc -l) -ge $MAX_PAR ]]; do sleep 3; done; }
FAIL=0

echo "=== phase 1: ensure 3d_to_nc inputs ==="; echo "Start: $(date)"
PIDS=()
for r in "${REPS[@]}"; do
    rd="$RUN_ROOT/$r"
    missing=()
    for fld in thl qt ql w b u v; do [[ -f "$rd/$fld.nc" ]] || missing+=("$fld"); done
    [[ ${#missing[@]} -eq 0 ]] && { echo "INPUTS OK: $r"; continue; }
    echo "3d_to_nc START: $r  (${missing[*]})"; throttle
    (cd "$rd" && python 3d_to_nc.py -v "${missing[@]}" >> "$rd/3d_to_nc.log" 2>&1 \
        && echo "3d_to_nc DONE: $r" || echo "3d_to_nc FAIL: $r") &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
[[ $FAIL -eq 0 ]] || { echo "3d_to_nc failed — see 3d_to_nc.log"; exit 1; }

echo "=== phase 2: composite_prep (events, solar) ==="; echo "$(date)"
PIDS=()
for r in "${REPS[@]}"; do
    rd="$RUN_ROOT/$r"; od="$OUT_ROOT/$r"; mkdir -p "$od"
    echo "PREP START: $r"; throttle
    (python "$PREP" --run-dir "$rd" --output-dir "$od" >> "${od}/composite.log" 2>&1 \
        && echo "PREP DONE: $r" || echo "PREP FAIL: $r") &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
[[ $FAIL -eq 0 ]] || { echo "prep failed — see composite.log"; exit 1; }

echo "=== phase 3: sw_surface_composite (solar) ==="; echo "$(date)"
PIDS=()
for r in "${REPS[@]}"; do
    rd="$RUN_ROOT/$r"; od="$OUT_ROOT/$r"; rt=$(dirname "$r")
    [[ -f "$od/events_xz.nc" || -f "$od/events_yz.nc" ]] || { echo "SW SKIP (no events): $r"; continue; }
    echo "SW START: $rt $r"; throttle
    (python "$SW" --run-dir "$rd" --rt "$rt" --comp-dir "$od" >> "${od}/sw_composite.log" 2>&1 \
        && echo "SW DONE: $r" || echo "SW FAIL: $r (see ${od}/sw_composite.log)") &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
echo "Done: $(date)"
[[ $FAIL -eq 0 ]] || { echo "sw_composite failed — see sw_composite.log"; exit 1; }
LOGIC

jid=$(sbatch "$tmp" | awk '{print $NF}')
echo "Submitted cr_v2_solar: $jid"
echo "Logs: $LOG_DIR/cr_v2_solar-${jid}.{out,err}"
rm -f "$tmp"
