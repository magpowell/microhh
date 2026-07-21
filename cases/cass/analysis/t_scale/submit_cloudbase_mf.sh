#!/bin/bash
# submit_cloudbase_mf.sh
#
# Per-cloud cloud-base mass flux (tagged by tobac track age & birth-LST) for
# all reps of one experiment (2stream + raytracer) on one CPU node, parallel.
#
# Needs, per rep:  3-D dumps w.nc & ql.nc  AND  the tobac
# cloud_track_features.nc (produce the latter with submit_cloud_lifetimes.sh).
# Reps with existing cb_mf.nc are skipped unless --force.
#
# Usage:
#   bash submit_cloudbase_mf.sh --expt no_aerosols_zero_wind
#   bash submit_cloudbase_mf.sh --expt no_aerosols_zero_wind --debug   # qos=debug, 30 min
#   bash submit_cloudbase_mf.sh --expt no_aerosols_zero_wind --dry-run
#   bash submit_cloudbase_mf.sh --expt no_aerosols_zero_wind --force

set -euo pipefail

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
LES_ROOT="$SCRATCH/CASS_LES"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$LES_ROOT/logs"

EXPT=""
DRY_RUN=false
FORCE=""
QL_THR="1e-5"
MAX_PAR=4                       # 3-D dump IO is heavier than the 2-D tracking
QOS="regular"
WALL="04:00:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --expt)    EXPT="$2";        shift 2 ;;
        --ql-thr)  QL_THR="$2";      shift 2 ;;
        --debug)   QOS="debug"; WALL="00:30:00"; shift ;;
        --dry-run) DRY_RUN=true;     shift ;;
        --force)   FORCE="--force";  shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXPT" ]]; then
    echo "Usage: bash submit_cloudbase_mf.sh --expt <name> [--ql-thr V] [--dry-run] [--force]"
    exit 1
fi

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
        w3="$EXPT_ROOT/$rt/rep_$rep/w.nc"
        ql3="$EXPT_ROOT/$rt/rep_$rep/ql.nc"
        feat="$LES_ROOT/analysis/lifetime/$EXPT/$rt/rep_$rep/cloud_track_features.nc"
        if [[ -f "$w3" && -f "$ql3" && -f "$feat" ]]; then
            PAIRS+=("$rt $rep")
        fi
    done
done

echo "Experiment : $EXPT"
echo "ql_thr     : $QL_THR"
echo "QOS / wall : $QOS / $WALL"
echo "Reps found : ${#PAIRS[@]}  (need w.nc, ql.nc, cloud_track_features.nc)"
for pair in "${PAIRS[@]}"; do
    rt="${pair%% *}"; rep="${pair##* }"
    out="$LES_ROOT/analysis/cloudbase_mf/$EXPT/$rt/rep_$rep/cb_mf.nc"
    status="compute"
    if [[ -f "$out" && -z "$FORCE" ]]; then status="skip (exists)"; fi
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

tmp=$(mktemp /tmp/cloudbase_mf_XXXXXX.sh)
cat > "$tmp" <<HEADER
#!/bin/bash
#SBATCH --qos=${QOS}
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=${WALL}
#SBATCH --job-name=cb_mf_${EXPT_TAG}
#SBATCH --output=${LOG_DIR}/cb_mf_${EXPT_TAG}-%j.out
#SBATCH --error=${LOG_DIR}/cb_mf_${EXPT_TAG}-%j.err
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
QL_THR="${QL_THR}"
SCRIPT="${SCRIPT_DIR}/compute_cloudbase_mf.py"

declare -a PAIRS=(
${_pairs_lines})

HEADER

cat >> "$tmp" <<'LOGIC'
echo "=== cloudbase_mf  expt=${EXPT}  Start: $(date) ==="
echo "Total (rt, rep) pairs: ${#PAIRS[@]}"

PIDS=()
FAIL=0

throttle() {
    while [[ $(jobs -rp | wc -l) -ge $MAX_PAR ]]; do sleep 2; done
}

for pair in "${PAIRS[@]}"; do
    rt="${pair%% *}"
    rep="${pair##* }"
    log="${SCRATCH}/CASS_LES/logs/cb_mf_${EXPT_TAG}_${rt}_rep${rep}.log"
    echo "START: $rt  rep_$rep"
    throttle
    (python "$SCRIPT" --expt "$EXPT" --rt "$rt" --rep "${rep#0}" \
            --ql-thr "$QL_THR" $FORCE \
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
echo "Submitted: $jid   log: $LOG_DIR/cb_mf_${EXPT_TAG}-${jid}.out"
