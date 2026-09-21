#!/bin/bash
#SBATCH --nodes=1
#
# Restart variant of sbatch_run_alpha.sh for Empire AI Alpha. Shared by every
# CASS experiment; chained by shared/submit_alpha.sh after each raytracer job
# with --dependency=afterany so it runs whether the parent finished or hit the
# wall.
#   - Does NOT delete restart dumps or cass.out; does NOT run init.
#   - Auto-detects the latest savetime from the restart dumps, verifies all
#     prognostic vars exist there, patches [time] starttime, then runs.
#   - Skips dirs whose run already completed (nothing to do after a clean
#     parent), and post-processes only when it actually ran.
#
# After any restart: pass `-t0 0` to 3d_to_nc.py / cross_to_nc.py to convert
# the full timeline, not the restart segment. The original ini is kept as
# cass.ini.before_restart.
#
# All site-specific sbatch flags come from the submitting script, since
# #SBATCH cannot expand $SCRATCH. Exported via --export:
#   MICROHH_DIR    repo root (sbatch copies this file into the spool dir).
#   SIM_DIRS       colon-separated run directories.
#   RESTART_VARS   space-separated prognostic set that must exist at the
#                  restart time. Default is the tracer-bearing rerun config.

set -uo pipefail

: "${MICROHH_DIR:?must be exported by the submitting script (--export=ALL,MICROHH_DIR=...)}"
: "${SIM_DIRS:?must be exported by the submitting script}"
export RESTART_VARS="${RESTART_VARS:-thl qt ql w u v couvreux qr nr}"
source "$MICROHH_DIR/config/empireai_alpha_env.sh"

IFS=':' read -ra DIRS <<< "$SIM_DIRS"
echo "Restarting ${#DIRS[@]} CASS simulation(s) at $(date)"

srun -n ${#DIRS[@]} bash -c '
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    IFS=":" read -ra DIRS <<< "$SIM_DIRS"
    dir="${DIRS[$SLURM_LOCALID]}"
    [[ -z "$dir" ]] && exit 0
    cd "$dir" || exit 1
    echo "[GPU $SLURM_LOCALID] Restart in $(pwd) at $(date)"

    read -ra VARS <<< "$RESTART_VARS"
    ref="${VARS[0]}"
    last_t=$(ls ${ref}.[0-9][0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null \
              | sort | tail -1 | awk -F. "{print \$2}")
    if [[ -z "$last_t" ]]; then
        echo "[GPU $SLURM_LOCALID] No restart files in $dir, aborting"
        exit 1
    fi
    restart_t=$(echo "$last_t" | sed "s/^0*//")
    [[ -z "$restart_t" ]] && restart_t=0

    endtime=$(awk -F= "/^endtime/{gsub(/[ .]/,\"\",\$2); print \$2}" cass.ini)
    if [[ -n "$endtime" && "$restart_t" -ge "$endtime" ]]; then
        echo "[GPU $SLURM_LOCALID] Already complete (t=${restart_t} >= ${endtime}), skipping"
        exit 0
    fi

    for v in "${VARS[@]}"; do
        if [[ ! -f "${v}.${last_t}" ]]; then
            echo "[GPU $SLURM_LOCALID] Missing ${v}.${last_t} in $dir, aborting"
            exit 1
        fi
    done

    if [[ ! -f "cass.ini.before_restart" ]]; then
        cp cass.ini "cass.ini.before_restart"
    fi
    sed -i "s/^starttime = [0-9.eE+-]*/starttime = ${restart_t}./" cass.ini
    grep "^starttime" cass.ini
    echo "[GPU $SLURM_LOCALID] Restarting at sim t=${restart_t}s"

    ./microhh run cass
    rc=$?
    echo "[GPU $SLURM_LOCALID] Done at $(date), exit=${rc}"
    # Signal the post-processing step below that this dir actually ran.
    touch .restarted_by_${SLURM_JOB_ID:-manual}
    exit $rc
'
rc=$?
echo "All restart simulations complete at $(date) (srun exit $rc)"

echo "Post-processing with $XR_PY (full timeline, -t0 0) ..."
for dir in "${DIRS[@]}"; do
    [[ -z "$dir" ]] && continue
    marker="$dir/.restarted_by_${SLURM_JOB_ID:-manual}"
    [[ -f "$marker" ]] || { echo "[$dir] not restarted here, skipping post-processing"; continue; }
    rm -f "$marker"
    (
        cd "$dir" || exit 1
        "$XR_PY" cross_to_nc.py -f cass.ini -m xy -t0 0
        echo "[$dir] cross_to_nc done (exit $?)"
    ) &
done
wait
echo "All tasks complete at $(date)"
exit $rc
