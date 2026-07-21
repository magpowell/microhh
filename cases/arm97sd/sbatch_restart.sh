#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --output=/pscratch/sd/m/mpowell/ARM97SD_LES/logs/arm97sd_restart-%j.out
#SBATCH --error=/pscratch/sd/m/mpowell/ARM97SD_LES/logs/arm97sd_restart-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266
#
# Restart variant of sbatch_runs.sh (adapted from the CASS restart workflow):
#   - Does NOT delete restart dumps or arm97sd.out; does NOT run init.
#   - Auto-detects the latest savetime from thl.NNNNNNN, verifies all
#     prognostic vars exist there, patches [time] starttime, then runs.
#   - Skips dirs whose run already completed (starttime would equal endtime).
#
# Post-processing after any restart: pass `-t0 0` to 3d_to_nc.py /
# cross_to_nc.py to convert the full timeline, not the restart segment.
#
# --qos, --constraint, --time, --job-name passed at submit time.
# SIM_DIRS: colon-separated run directories, via --export.

mkdir -p /pscratch/sd/m/mpowell/ARM97SD_LES/logs

IFS=':' read -ra DIRS <<< "$SIM_DIRS"
echo "Restarting ${#DIRS[@]} arm97sd simulations at $(date)"

srun -n ${#DIRS[@]} bash -c '
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    IFS=":" read -ra DIRS <<< "$SIM_DIRS"
    dir="${DIRS[$SLURM_LOCALID]}"
    [[ -z "$dir" ]] && exit 0
    cd "$dir"
    echo "[GPU $SLURM_LOCALID] Restart in $(pwd) at $(date)"

    last_t=$(ls thl.[0-9][0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null \
              | sort | tail -1 | awk -F. "{print \$2}")
    if [[ -z "$last_t" ]]; then
        echo "[GPU $SLURM_LOCALID] No restart files in $dir, aborting"
        exit 1
    fi
    restart_t=$(echo "$last_t" | sed "s/^0*//")
    [[ -z "$restart_t" ]] && restart_t=0

    endtime=$(awk -F= "/^endtime/{gsub(/[ .]/,\"\",\$2); print \$2}" arm97sd.ini)
    if [[ -n "$endtime" && "$restart_t" -ge "$endtime" ]]; then
        echo "[GPU $SLURM_LOCALID] Already complete (t=${restart_t} >= ${endtime}), skipping"
        exit 0
    fi

    # nsw6 prognostic set (no ql restart var in this config; sat-adjust diagnosed)
    for v in thl qt w u v qr qs qg; do
        if [[ ! -f "${v}.${last_t}" ]]; then
            echo "[GPU $SLURM_LOCALID] Missing ${v}.${last_t} in $dir, aborting"
            exit 1
        fi
    done

    if [[ ! -f "arm97sd.ini.before_restart" ]]; then
        cp arm97sd.ini "arm97sd.ini.before_restart"
    fi
    sed -i "s/^starttime = [0-9.eE+-]*/starttime = ${restart_t}./" arm97sd.ini

    grep "^starttime" arm97sd.ini
    echo "[GPU $SLURM_LOCALID] Restarting at sim t=${restart_t}s (CST $(awk "BEGIN{printf \"%.2f\", 5.5 + ${restart_t}/3600}"))"

    ./microhh run arm97sd
    rc=$?
    echo "[GPU $SLURM_LOCALID] Done at $(date), exit=${rc}"
'

echo "All restart simulations complete at $(date)"
