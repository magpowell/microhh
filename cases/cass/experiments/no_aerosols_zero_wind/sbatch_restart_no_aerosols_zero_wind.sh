#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --output=/pscratch/sd/m/mpowell/CASS_LES/logs/no_aerosols_zero_wind_restart-%j.out
#SBATCH --error=/pscratch/sd/m/mpowell/CASS_LES/logs/no_aerosols_zero_wind_restart-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266
#
# Restart variant of sbatch_no_aerosols_zero_wind.sh.  Differences:
#   - Does NOT delete *.0000000 binary dumps / xy / cass.out (these are the
#     restart files from the previous run + the timeseries we want to keep).
#   - Does NOT call `microhh init cass` (would regenerate t=0 state).
#   - Auto-detects the latest restart timestamp from `couvreux.NNNNNNN`,
#     verifies all required prognostic vars exist there, then patches
#     [time] starttime in cass.ini before calling `microhh run cass`.
#   - The previous cass.ini is preserved at cass.ini.before_restart_<t>.
#
# After this run finishes, when running 3d_to_nc.py / cross_to_nc.py,
# pass `-t0 0` to override the patched starttime so the full simulation
# (t=0 → endtime) is converted, not just the restart segment.
#
# --constraint, --time, --job-name passed at submit time
# SIM_DIRS: colon-separated list of run directories (passed via --export)

mkdir -p /pscratch/sd/m/mpowell/CASS_LES/logs

IFS=':' read -ra DIRS <<< "$SIM_DIRS"
echo "Restarting ${#DIRS[@]} no_aerosols_zero_wind simulations at $(date)"

srun -n ${#DIRS[@]} bash -c '
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    IFS=":" read -ra DIRS <<< "$SIM_DIRS"
    dir="${DIRS[$SLURM_LOCALID]}"
    [[ -z "$dir" ]] && exit 0
    cd "$dir"
    echo "[GPU $SLURM_LOCALID] Restart in $(pwd) at $(date)"

    # Find latest restart point from couvreux dumps (any prognostic var would do)
    last_t=$(ls couvreux.[0-9][0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null \
              | sort | tail -1 | awk -F. "{print \$2}")
    if [[ -z "$last_t" ]]; then
        echo "[GPU $SLURM_LOCALID] No restart files in $dir, aborting"
        exit 1
    fi
    restart_t=$(echo "$last_t" | sed "s/^0*//")  # strip leading zeros
    [[ -z "$restart_t" ]] && restart_t=0

    # Verify all required prognostic vars are present at that timestamp
    for v in thl qt ql w u v couvreux qr nr; do
        if [[ ! -f "${v}.${last_t}" ]]; then
            echo "[GPU $SLURM_LOCALID] Missing ${v}.${last_t} in $dir, aborting"
            exit 1
        fi
    done

    # Patch cass.ini once.  If a backup already exists from a prior restart,
    # do not overwrite it — preserve the original starttime=0 baseline.
    if [[ ! -f "cass.ini.before_restart" ]]; then
        cp cass.ini "cass.ini.before_restart"
    fi
    sed -i "s/^starttime = [0-9.eE+-]*/starttime = ${restart_t}./" cass.ini

    grep "^starttime" cass.ini
    echo "[GPU $SLURM_LOCALID] Restart at sim t=${restart_t} (LST $(awk "BEGIN{printf \"%.2f\", 5.5 + ${restart_t}/3600}"))"

    ./microhh run cass
    rc=$?
    echo "[GPU $SLURM_LOCALID] Done at $(date), exit=${rc}"
'

echo "All restart simulations complete at $(date)"
