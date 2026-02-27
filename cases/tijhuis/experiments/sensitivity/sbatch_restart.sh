#!/bin/bash
#SBATCH --qos=regular
#SBATCH --output=mhh-restart-%j.out
#SBATCH --error=mhh-restart-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --job-name=SENS_RESTART
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -A m1266

# --- Restart sbatch script for microHH ---
# Unlike sbatch_sensitivity.sh, this script:
#   1. Does NOT delete any existing output/restart files
#   2. Does NOT run `microhh init` (skips straight to `run`)
#   3. Patches starttime in each .ini to the requested RESTART_TIME
#
# Required environment variables (passed via --export):
#   SIM_DIRS      : colon-separated list of simulation directories
#   RESTART_TIME  : the checkpoint time (in seconds) to restart from
#   INI_NAME      : (optional) name of the .ini file, defaults to "cabauw"

IFS=':' read -ra DIRS <<< "$SIM_DIRS"
INI_NAME="${INI_NAME:-cabauw}"

echo "=== Restart job ==="
echo "  Restart time : $RESTART_TIME"
echo "  INI file     : ${INI_NAME}.ini"
echo "  Simulations  : ${#DIRS[@]}"
echo "  Started at   : $(date)"

# Validate restart files exist before launching GPUs
for dir in "${DIRS[@]}"; do
    [[ -z "$dir" ]] && continue
    restart_file="$dir/time.$(printf '%07d' "$RESTART_TIME")"
    if [[ ! -f "$restart_file" ]]; then
        echo "ERROR: Missing restart file $restart_file — aborting."
        exit 1
    fi
done

# Patch starttime in each .ini file
for dir in "${DIRS[@]}"; do
    [[ -z "$dir" ]] && continue
    ini="$dir/${INI_NAME}.ini"
    if [[ ! -f "$ini" ]]; then
        echo "ERROR: Missing $ini — aborting."
        exit 1
    fi
    sed -i "s/^starttime = .*/starttime = ${RESTART_TIME}/" "$ini"
    echo "  Patched starttime = $RESTART_TIME in $ini"
done

# Run simulations (no init, just run)
srun -n ${#DIRS[@]} bash -c '
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    IFS=":" read -ra DIRS <<< "$SIM_DIRS"
    dir="${DIRS[$SLURM_LOCALID]}"
    if [[ -z "$dir" ]]; then exit 0; fi

    cd "$dir"
    echo "[Task $SLURM_LOCALID] GPU=$CUDA_VISIBLE_DEVICES RESTART in $(pwd) at $(date)"
    ./microhh run '"$INI_NAME"'
    echo "[Task $SLURM_LOCALID] Simulation complete at $(date)"
'

echo "All simulations complete at $(date)"

# Post-processing: convert cross-sections to netCDF
echo "Starting post-processing at $(date)"
module load conda
conda activate xr_env

for i in "${!DIRS[@]}"; do
    dir="${DIRS[$i]}"
    [[ -z "$dir" ]] && continue
    (
        cd "$dir"
        echo "[Task $i] Converting cross sections at $(date)"
        python cross_to_nc.py -f ${INI_NAME}.ini -m xy -t0 0
        echo "[Task $i] Post-processing complete at $(date)"
    ) &
done

wait
echo "All tasks complete at $(date)"
