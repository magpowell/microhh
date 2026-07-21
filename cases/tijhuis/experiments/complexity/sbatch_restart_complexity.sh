#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --account=m1266
#SBATCH --output=/pscratch/sd/m/mpowell/SENSITIVITY_LES_CUMULUS/complexity_20140325/mhh-restart-%j.out
#SBATCH --error=/pscratch/sd/m/mpowell/SENSITIVITY_LES_CUMULUS/complexity_20140325/mhh-restart-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=ALL
#
# Required env vars (passed via --export from submit_restart_complexity.sh):
#   SIM_DIRS      : colon-separated list of seed simulation directories
#   RESTART_TIME  : checkpoint time in seconds to restart from

IFS=':' read -ra DIRS <<< "$SIM_DIRS"

echo "=== Complexity restart job ==="
echo "  Restart time : ${RESTART_TIME}s"
echo "  Simulations  : ${#DIRS[@]}"
echo "  Started at   : $(date)"

# Run simulations (no init, just run from restart)
srun -n ${#DIRS[@]} /bin/bash -c '
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    IFS=":" read -ra DIRS <<< "$SIM_DIRS"
    dir="${DIRS[$SLURM_LOCALID]}"
    [[ -z "$dir" ]] && exit 0
    cd "$dir"
    echo "[Task $SLURM_LOCALID] GPU=$CUDA_VISIBLE_DEVICES RESTART in $(pwd) at $(date)"
    ./microhh run cabauw
    echo "[Task $SLURM_LOCALID] Simulation complete at $(date)"
'

echo "All simulations complete at $(date)"

# Post-processing: convert cross-sections to NetCDF
echo "Starting post-processing at $(date)"
module load conda
conda activate xr_env

for i in "${!DIRS[@]}"; do
    dir="${DIRS[$i]}"
    [[ -z "$dir" ]] && continue
    (
        cd "$dir"
        echo "[Task $i] Converting cross sections at $(date)"
        python cross_to_nc.py -f cabauw.ini -m xy -t0 0
        echo "[Task $i] Post-processing complete at $(date)"
    ) &
done

wait
echo "All tasks complete at $(date)"
