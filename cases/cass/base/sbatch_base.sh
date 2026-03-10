#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --job-name=CASS_base
#SBATCH --output=/pscratch/sd/m/mpowell/CASS_LES/logs/base-%j.out
#SBATCH --error=/pscratch/sd/m/mpowell/CASS_LES/logs/base-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266
#
# --constraint is passed at submit time by submit_base.sh
# SIM_DIRS is a colon-separated list of 4 run directories, passed via --export

mkdir -p /pscratch/sd/m/mpowell/CASS_LES/logs

IFS=':' read -ra DIRS <<< "$SIM_DIRS"
echo "Starting ${#DIRS[@]} CASS base simulations at $(date)"

srun -n ${#DIRS[@]} bash -c '
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    IFS=":" read -ra DIRS <<< "$SIM_DIRS"
    dir="${DIRS[$SLURM_LOCALID]}"
    [[ -z "$dir" ]] && exit 0
    cd "$dir"
    echo "[GPU $SLURM_LOCALID] Running in $(pwd) at $(date)"
    ./microhh init cass
    ./microhh run cass
    echo "[GPU $SLURM_LOCALID] Done at $(date)"
'

echo "All simulations complete at $(date)"

echo "Post-processing..."
module load conda
conda activate xr_env

for i in "${!DIRS[@]}"; do
    dir="${DIRS[$i]}"
    [[ -z "$dir" ]] && continue
    (
        cd "$dir"
        python cross_to_nc.py -f cass.ini -m xy
        echo "[Task $i] Post-processing done"
    ) &
done

wait
echo "All tasks complete at $(date)"
