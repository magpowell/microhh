#!/bin/bash
#SBATCH --qos=regular
#SBATCH --output=mhh-%j.out
#SBATCH --error=mhh-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --job-name=SENS_LES
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=ALL
#SBATCH -A m1266

# SIM_DIRS is passed via --export as a colon-separated list of up to 4 directories
IFS=':' read -ra DIRS <<< "$SIM_DIRS"

echo "Starting batch of ${#DIRS[@]} simulations at $(date)"

# Single srun launches all tasks; SLURM_LOCALID (0-3) selects directory and GPU.
srun -n ${#DIRS[@]} bash -c '
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    IFS=":" read -ra DIRS <<< "$SIM_DIRS"
    dir="${DIRS[$SLURM_LOCALID]}"
    if [[ -z "$dir" ]]; then exit 0; fi

    cd "$dir"
    echo "[Task $SLURM_LOCALID] GPU=$CUDA_VISIBLE_DEVICES Running in $(pwd) at $(date)"
    rm -f *00* *.txt cabauw.out *.xy.nc q*.nc thl.nc T.nc w.nc
    ./microhh init cabauw
    ./microhh run cabauw
    echo "[Task $SLURM_LOCALID] Simulation complete at $(date)"
'

echo "All simulations complete at $(date)"

# Post-processing: convert cross-sections to netCDF (no GPU needed)
echo "Starting post-processing at $(date)"
module load conda
conda activate xr_env

for i in "${!DIRS[@]}"; do
    dir="${DIRS[$i]}"
    [[ -z "$dir" ]] && continue
    (
        cd "$dir"
        echo "[Task $i] Converting cross sections at $(date)"
        python cross_to_nc.py -f cabauw.ini -m xy
        echo "[Task $i] Post-processing complete at $(date)"
    ) &
done

wait
echo "All tasks complete at $(date)"
