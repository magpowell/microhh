#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --output=/pscratch/sd/m/mpowell/GOAMAZON_SHCU_LES/logs/goamazon_shcu-%j.out
#SBATCH --error=/pscratch/sd/m/mpowell/GOAMAZON_SHCU_LES/logs/goamazon_shcu-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266
#
# --qos, --constraint, --time, --job-name passed at submit time.
# SIM_DIRS: colon-separated run directories (max 4), via --export.
# GPU_MEM_LOG=1 additionally logs nvidia-smi memory every 10 s (fit test).

mkdir -p /pscratch/sd/m/mpowell/GOAMAZON_SHCU_LES/logs

IFS=':' read -ra DIRS <<< "$SIM_DIRS"
echo "Starting ${#DIRS[@]} goamazon_shcu simulations at $(date)"

srun -n ${#DIRS[@]} bash -c '
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    IFS=":" read -ra DIRS <<< "$SIM_DIRS"
    dir="${DIRS[$SLURM_LOCALID]}"
    [[ -z "$dir" ]] && exit 0
    cd "$dir"
    echo "[GPU $SLURM_LOCALID] Running in $(pwd) at $(date)"
    if [[ "${GPU_MEM_LOG:-0}" == "1" ]]; then
        nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 10 \
            > gpu_mem.log 2>/dev/null &
        MEMPID=$!
    fi
    rm -f goamazon_shcu.default.*.nc goamazon_shcu.column.*.nc goamazon_shcu.out
    rm -f *.[0-9][0-9][0-9][0-9][0-9][0-9][0-9]
    rm -f *.xy.* *_kernel.txt
    ./microhh init goamazon_shcu
    ./microhh run goamazon_shcu
    status=$?
    [[ -n "${MEMPID:-}" ]] && kill $MEMPID 2>/dev/null
    echo "[GPU $SLURM_LOCALID] Done (exit $status) at $(date)"
'

echo "All simulations complete at $(date)"
