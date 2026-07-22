#!/bin/bash
#SBATCH --nodes=1
#
# GoAmazon composite ShCu run body for Empire AI Alpha.
#
# Deliberately minimal #SBATCH headers: --account, --partition, --gres,
# --ntasks-per-node, --cpus-per-task, --time, --output, --error and --job-name
# are ALL passed at submit time by submit_*_alpha.sh. Slurm does not expand
# environment variables inside #SBATCH directives, so hardcoding a log path
# here is exactly how the Perlmutter scripts ended up pinned to /pscratch.
#
# SIM_DIRS: colon-separated run directories, one per task, via --export.
#           With the recommended 1-GPU-per-job submission this is a single dir.
# GPU_MEM_LOG=1 additionally logs nvidia-smi memory every 10 s (fit test).

set -uo pipefail

# MICROHH_DIR must be supplied by the submitting script via --export.
#
# Do NOT try to derive it from BASH_SOURCE here: sbatch COPIES this script into
# Slurm's spool directory before running it, so inside a batch job BASH_SOURCE
# is /cm/local/apps/slurm/var/spool/job<N>/slurm_script and any path derived
# from it points into the spool dir. That silently breaks the env sourcing
# below, and the failure only surfaces later as a missing libnetcdf.so.19.
: "${MICROHH_DIR:?must be exported by the submitting script (--export=ALL,MICROHH_DIR=...)}"

# Compute nodes do not inherit a usable library environment for the netCDF /
# HDF5 / CUDA stack MicroHH links against, and --export=ALL only carries
# whatever the submitting shell happened to have. Source it explicitly so a
# job behaves the same whether or not the login shell was set up.
source "$MICROHH_DIR/config/empireai_alpha_env.sh"

IFS=':' read -ra DIRS <<< "$SIM_DIRS"
echo "Starting ${#DIRS[@]} goamazon_shcu simulation(s) at $(date)"
echo "Node: $(hostname)  GPUs: ${SLURM_GPUS_ON_NODE:-?}  JobID: ${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

srun -n ${#DIRS[@]} bash -c '
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    IFS=":" read -ra DIRS <<< "$SIM_DIRS"
    dir="${DIRS[$SLURM_LOCALID]}"
    [[ -z "$dir" ]] && exit 0
    cd "$dir" || exit 1
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
    exit $status
'
rc=$?

echo "All simulations complete at $(date) (srun exit $rc)"
exit $rc
