#!/bin/bash
#SBATCH --nodes=1
#
# CASS run body for Empire AI Alpha (single GPU per job). Shared by every CASS
# experiment; submitted by shared/submit_alpha.sh.
#
# Deliberately minimal #SBATCH headers: --account, --partition, --qos, --gres,
# --ntasks-per-node, --cpus-per-task, --time, --output, --error and --job-name
# are ALL passed at submit time. Slurm does not expand environment variables
# inside #SBATCH directives, so hardcoding a log path here is exactly how the
# Perlmutter scripts ended up pinned to /pscratch.
#
# Exported by the submitting script (--export=ALL,...):
#   MICROHH_DIR   repo root; sbatch copies this file into Slurm's spool dir, so
#                 BASH_SOURCE cannot locate the repo from inside the job.
#   SIM_DIRS      colon-separated run directories, one per task. With the
#                 recommended 1-GPU-per-job submission this is a single dir.
#   GPU_MEM_LOG=1 additionally logs nvidia-smi memory every 10 s.
#
# Post-processing (cross_to_nc.py on the xy crosses) runs inside the job under
# $XR_PY from the env script, as on Perlmutter.

set -uo pipefail

: "${MICROHH_DIR:?must be exported by the submitting script (--export=ALL,MICROHH_DIR=...)}"
: "${SIM_DIRS:?must be exported by the submitting script}"

# Compute nodes do not inherit a usable library environment for the netCDF /
# HDF5 / CUDA stack MicroHH links against. Source it explicitly so a job
# behaves the same whether or not the login shell was set up.
source "$MICROHH_DIR/config/empireai_alpha_env.sh"

IFS=':' read -ra DIRS <<< "$SIM_DIRS"
echo "Starting ${#DIRS[@]} CASS simulation(s) at $(date)"
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
    echo "[GPU $SLURM_LOCALID] Clearing previous outputs..."
    rm -f cass.default.*.nc cass.column.*.nc b.xy.nc cass.out
    rm -f *.[0-9][0-9][0-9][0-9][0-9][0-9][0-9]
    rm -f *.xy.* *_kernel.txt
    ./microhh init cass
    ./microhh run cass
    status=$?
    [[ -n "${MEMPID:-}" ]] && kill $MEMPID 2>/dev/null
    echo "[GPU $SLURM_LOCALID] Done (exit $status) at $(date)"
    exit $status
'
rc=$?
echo "All simulations complete at $(date) (srun exit $rc)"

# Post-process the 60 s xy crosses to netCDF. Runs even after a non-zero rc so
# a partial run is still inspectable; the exit code below is still the run's.
echo "Post-processing with $XR_PY ..."
for dir in "${DIRS[@]}"; do
    [[ -z "$dir" ]] && continue
    (
        cd "$dir" || exit 1
        "$XR_PY" cross_to_nc.py -f cass.ini -m xy
        echo "[$dir] cross_to_nc done (exit $?)"
    ) &
done
wait
echo "All tasks complete at $(date)"
exit $rc
