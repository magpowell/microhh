#!/bin/bash
# Site layer: detect the machine and export one consistent set of variables so
# the case scripts (cases/cass/shared/submit.sh and the sbatch bodies, the
# setup_*.py scripts) run unchanged on NERSC Perlmutter and Empire AI Alpha.
#
# Usage:  source config/site_env.sh          (auto-detect)
#         SITE=perlmutter source config/site_env.sh
#
# Exports:
#   SITE            perlmutter | empireai_alpha
#   MICROHH_DIR     repo root (from this file's location)
#   SCRATCH         run output root (must exist on both sites)
#   XR_PY           Python with xarray/netCDF4/scipy for setup and post-processing
#   MICROHH_EXEC    binary the setup scripts link (default build_gpu/microhh)
#   SITE_SBATCH_COMMON        array-as-string: account/partition flags
#   SITE_SBATCH_2STREAM       extra flags for 2stream legs (constraint etc.)
#   SITE_SBATCH_RAYTRACER     extra flags for raytracer legs
#   SITE_QOS_PROD / SITE_QOS_DEBUG     QoS names (verified per site, see below)
#   SITE_MAXWALL_PROD         wall-time cap of SITE_QOS_PROD (hh:mm:ss)
#   SITE_DEBUG_WALLTIME       wall time for debug legs
#   SITE_CPUS_PER_GPU         --cpus-per-task for a single-GPU job
#
# Every value can be overridden by exporting it before sourcing.

_site_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MICROHH_DIR="${MICROHH_DIR:-$_site_root}"

if [[ -z "${SITE:-}" ]]; then
    if [[ "${NERSC_HOST:-}" == "perlmutter" ]]; then
        SITE=perlmutter
    elif [[ "$(hostname -s)" == alpha* ]]; then
        SITE=empireai_alpha
    else
        echo "site_env.sh: cannot detect the site from NERSC_HOST/hostname; export SITE=perlmutter|empireai_alpha" >&2
        return 1 2>/dev/null || exit 1
    fi
fi
export SITE

case "$SITE" in
perlmutter)
    # NERSC exports SCRATCH (/pscratch/sd/m/$USER). Python: the xr_env conda env
    # the CASS scripts have always used (module load conda; conda activate xr_env).
    : "${SCRATCH:?NERSC should export SCRATCH; it is unset}"
    export XR_PY="${XR_PY:-$HOME/.conda/envs/xr_env/bin/python}"
    export SITE_SBATCH_COMMON="${SITE_SBATCH_COMMON:---account=m1266}"
    # Raytracer legs need the 80 GB HBM nodes; 2stream fits any GPU node.
    export SITE_SBATCH_2STREAM="${SITE_SBATCH_2STREAM:---constraint=gpu}"
    export SITE_SBATCH_RAYTRACER="${SITE_SBATCH_RAYTRACER:---constraint=gpu&hbm80g}"
    # QoS names in use by the existing Perlmutter scripts: regular (whole node),
    # shared (single-GPU, half-node charging, MaxWall 48 h per the arm97sd July
    # scripts), debug (30 min). Single-GPU jobs go through `shared`.
    # VERIFY on Perlmutter with `sacctmgr show qos format=Name,MaxWall` before
    # relying on the 48 h figure or on hbm80g reachability through `shared`.
    export SITE_QOS_PROD="${SITE_QOS_PROD:-shared}"
    export SITE_QOS_DEBUG="${SITE_QOS_DEBUG:-debug}"
    export SITE_MAXWALL_PROD="${SITE_MAXWALL_PROD:-48:00:00}"
    export SITE_DEBUG_WALLTIME="${SITE_DEBUG_WALLTIME:-00:30:00}"
    export SITE_CPUS_PER_GPU="${SITE_CPUS_PER_GPU:-32}"    # 128 cores / 4 GPUs
    ;;
empireai_alpha)
    # Toolchain, SCRATCH and XR_PY come from the Alpha env script (mamba prefix).
    source "$MICROHH_DIR/config/empireai_alpha_env.sh" || return 1 2>/dev/null || exit 1
    export SITE_SBATCH_COMMON="${SITE_SBATCH_COMMON:---account=cu_rpincus_illuminating --partition=alpha}"
    # Both H100 and H200 nodes fit the raytracer at 512^2 x 384 (64 GB peak);
    # set GPU_TYPE=nvidia_h200 to force the six H200 nodes.
    export SITE_SBATCH_2STREAM="${SITE_SBATCH_2STREAM:-}"
    export SITE_SBATCH_RAYTRACER="${SITE_SBATCH_RAYTRACER:-}"
    # Alpha QoS tiers (sacctmgr, 2026-09-21): test 2 h, standard 48 h,
    # long 7 d, priority 24 h.
    export SITE_QOS_PROD="${SITE_QOS_PROD:-standard}"
    export SITE_QOS_DEBUG="${SITE_QOS_DEBUG:-test}"
    export SITE_MAXWALL_PROD="${SITE_MAXWALL_PROD:-48:00:00}"
    export SITE_DEBUG_WALLTIME="${SITE_DEBUG_WALLTIME:-02:00:00}"
    export SITE_CPUS_PER_GPU="${SITE_CPUS_PER_GPU:-12}"    # 96 cores / 8 GPUs
    ;;
*)
    echo "site_env.sh: unknown SITE=$SITE" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

export MICROHH_EXEC="${MICROHH_EXEC:-$MICROHH_DIR/build_gpu/microhh}"
unset _site_root
