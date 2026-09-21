#!/bin/bash
# Environment for building/running MicroHH on Empire AI Alpha (x86_64, H100/H200).
# Usage:  source config/empireai_alpha_env.sh
#
# Pair with config/empireai_alpha.cmake:
#   cmake -DUSECUDA=TRUE -DUSEMPI=FALSE -DSYST=empireai_alpha ..
#
# NOT for Beta/NVL72 -- separate cluster, separate Slurm, ARM64 hosts.
#
# ---------------------------------------------------------------------------
# Why everything comes from a mamba prefix in $HOME, and not from modules
# ---------------------------------------------------------------------------
# The login node and the compute nodes mount DIFFERENT /cm/shared/apps trees.
# Verified on alphagpu20 -- its /cm/shared/apps contains only:
#   apptainer cm-pmix3 cm-pmix4 hdf5_18 hpl hwloc hwloc2 mvapich2 openblas
#   openmpi4 slurm ucx uge
# No cuda12.4, no netcdf, no fftw; /cm/local/apps/gcc/13.1.0 is absent too, and
# `module load cuda12.4/toolkit/12.4.1` fails there ("module(s) are unknown").
#
# A binary linked against the login node's /cm/shared therefore cannot run in a
# batch job. /mnt/home is the only filesystem both node types share, so the
# toolchain lives in the mamba env there and this script simply points at it.
#
# Related trap, kept as a warning: do NOT `module purge` on this cluster. The
# `shared` module is what puts /cm/shared/modulefiles on MODULEPATH, so purging
# makes every later `module load` fail as "unknown" -- which under `set -u`
# kills a batch job on the first unset variable.
# ---------------------------------------------------------------------------

export MICROHH_CONDA_PREFIX="${MICROHH_CONDA_PREFIX:-$HOME/miniforge3/envs/microhh}"

export PATH="${MICROHH_CONDA_PREFIX}/bin:${PATH:-}"
export LD_LIBRARY_PATH="${MICROHH_CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# For empireai_alpha.cmake / anything expecting a CUDA root.
export CUDA_HOME="$MICROHH_CONDA_PREFIX"
export CUDA_PATH="$MICROHH_CONDA_PREFIX"

# Interpreter used by the cases' setup_runs.py (xarray/netCDF4/scipy).
export XR_PY="${XR_PY:-$MICROHH_CONDA_PREFIX/bin/python}"

# Run output location.
#
# CAVEAT: /mnt/lustre is documented as a DDN Lustre scratch appliance, but on
# both the login node and alphagpu20 it is a symlink to /mnt/home/DDN_Copy and
# resolves to plain home NFS (10.0.4.155:/home on the node). It is NOT Lustre
# as configured today. That matters for TB-scale LES output -- raised with
# Empire AI support; revisit if a real Lustre mount appears.
export SCRATCH="${SCRATCH:-/mnt/lustre/columbia/${USER}}"

# Fail loudly if the toolchain is missing, rather than dying later on a
# confusing missing-.so error inside a batch job.
for _f in bin/nvcc lib/libnetcdf.so lib/libhdf5.so lib/libfftw3.so lib/libcufft.so; do
    if [ ! -e "${MICROHH_CONDA_PREFIX}/${_f}" ]; then
        echo "empireai_alpha_env.sh: missing ${MICROHH_CONDA_PREFIX}/${_f}" >&2
        echo "  (is the mamba env built, and is /mnt/home visible on $(hostname)?)" >&2
        unset _f
        return 1 2>/dev/null || exit 1
    fi
done
unset _f
