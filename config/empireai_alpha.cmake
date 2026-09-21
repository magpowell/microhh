# CMake file for Empire AI "Alpha" (alpha.empire-ai.org), x86_64.
#
# Alpha GPU nodes: alphagpu01-18 = 8x H100 80GB, alphagpu19-24 = 8x H200 141GB.
# Both are compute capability 9.0 (sm_90), hence CMAKE_CUDA_ARCHITECTURES 90.
#
# NOT for Beta/NVL72 -- Beta is a separate cluster, separate Slurm, ARM64 hosts.
#
# ---------------------------------------------------------------------------
# Why the whole toolchain comes from a conda (mamba) prefix
# ---------------------------------------------------------------------------
# The login node and the compute nodes mount DIFFERENT /cm/shared/apps trees.
# Verified on alphagpu20: of everything this build needs, only hdf5_18 exists
# there. cuda12.4, netcdf, fftw and /cm/local/apps/gcc are login-node-only, and
# `module load cuda12.4/...` fails on the node because that modulefile is not
# in its MODULEPATH either.
#
# So a binary linked against /cm/shared cannot run in a batch job -- it dies
# with "libnetcdf.so.19: cannot open shared object file". /mnt/home is the one
# filesystem both node types share, so the entire toolchain (nvcc, cuFFT,
# cuRAND, netCDF, HDF5, FFTW, gcc) is taken from the mamba env there. Build and
# run environments are then identical by construction.
#
# CUDA is pinned to 12.6 in that env rather than matching the nodes' system
# CUDA 13.0.3: driver 580.126.09 is CUDA 13-era and runs 12.x fine, while the
# nodes' CUDA 13 ships libcufft.so.12 and the 12.x build wants libcufft.so.11.
#
# Requires: source config/empireai_alpha_env.sh   (sets CONDA_PREFIX etc.)
# ---------------------------------------------------------------------------

if(DEFINED ENV{MICROHH_CONDA_PREFIX})
    set(CONDA_ROOT "$ENV{MICROHH_CONDA_PREFIX}")
else()
    set(CONDA_ROOT "$ENV{HOME}/miniforge3/envs/microhh")
endif()

if(NOT EXISTS "${CONDA_ROOT}/bin/nvcc")
    message(FATAL_ERROR
        "No nvcc under ${CONDA_ROOT}. Source config/empireai_alpha_env.sh first, "
        "or set MICROHH_CONDA_PREFIX to the mamba env holding the toolchain.")
endif()

if(USEMPI)
    set(ENV{CC}  mpicc)
    set(ENV{CXX} mpicxx)
    set(ENV{FC}  mpif90)
else()
    set(ENV{CC}  "${CONDA_ROOT}/bin/x86_64-conda-linux-gnu-gcc")
    set(ENV{CXX} "${CONDA_ROOT}/bin/x86_64-conda-linux-gnu-g++")
    set(ENV{FC}  "${CONDA_ROOT}/bin/x86_64-conda-linux-gnu-gfortran")
endif()

set(CMAKE_C_COMPILER       $ENV{CC})
set(CMAKE_CXX_COMPILER     $ENV{CXX})
set(CMAKE_Fortran_COMPILER $ENV{FC})

# Compiler flags.
# NOTE: deliberately NOT -march=native. The login node where this compiles is a
# different machine class from alphagpu*, so tuning to the build host risks
# illegal instructions at run time. x86-64-v3 covers every Alpha node.
set(USER_CXX_FLAGS "-std=c++17")
set(USER_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=x86-64-v3")
set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

set(USER_FC_FLAGS "-fdefault-real-8 -fdefault-double-8 -fPIC -ffixed-line-length-none -fno-range-check")
set(USER_FC_FLAGS_RELEASE "-DNDEBUG -O3 -march=x86-64-v3")
set(USER_FC_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

add_definitions(-DRESTRICTKEYWORD=__restrict__)

# Library settings -- all from the conda prefix.
set(NETCDF_INCLUDE_DIR "${CONDA_ROOT}/include")
set(NETCDF_LIB_C       "${CONDA_ROOT}/lib/libnetcdf.so")

set(HDF5_LIB_1 "${CONDA_ROOT}/lib/libhdf5.so")
set(HDF5_LIB_2 "${CONDA_ROOT}/lib/libhdf5_hl.so")

set(FFTW_INCLUDE_DIR "${CONDA_ROOT}/include")
set(FFTW_LIB  "${CONDA_ROOT}/lib/libfftw3.so")
set(FFTWF_LIB "${CONDA_ROOT}/lib/libfftw3f.so")

# conda-forge's CUDA packages put their headers under targets/<arch>/include,
# NOT directly in $PREFIX/include -- cuda_runtime_api.h lives only there.
set(CUDA_TARGET_INCLUDE "${CONDA_ROOT}/targets/x86_64-linux/include")

set(CURAND_LIB_DIR     "${CONDA_ROOT}/lib")
set(CURAND_INCLUDE_DIR "${CUDA_TARGET_INCLUDE}")
set(CURAND_LIB_1       "${CURAND_LIB_DIR}/libcurand.so")

set(CUFFT_LIB_DIR     "${CONDA_ROOT}/lib")
set(CUFFT_INCLUDE_DIR "${CUDA_TARGET_INCLUDE}")
set(CUFFT_LIB_1       "${CUFFT_LIB_DIR}/libcufft.so")
set(CUDA_INCLUDE_DIR  "${CUDA_TARGET_INCLUDE}")

set(SZIP_LIB "")
# 'curl' omitted (default.cmake has it): there is no libcurl.so to link against
# and the shared libnetcdf.so resolves curl itself.
set(LIBS ${NETCDF_LIB_C} ${HDF5_LIB_2} ${HDF5_LIB_1} ${CURAND_LIB_1} ${CUFFT_LIB_1} ${FFTW_LIB} ${FFTWF_LIB} ${SZIP_LIB} m z)
# ${CONDA_ROOT}/include carries the boost headers that rte-rrtmgp-cpp needs
# (conda's gcc uses its own sysroot and does not see /usr/include/boost).
set(INCLUDE_DIRS ${NETCDF_INCLUDE_DIR} ${CURAND_INCLUDE_DIR} ${CUFFT_INCLUDE_DIR} ${FFTW_INCLUDE_DIR} ${CUDA_INCLUDE_DIR} "${CONDA_ROOT}/include")
link_directories(${CURAND_LIB_DIR} ${CUFFT_LIB_DIR})

# CUDA support
if(USECUDA)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CMAKE_CUDA_COMPILER "${CONDA_ROOT}/bin/nvcc")
    set(CMAKE_CUDA_HOST_COMPILER "${CONDA_ROOT}/bin/x86_64-conda-linux-gnu-g++")
    set(CMAKE_CUDA_ARCHITECTURES 90)
    set(USER_CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
    set(USER_CUDA_NVCC_FLAGS_RELEASE "-DNDEBUG")
    set(USER_CUDA_NVCC_FLAGS_DEBUG "-O0 -g -DCUDACHECKS")
    add_definitions(-DRTE_RRTMGP_GPU_MEMPOOL_CUDA)
endif()

# Scratch here is NFS-backed (see env script note), not GPFS/Lustre; keep
# MPI-IO for cross-sections disabled as it was on Perlmutter.
add_definitions(-DDISABLE_2D_MPIIO=1)
add_definitions(-DRTE_USE_CBOOL)
