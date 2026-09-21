# CMake file for Ubuntu 20 and 22 LTS based on generic.cmake.

if(USEMPI)
    set(ENV{CC} mpicc)
    set(ENV{CXX} mpicxx)
    set(ENV{FC} mpif90)

    set(CMAKE_C_COMPILER $ENV{CC})
    set(CMAKE_CXX_COMPILER $ENV{CXX})
    set(CMAKE_Fortran_COMPILER $ENV{FC})
else()
    set(ENV{CC} gcc)
    set(ENV{CXX} g++)
    set(ENV{FC} gfortran)

    set(CMAKE_C_COMPILER $ENV{CC})
    set(CMAKE_CXX_COMPILER $ENV{CXX})
    set(CMAKE_Fortran_COMPILER $ENV{FC})
endif()

# Compiler flags
set(USER_CXX_FLAGS "-std=c++17")
set(USER_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

set(USER_FC_FLAGS "-fdefault-real-8 -fdefault-double-8 -fPIC -ffixed-line-length-none -fno-range-check")
set(USER_FC_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
set(USER_FC_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

add_definitions(-DRESTRICTKEYWORD=__restrict__)

# Library settings

# Library settings
if(DEFINED ENV{CRAY_NETCDF_PREFIX})
    set(NETCDF_INCLUDE_DIR "$ENV{CRAY_NETCDF_PREFIX}/include")
    set(NETCDF_LIB_C "$ENV{CRAY_NETCDF_PREFIX}/lib/libnetcdf.so")
else()
    set(NETCDF_INCLUDE_DIR "/opt/cray/pe/netcdf/4.9.0.13/include")
    set(NETCDF_LIB_C "/opt/cray/pe/netcdf/4.9.0.13/gnu/12.3/lib/libnetcdf.so")
endif()

if(DEFINED ENV{CRAY_HDF5_PREFIX})
    set(HDF5_LIB_1 "$ENV{CRAY_HDF5_PREFIX}/lib/libhdf5.so")
    set(HDF5_LIB_2 "$ENV{CRAY_HDF5_PREFIX}/lib/libhdf5_hl.so")
else()
    set(HDF5_LIB_1 "/opt/cray/pe/hdf5/1.14.3.1/gnu/12.3/lib/libhdf5.so")
    set(HDF5_LIB_2 "/opt/cray/pe/hdf5/1.14.3.1/gnu/12.3/lib/libhdf5_hl.so")
endif()

if(DEFINED ENV{FFTW_ROOT})
    set(FFTW_ROOT_DIR "$ENV{FFTW_ROOT}")
else()
    set(FFTW_ROOT_DIR "/opt/cray/pe/fftw/3.3.10.11/x86_milan")
endif()
set(FFTW_INCLUDE_DIR "${FFTW_ROOT_DIR}/include")
set(FFTW_LIB "${FFTW_ROOT_DIR}/lib/libfftw3.so")
set(FFTWF_LIB "${FFTW_ROOT_DIR}/lib/libfftw3f.so")

# CUDA/math libs follow the loaded cudatoolkit module (NVHPC_CUDA_HOME = <hpc_sdk>/<ver>/cuda/<cuda_ver>)
if(DEFINED ENV{NVHPC_CUDA_HOME})
    get_filename_component(NVHPC_ROOT "$ENV{NVHPC_CUDA_HOME}/../.." ABSOLUTE)
    get_filename_component(NVHPC_CUDA_VER "$ENV{NVHPC_CUDA_HOME}" NAME)
else()
    set(NVHPC_ROOT "/opt/nvidia/hpc_sdk/Linux_x86_64/26.5")
    set(NVHPC_CUDA_VER "13.2")
endif()
set(CURAND_LIB_DIR "${NVHPC_ROOT}/math_libs/${NVHPC_CUDA_VER}/lib64")
set(CURAND_INCLUDE_DIR "${NVHPC_ROOT}/math_libs/${NVHPC_CUDA_VER}/include")
set(CURAND_LIB_1 "${CURAND_LIB_DIR}/libcurand.so")

set(CUFFT_LIB_DIR "${CURAND_LIB_DIR}")
set(CUFFT_INCLUDE_DIR "${CURAND_INCLUDE_DIR}")
set(CUFFT_LIB_1 "${CUFFT_LIB_DIR}/libcufft.so")
set(CUDA_INCLUDE_DIR "${NVHPC_ROOT}/cuda/${NVHPC_CUDA_VER}/include")


set(SZIP_LIB "")
set(LIBS ${NETCDF_LIB_C} ${HDF5_LIB_2} ${HDF5_LIB_1} ${CURAND_LIB_1} ${CUFFT_LIB_1} ${FFTW_LIB} ${FFTWF_LIB} ${SZIP_LIB} m z curl)
set(INCLUDE_DIRS ${NETCDF_INCLUDE_DIR} ${CURAND_INCLUDE_DIR} ${CUFFT_INCLUDE_DIR} ${FFTW_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
link_directories(${CURAND_LIB_DIR} ${CUFFT_LIB_DIR})

# CUDA support
if(USECUDA)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-14")
    set(CMAKE_CUDA_ARCHITECTURES 80)
    set(USER_CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
    set(USER_CUDA_NVCC_FLAGS_RELEASE "-DNDEBUG")
    set(USER_CUDA_NVCC_FLAGS_DEBUG "-O0 -g -DCUDACHECKS")
    add_definitions(-DRTE_RRTMGP_GPU_MEMPOOL_CUDA)
endif()

# Disable MPI-IO for cross-sections on GPFS
add_definitions(-DDISABLE_2D_MPIIO=1)
add_definitions(-DRTE_USE_CBOOL)

