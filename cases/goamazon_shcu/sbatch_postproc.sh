#!/bin/bash
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/pscratch/sd/m/mpowell/GOAMAZON_SHCU_LES/logs/postproc-%j.out
#SBATCH --error=/pscratch/sd/m/mpowell/GOAMAZON_SHCU_LES/logs/postproc-%j.err
#SBATCH --mail-user=mp4257@columbia.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -A m1266
#
# Convert MicroHH binary output (xy crosses + hourly 3D dumps) to NetCDF.
# SIM_DIRS: colon-separated run directories, via --export.
# For RESTARTED runs pass T0=0 (adds `-t0 0` so the full timeline converts).

PY=/global/homes/m/mpowell/.conda/envs/xr_env/bin/python
T0_ARG=""
[[ -n "${T0:-}" ]] && T0_ARG="-t0 ${T0}"

IFS=':' read -ra DIRS <<< "$SIM_DIRS"
for dir in "${DIRS[@]}"; do
    [[ -z "$dir" ]] && continue
    echo "=== $dir ($(date)) ==="
    cd "$dir"
    $PY cross_to_nc.py -f goamazon_shcu.ini -m xy -n 8 $T0_ARG
    $PY 3d_to_nc.py -f goamazon_shcu.ini -v T ql qi qt thl w b qr qs qg \
        -p single -n 8 $T0_ARG
    echo "=== done $dir ($(date)) ==="
done
