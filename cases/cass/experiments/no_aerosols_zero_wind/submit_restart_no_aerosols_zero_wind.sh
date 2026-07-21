#!/bin/bash
# Resubmit timed-out CASS no_aerosols_zero_wind runs from the latest savetime.
# Use after a TIMEOUT.  Uses sbatch_restart_no_aerosols_zero_wind.sh which
# auto-detects the restart point and patches cass.ini in-place.
#
# After the restart finishes:
#   * cass.ini will have starttime = <restart_t>; the original is at
#     cass.ini.before_restart.
#   * When running 3d_to_nc.py or cross_to_nc.py to convert binary dumps
#     across the FULL simulation (t=0 → endtime), pass `-t0 0`.
#
# Usage:
#   ./submit_restart_no_aerosols_zero_wind.sh           # default: 2stream only
#   ./submit_restart_no_aerosols_zero_wind.sh raytracer # restart raytracer

set -euo pipefail

WHICH="${1:-2stream}"

SCRATCH=${SCRATCH:-/pscratch/sd/m/mpowell}
EXP="$SCRATCH/CASS_LES/experiments/no_aerosols_zero_wind_v2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRATCH/CASS_LES/logs"

case "$WHICH" in
    2stream)
        # 2stream restart: ~6800 s of sim remaining → ~2 h wall at 1:1 ratio,
        # 3 h with buffer (passive tracer adds overhead vs pre-tracer runs).
        WALL=03:00:00
        CONSTRAINT=gpu
        JOBNAME=nazw_2s_restart
        SUBDIR=2stream
        ;;
    raytracer)
        # Raytracer restart: roughly 2-3x slower than 2stream.  Pick wall
        # generous enough to absorb the tracer overhead.
        WALL=06:00:00
        CONSTRAINT="gpu&hbm80g"
        JOBNAME=nazw_rt_restart
        SUBDIR=raytracer
        ;;
    *)
        echo "Usage: $0 [2stream|raytracer]" >&2
        exit 1
        ;;
esac

SIM_DIRS_LIST=()
for rep in 01 02 03 04; do
    SIM_DIRS_LIST+=("$EXP/$SUBDIR/rep_$rep")
done
SIM_DIRS=$(IFS=':'; echo "${SIM_DIRS_LIST[*]}")

sbatch \
    --constraint="$CONSTRAINT" \
    --time="$WALL" \
    --job-name="$JOBNAME" \
    --export=ALL,SIM_DIRS="$SIM_DIRS" \
    "$SCRIPT_DIR/sbatch_restart_no_aerosols_zero_wind.sh"
echo "Submitted no_aerosols_zero_wind restart ($WHICH, wall $WALL)"
