#!/bin/bash
# Run the per-run stages and collect.py for one experiment on a CPU node (8 runs in parallel).
# Usage: [STAGES="snapshot parcel activation"] bash submit_characterize.sh [expt] [qos] [walltime]
set -euo pipefail
EXPT=${1:-no_aerosols_zero_wind_v2}
QOS=${2:-debug}
WALL=${3:-00:30:00}
STAGES=${STAGES:-snapshot parcel activation}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$SCRATCH/CASS_LES/logs"
mkdir -p "$LOGS"
sbatch --qos="$QOS" --constraint=cpu --nodes=1 --ntasks=1 --cpus-per-task=128 --time="$WALL" -A m1266 \
    --job-name="char_${EXPT}" --output="$LOGS/characterize_${EXPT}-%j.out" --error="$LOGS/characterize_${EXPT}-%j.err" \
    --export=ALL,EXPT="$EXPT",HERE="$HERE",STAGES="$STAGES" --wrap='
module load conda; conda activate xr_env
cd "$HERE"
export OMP_NUM_THREADS=8
for rt in 2stream raytracer; do for rep in 1 2 3 4; do
    ( for s in $STAGES; do python $s.py --expt "$EXPT" --rt $rt --rep $rep || exit 1; done ) &
done; done
wait
python collect.py --expt "$EXPT"
echo "done at $(date)"'
