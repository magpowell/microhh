#!/usr/bin/env bash
# Prep slice files for the mechanism cross-section figure.
#
# Reads raw 3D fields + 2D surface dumps from a run directory and writes
# trimmed, single-time-step nc3 slice files into
#   $SCRATCH/CASS_LES/analysis/mechanism_slices/<expt>_<rep>_t<sec>/
#
# Render: the cross-section figure itself is generated inline by
# mechanism.ipynb after these slices exist.
#
# Usage:  bash build_xsection.sh <DUMP_TIME_S>  [<EXPT> <REP>]
# e.g.:   bash build_xsection.sh 36000
#         bash build_xsection.sh 43200 no_aerosols_zero_wind_v2 rep_02

set -euo pipefail

DUMP_TIME=${1:?"usage: bash build_xsection.sh <DUMP_TIME_S> [<EXPT> <REP>]"}
EXPT=${2:-no_aerosols_zero_wind_v2}
REP=${3:-rep_01}

PREP=$(dirname "$(readlink -f "$0")")
LES_ROOT=${SCRATCH:-/pscratch/sd/m/mpowell}/CASS_LES
RUN_2S=$LES_ROOT/experiments/$EXPT/2stream/$REP
RUN_RT=$LES_ROOT/experiments/$EXPT/raytracer/$REP
SLICE_DIR=$LES_ROOT/analysis/mechanism_slices/${EXPT}_${REP}_t${DUMP_TIME}
mkdir -p "$SLICE_DIR"

LST=$(awk -v t="$DUMP_TIME" 'BEGIN{printf "%.2f", t/3600+5.5}')
echo "[xsection prep]  $EXPT  $REP  t=$DUMP_TIME s  → LST $LST"
echo "[xsection prep]  slices → $SLICE_DIR"

prep() {
    local label="$1"  out="$2"  cmd="$3"
    if [[ -f "$out" ]]; then
        echo "[prep] reusing $label"
    else
        echo "[prep] $label"
        eval "$cmd"
    fi
}
prep "ql 2stream"        "$SLICE_DIR/ql_2stream.nc" \
     "python $PREP/prep_volume_slice.py    --in-nc $RUN_2S/ql.nc       --var ql       --dump-time $DUMP_TIME --out $SLICE_DIR/ql_2stream.nc"
prep "ql raytracer"      "$SLICE_DIR/ql_raytracer.nc" \
     "python $PREP/prep_volume_slice.py    --in-nc $RUN_RT/ql.nc       --var ql       --dump-time $DUMP_TIME --out $SLICE_DIR/ql_raytracer.nc"
prep "couvreux 2stream"  "$SLICE_DIR/couvreux_2stream.nc" \
     "python $PREP/prep_couvreux_anomaly.py --in-nc $RUN_2S/couvreux.nc                --dump-time $DUMP_TIME --out $SLICE_DIR/couvreux_2stream.nc"
prep "couvreux raytracer" "$SLICE_DIR/couvreux_raytracer.nc" \
     "python $PREP/prep_couvreux_anomaly.py --in-nc $RUN_RT/couvreux.nc                --dump-time $DUMP_TIME --out $SLICE_DIR/couvreux_raytracer.nc"
prep "sw 2stream"        "$SLICE_DIR/sw_2stream.nc" \
     "python $PREP/prep_sw_surface.py      --run-dir $RUN_2S --rt 2stream   --dump-time $DUMP_TIME --out $SLICE_DIR/sw_2stream.nc"
prep "sw raytracer"      "$SLICE_DIR/sw_raytracer.nc" \
     "python $PREP/prep_sw_surface.py      --run-dir $RUN_RT --rt raytracer --dump-time $DUMP_TIME --out $SLICE_DIR/sw_raytracer.nc"
prep "w 2stream"         "$SLICE_DIR/w_2stream.nc" \
     "python $PREP/prep_w_slice.py         --in-nc $RUN_2S/w.nc                --dump-time $DUMP_TIME --out $SLICE_DIR/w_2stream.nc"
prep "w raytracer"       "$SLICE_DIR/w_raytracer.nc" \
     "python $PREP/prep_w_slice.py         --in-nc $RUN_RT/w.nc                --dump-time $DUMP_TIME --out $SLICE_DIR/w_raytracer.nc"

echo "[xsection prep]  done; render via mechanism.ipynb"
