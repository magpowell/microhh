#!/bin/bash
# Set up and submit a CASS experiment on Empire AI Alpha.
#
# Usage:
#   shared/submit_alpha.sh <experiment> [--debug] [-- <extra setup args>]
#
#   <experiment>  base | no_aerosols | no_aerosols_zero_wind | rs_scale |
#                 sw_scale | wind_geo | wind_sun  (any dir with setup_<name>.py)
#   --debug       64x64 grid, rep_01 only, QoS test, no restart chain. An
#                 end-to-end smoke test of build, input pipeline and Slurm
#                 wiring before spending real time.
#   -- ...        passed through to setup_<experiment>.py (e.g. -- --values 2 4)
#
# One single-GPU job per (value, RT mode, rep), not node-packed groups:
#   1. Alpha has 8 GPUs/node and no shared-node QoS, so a whole-node request
#      idles GPUs as soon as the fastest run finishes.
#   2. Single-GPU jobs backfill; measured 1-GPU waits in July were ~30 min.
# Raytracer reps get a chained restart job (afterany) because a CASS
# raytracer sim is 14-28 wall hours on H200 and standard QoS is capped at 48 h.
#
# Site settings come from config/empireai_alpha_env.sh (SCRATCH, XR_PY) and
# can be overridden in the environment:
#   ACCOUNT, PARTITION, QOS, WALLTIME (raytracer), TS_WALLTIME (2stream),
#   DEBUG_WALLTIME, GPU_TYPE (e.g. nvidia_h200), MICROHH_EXEC (binary to link).
#
# Alpha QoS tiers (2026-09-21): test 2 h, standard 48 h, long 7 d, priority 24 h.
set -euo pipefail

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 1; }
[[ $# -ge 1 ]] || usage
EXPT="$1"; shift
DEBUG=0
SETUP_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) DEBUG=1; shift ;;
        --) shift; SETUP_ARGS=("$@"); break ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # cases/cass/shared
CASS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MICROHH_DIR="$(cd "$CASS_DIR/../.." && pwd)"

# Env script provides SCRATCH and XR_PY and fails loudly if the toolchain is
# missing. Sourcing here means the setup step below sees the same values the
# job will.
source "$MICROHH_DIR/config/empireai_alpha_env.sh"

ACCOUNT=${ACCOUNT:-cu_rpincus_illuminating}
PARTITION=${PARTITION:-alpha}
WALLTIME=${WALLTIME:-48:00:00}
TS_WALLTIME=${TS_WALLTIME:-16:00:00}
DEBUG_WALLTIME=${DEBUG_WALLTIME:-02:00:00}
GPU_TYPE=${GPU_TYPE:-}
if [[ $DEBUG == 1 ]]; then QOS=${QOS:-test}; else QOS=${QOS:-standard}; fi

# Setup script and run root for this experiment.
if [[ "$EXPT" == "base" ]]; then
    SETUP="$CASS_DIR/base/setup_base.py"
    RUN_ROOT="$SCRATCH/CASS_LES/base"
else
    SETUP="$CASS_DIR/experiments/$EXPT/setup_$EXPT.py"
    RUN_ROOT="$SCRATCH/CASS_LES/experiments/$EXPT"
    # no_aerosols_zero_wind writes to a _v2 root (see its setup script).
    [[ "$EXPT" == "no_aerosols_zero_wind" ]] && RUN_ROOT="${RUN_ROOT}_v2"
fi
[[ -f "$SETUP" ]] || { echo "no setup script: $SETUP" >&2; exit 1; }
if [[ $DEBUG == 1 ]]; then
    RUN_ROOT="$SCRATCH/CASS_LES/debug/$EXPT"
    SETUP_ARGS=(--debug "${SETUP_ARGS[@]}")
fi

LOGS="$SCRATCH/CASS_LES/logs"
mkdir -p "$LOGS"

echo "Setting up CASS $EXPT (SCRATCH=$SCRATCH, debug=$DEBUG) ..."
MICROHH_DIR="$MICROHH_DIR" "$XR_PY" "$SETUP" "${SETUP_ARGS[@]}"

if [[ -n "$GPU_TYPE" ]]; then GRES="gpu:${GPU_TYPE}:1"; else GRES="gpu:1"; fi
# 12 CPUs/GPU (nodes are 96 cores / 8 GPUs).
BASE=(--account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS"
      --gres="$GRES" --ntasks-per-node=1 --cpus-per-task=12)

# Run dirs: <root>/[<value>/]<rt>/rep_NN. The sweep experiments (rs_scale,
# wind_geo) have the extra <value> level; the others do not. Enumerating the
# directories the setup script created keeps this file free of per-experiment
# layout knowledge.
mapfile -t RUN_DIRS < <(find "$RUN_ROOT" -mindepth 2 -maxdepth 3 -type d -name 'rep_[0-9][0-9]' \
                        \( -path '*/2stream/*' -o -path '*/raytracer/*' \) | sort)
[[ ${#RUN_DIRS[@]} -gt 0 ]] || { echo "no run dirs under $RUN_ROOT" >&2; exit 1; }

echo "Submitting ${#RUN_DIRS[@]} single-GPU job(s) (qos=$QOS) ..."
for SIM_DIR in "${RUN_DIRS[@]}"; do
    RT="$(basename "$(dirname "$SIM_DIR")")"                # 2stream | raytracer
    REP="${SIM_DIR##*/rep_}"
    VAL_DIR="$(dirname "$(dirname "$SIM_DIR")")"
    if [[ "$VAL_DIR" == "$RUN_ROOT" ]]; then TAG="$EXPT"; else TAG="${EXPT}_$(basename "$VAL_DIR")"; fi
    if [[ $DEBUG == 1 ]]; then W="$DEBUG_WALLTIME"
    elif [[ "$RT" == "2stream" ]]; then W="$TS_WALLTIME"
    else W="$WALLTIME"; fi
    RTS="${RT:0:2}"                                         # 2s | ra
    [[ "$RT" == "raytracer" ]] && RTS="rt"

    jid=$(sbatch --parsable "${BASE[@]}" --time="$W" \
        --job-name="cass_${TAG}_${RTS}_${REP}" \
        --output="$LOGS/${TAG}-${RT}-rep${REP}-%j.out" \
        --error="$LOGS/${TAG}-${RT}-rep${REP}-%j.err" \
        --export=ALL,MICROHH_DIR="$MICROHH_DIR",SIM_DIRS="$SIM_DIR" \
        "$SCRIPT_DIR/sbatch_run_alpha.sh")
    echo "  $TAG $RT rep_$REP: job $jid (walltime $W)"

    # Chain a restart for production raytracer reps. afterany: the chain must
    # also run when the parent hits the wall, which is the case it exists for.
    if [[ $DEBUG == 0 && "$RT" == "raytracer" ]]; then
        rid=$(sbatch --parsable "${BASE[@]}" --time="$WALLTIME" \
            --job-name="cass_${TAG}_rst_${REP}" \
            --dependency=afterany:"$jid" \
            --output="$LOGS/${TAG}-restart-rep${REP}-%j.out" \
            --error="$LOGS/${TAG}-restart-rep${REP}-%j.err" \
            --export=ALL,MICROHH_DIR="$MICROHH_DIR",SIM_DIRS="$SIM_DIR" \
            "$SCRIPT_DIR/sbatch_restart_alpha.sh")
        echo "    chained restart: job $rid (afterany:$jid)"
    fi
done

echo
echo "Watch with:  squeue -u $USER"
echo "Logs in:     $LOGS"
