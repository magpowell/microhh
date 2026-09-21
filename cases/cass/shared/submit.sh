#!/bin/bash
# Set up and submit a CASS experiment on the current site (Perlmutter or
# Empire AI Alpha). Site differences live in config/site_env.sh.
#
# Usage:
#   shared/submit.sh <experiment> [--debug] [-- <extra setup args>]
#
#   <experiment>  base | no_aerosols | no_aerosols_zero_wind | rs_scale |
#                 sw_scale | wind_geo | wind_sun  (any dir with setup_<name>.py)
#   --debug       64x64 grid, rep_01 only, debug QoS, no restart chain. An
#                 end-to-end smoke test of build, input pipeline and Slurm
#                 wiring before spending real time.
#   -- ...        passed through to setup_<experiment>.py (e.g. -- --values 2 4)
#
# One single-GPU job per (value, RT mode, rep), on both sites:
#   Alpha has 8 GPUs/node and no shared-node QoS, so a whole-node request idles
#   GPUs as soon as the fastest run finishes; Perlmutter's `shared` QoS charges
#   per GPU. Single-GPU jobs backfill well on both.
# Raytracer reps get a chained restart job (afterany) because a CASS raytracer
# sim is up to about 28 wall hours and the production QoS caps at 48 h.
#
# Overrides (export before running): SITE, ACCOUNT/PARTITION via
# SITE_SBATCH_COMMON, QOS, WALLTIME (raytracer), TS_WALLTIME (2stream),
# DEBUG_WALLTIME, GPU_TYPE (Alpha: nvidia_h200), MICROHH_EXEC.
set -euo pipefail

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 1; }
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
source "$CASS_DIR/../../config/site_env.sh"

WALLTIME=${WALLTIME:-$SITE_MAXWALL_PROD}
TS_WALLTIME=${TS_WALLTIME:-16:00:00}
DEBUG_WALLTIME=${DEBUG_WALLTIME:-$SITE_DEBUG_WALLTIME}
GPU_TYPE=${GPU_TYPE:-}
if [[ $DEBUG == 1 ]]; then QOS=${QOS:-$SITE_QOS_DEBUG}; else QOS=${QOS:-$SITE_QOS_PROD}; fi

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

echo "Setting up CASS $EXPT on $SITE (SCRATCH=$SCRATCH, debug=$DEBUG) ..."
"$XR_PY" "$SETUP" "${SETUP_ARGS[@]}"

if [[ -n "$GPU_TYPE" ]]; then GRES="gpu:${GPU_TYPE}:1"; else GRES="gpu:1"; fi
read -ra COMMON <<< "$SITE_SBATCH_COMMON"
read -ra EXTRA_2S <<< "$SITE_SBATCH_2STREAM"
read -ra EXTRA_RT <<< "$SITE_SBATCH_RAYTRACER"
BASE=("${COMMON[@]}" --qos="$QOS" --gres="$GRES" --nodes=1
      --ntasks-per-node=1 --cpus-per-task="$SITE_CPUS_PER_GPU")

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
    if [[ "$RT" == "raytracer" ]]; then
        EXTRA=("${EXTRA_RT[@]}"); RTS=rt
        if [[ $DEBUG == 1 ]]; then W="$DEBUG_WALLTIME"; else W="$WALLTIME"; fi
    else
        EXTRA=("${EXTRA_2S[@]}"); RTS=2s
        if [[ $DEBUG == 1 ]]; then W="$DEBUG_WALLTIME"; else W="$TS_WALLTIME"; fi
    fi

    jid=$(sbatch --parsable "${BASE[@]}" "${EXTRA[@]}" --time="$W" \
        --job-name="cass_${TAG}_${RTS}_${REP}" \
        --output="$LOGS/${TAG}-${RT}-rep${REP}-%j.out" \
        --error="$LOGS/${TAG}-${RT}-rep${REP}-%j.err" \
        --export=ALL,MICROHH_DIR="$MICROHH_DIR",SITE="$SITE",SIM_DIRS="$SIM_DIR" \
        "$SCRIPT_DIR/sbatch_run.sh")
    echo "  $TAG $RT rep_$REP: job $jid (walltime $W)"

    # Chain a restart for production raytracer reps. afterany: the chain must
    # also run when the parent hits the wall, which is the case it exists for.
    if [[ $DEBUG == 0 && "$RT" == "raytracer" ]]; then
        rid=$(sbatch --parsable "${BASE[@]}" "${EXTRA[@]}" --time="$WALLTIME" \
            --job-name="cass_${TAG}_rst_${REP}" \
            --dependency=afterany:"$jid" \
            --output="$LOGS/${TAG}-restart-rep${REP}-%j.out" \
            --error="$LOGS/${TAG}-restart-rep${REP}-%j.err" \
            --export=ALL,MICROHH_DIR="$MICROHH_DIR",SITE="$SITE",SIM_DIRS="$SIM_DIR" \
            "$SCRIPT_DIR/sbatch_restart.sh")
        echo "    chained restart: job $rid (afterany:$jid)"
    fi
done

echo
echo "Watch with:  squeue -u $USER"
echo "Logs in:     $LOGS"
