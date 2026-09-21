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
# One GPU per sim, SITE_GPUS_PER_JOB sims per job, grouped within an RT mode so
# a job's sims share a wall time:
#   Perlmutter: 1 sim per job under the per-GPU-charged `shared` QoS.
#   Alpha: 2 sims per job with a typed gres, because the GPU governance plugin
#   sends every 1-GPU job to the RTX 6000 (Blackwell, weak FP64) pool.
# Raytracer jobs get a chained restart job (afterany) because a CASS raytracer
# sim is up to about 28 wall hours and the production QoS caps at 48 h.
#
# Overrides (export before running): SITE, the SITE_* variables, QOS,
# WALLTIME (raytracer), TS_WALLTIME (2stream), DEBUG_WALLTIME, MICROHH_EXEC.
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
if [[ $DEBUG == 1 ]]; then QOS=${QOS:-$SITE_QOS_DEBUG}; else QOS=${QOS:-$SITE_QOS_PROD}; fi
NGPU=$SITE_GPUS_PER_JOB
GRES="gpu:${SITE_GPU_TYPE:+$SITE_GPU_TYPE:}$NGPU"

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

read -ra COMMON <<< "$SITE_SBATCH_COMMON"
read -ra EXTRA_2S <<< "$SITE_SBATCH_2STREAM"
read -ra EXTRA_RT <<< "$SITE_SBATCH_RAYTRACER"
BASE=("${COMMON[@]}" --qos="$QOS" --gres="$GRES" --nodes=1
      --ntasks-per-node="$NGPU" --cpus-per-task="$SITE_CPUS_PER_GPU")

# Run dirs: <root>/[<value>/]<rt>/rep_NN. The sweep experiments (rs_scale,
# wind_geo) have the extra <value> level; the others do not. Enumerating the
# directories the setup script created keeps this file free of per-experiment
# layout knowledge.
find_runs() {  # $1 = rt
    find "$RUN_ROOT" -mindepth 2 -maxdepth 3 -type d -name 'rep_[0-9][0-9]' -path "*/$1/*" | sort
}

# Short tag for job names and log files: <expt>[_<value>]-<rt>-rep<NN>[+<M>]
tag_of() {  # $1 = sim dir
    local val_dir; val_dir="$(dirname "$(dirname "$1")")"
    if [[ "$val_dir" == "$RUN_ROOT" ]]; then echo "$EXPT"; else echo "${EXPT}_$(basename "$val_dir")"; fi
}

NJOBS=0
for RT in 2stream raytracer; do
    mapfile -t DIRS < <(find_runs "$RT")
    [[ ${#DIRS[@]} -gt 0 ]] || continue
    if [[ "$RT" == "raytracer" ]]; then
        EXTRA=("${EXTRA_RT[@]}"); RTS=rt
        if [[ $DEBUG == 1 ]]; then W="$DEBUG_WALLTIME"; else W="$WALLTIME"; fi
    else
        EXTRA=("${EXTRA_2S[@]}"); RTS=2s
        if [[ $DEBUG == 1 ]]; then W="$DEBUG_WALLTIME"; else W="$TS_WALLTIME"; fi
    fi

    # Pack NGPU consecutive sims (same RT, sorted so a value's reps stay
    # together) into one job. A leftover group of fewer sims still requests
    # NGPU GPUs, which is what the Alpha governance rule needs.
    for ((i = 0; i < ${#DIRS[@]}; i += NGPU)); do
        GROUP=("${DIRS[@]:i:NGPU}")
        SIM_DIRS=$(IFS=':'; echo "${GROUP[*]}")
        FIRST="${GROUP[0]}"
        TAG="$(tag_of "$FIRST")"
        REP="${FIRST##*/rep_}"
        SUFFIX=""; [[ ${#GROUP[@]} -gt 1 ]] && SUFFIX="+$(( ${#GROUP[@]} - 1 ))"
        NAME="${TAG}-${RT}-rep${REP}${SUFFIX}"

        jid=$(sbatch --parsable "${BASE[@]}" "${EXTRA[@]}" --time="$W" \
            --job-name="cass_${TAG}_${RTS}_${REP}${SUFFIX}" \
            --output="$LOGS/${NAME}-%j.out" \
            --error="$LOGS/${NAME}-%j.err" \
            --export=ALL,MICROHH_DIR="$MICROHH_DIR",SITE="$SITE",SIM_DIRS="$SIM_DIRS" \
            "$SCRIPT_DIR/sbatch_run.sh")
        echo "  $NAME (${#GROUP[@]} sim(s), $GRES): job $jid, walltime $W"
        NJOBS=$((NJOBS + 1))

        # Chain a restart for production raytracer jobs. afterany: the chain
        # must also run when the parent hits the wall, which is the case it
        # exists for. The restart body skips sims that already completed.
        if [[ $DEBUG == 0 && "$RT" == "raytracer" ]]; then
            rid=$(sbatch --parsable "${BASE[@]}" "${EXTRA[@]}" --time="$WALLTIME" \
                --job-name="cass_${TAG}_rst_${REP}${SUFFIX}" \
                --dependency=afterany:"$jid" \
                --output="$LOGS/${TAG}-restart-rep${REP}${SUFFIX}-%j.out" \
                --error="$LOGS/${TAG}-restart-rep${REP}${SUFFIX}-%j.err" \
                --export=ALL,MICROHH_DIR="$MICROHH_DIR",SITE="$SITE",SIM_DIRS="$SIM_DIRS" \
                "$SCRIPT_DIR/sbatch_restart.sh")
            echo "    chained restart: job $rid (afterany:$jid)"
        fi
    done
done
[[ $NJOBS -gt 0 ]] || { echo "no run dirs under $RUN_ROOT" >&2; exit 1; }

echo
echo "Submitted $NJOBS job(s) on $SITE (qos=$QOS, $NGPU sim(s) per job)."
echo "Watch with:  squeue -u $USER"
echo "Logs in:     $LOGS"
