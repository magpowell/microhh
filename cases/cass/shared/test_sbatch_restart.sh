#!/bin/bash
# Test of sbatch_restart.sh with a stub srun and a stub model. Runs on a login node in a few seconds.
#   bash test_sbatch_restart.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MICROHH_DIR="$(cd "$HERE/../../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
fail=0

mkdir -p "$T/bin"
printf '#!/bin/bash\nshift 2\nSLURM_LOCALID=0 exec "$@"\n' > "$T/bin/srun"
chmod +x "$T/bin/srun"

make_run() {  # $1 = dir, $2 = last time in cass.out, $3... = times with restart files
    local d="$1" out_t="$2"; shift 2
    mkdir -p "$d"
    printf '[time]\nendtime = 50000.0\nsavetime = 3600\nstarttime = 0.\n' > "$d/cass.ini"
    printf '    ITER          TIME     CPUDT          DT\n       0             0    0.1000   6.000E+00\n   12091   %s    0.8767   2.299E+01\n' "$out_t" > "$d/cass.out"
    for t in "$@"; do
        for v in thl qt ql w u v couvreux qr nr; do : > "$d/$v.$(printf '%07d' "$t")"; done
    done
    printf '#!/bin/bash\necho "$@" > model_was_called\n' > "$d/microhh"
    printf 'pass\n' > "$d/cross_to_nc.py"
    chmod +x "$d/microhh"
}

run() { PATH="$T/bin:$PATH" SIM_DIRS="$1" bash "$HERE/sbatch_restart.sh" > "$1/restart.log" 2>&1; }

check() {  # $1 = name, $2 = condition (0 is true)
    if [[ "$2" == 0 ]]; then echo "ok    $1"; else echo "FAIL  $1"; fail=1; fi
}

# 1. Complete run whose end is not a multiple of savetime: nothing may happen.
make_run "$T/complete" 50000 0 43200 46800
run "$T/complete"
[[ ! -f "$T/complete/model_was_called" ]]; check "complete run is not restarted" $?
grep -q "^starttime = 0\.$" "$T/complete/cass.ini"; check "complete run keeps its ini" $?

# 2. Run stopped by the wall time at 40100 s: restart from 39600 s.
make_run "$T/stopped" 40100.5 0 36000 39600
run "$T/stopped"
[[ -f "$T/stopped/model_was_called" ]]; check "stopped run is restarted" $?
grep -q "^starttime = 39600\.$" "$T/stopped/cass.ini"; check "stopped run starts from the last restart files" $?
grep -q "^starttime = 0\.$" "$T/stopped/cass.ini.before_restart"; check "original ini is kept" $?

# 3. Restart files incomplete at the last time: abort without touching the ini.
make_run "$T/missing" 40100.5 0 36000 39600
rm "$T/missing/qr.0039600"
run "$T/missing"
[[ ! -f "$T/missing/model_was_called" ]]; check "run with a missing restart file is not restarted" $?
grep -q "^starttime = 0\.$" "$T/missing/cass.ini"; check "run with a missing restart file keeps its ini" $?

exit $fail
