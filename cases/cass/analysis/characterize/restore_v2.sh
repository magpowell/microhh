#!/bin/bash
#SBATCH --qos=xfer
#SBATCH --time=12:00:00
#SBATCH --job-name=htar_restore_v2
#SBATCH --licenses=SCRATCH
#
# Restore the base-case (no_aerosols_zero_wind_v2) subset needed by characterize/ from HPSS.
# Member list (path) and expected sizes (bytes path) are built from the HPSS index beforehand.
# NERSC only. --output is passed at submit time.
set -uo pipefail
PARENT=${PARENT:-$SCRATCH/CASS_LES/experiments}
TAR=/home/m/mpowell/CASS_LES/no_aerosols_zero_wind_v2/no_aerosols_zero_wind_v2.tar
NAME=${NAME:-members_v2_characterize}      # member list basename in $PARENT
LIST=$PARENT/$NAME.txt
SIZES=$PARENT/$NAME.sizes

cd "$PARENT" || exit 1
echo "Restoring $(wc -l < "$LIST") members into $PARENT at $(date)"
htar -xf "$TAR" -L "$LIST"
echo "htar exit $? at $(date)"

bad=0
while read -r size path; do
    if [[ ! -f "$path" ]]; then echo "MISSING $path"; bad=$((bad+1)); continue; fi
    got=$(stat -c %s "$path")
    if [[ "$got" != "$size" ]]; then echo "SIZE $path expected $size got $got"; bad=$((bad+1)); fi
done < "$SIZES"
if [[ $bad -eq 0 ]]; then echo "RESTORE OK: $(wc -l < "$SIZES") files verified"; else echo "RESTORE FAILED: $bad problems"; exit 1; fi
