#!/bin/bash
#SBATCH --qos=xfer
#SBATCH --time=12:00:00
#SBATCH --job-name=htar_no_aerosols
#SBATCH --licenses=SCRATCH
#SBATCH --output=/pscratch/sd/m/mpowell/CASS_LES/logs/htar_no_aerosols_%j.out

cd /pscratch/sd/m/mpowell/CASS_LES/experiments

ARCHIVE="/home/m/mpowell/CASS_LES/no_aerosols/no_aerosols.tar"

if htar -tf "${ARCHIVE}" > /dev/null 2>&1; then
    echo "SKIP: ${ARCHIVE} already exists on HPSS"
    exit 0
fi

if [ ! -d "no_aerosols" ]; then
    echo "FAIL: no_aerosols/ not found on pscratch"
    exit 1
fi

echo "Archiving no_aerosols..."
htar -Pcf "${ARCHIVE}" "no_aerosols/"
rc=$?

if [ $rc -ne 0 ]; then
    echo "FAIL: htar exit code ${rc}"
    exit $rc
fi

n_local=$(find no_aerosols/ -type f | wc -l)
n_hpss_files=$(htar -tf "${ARCHIVE}" 2>&1 | grep -c "^HTAR: -rw")

echo "  Local files: ${n_local}, HPSS members: ${n_hpss_files}"

if [ "${n_hpss_files}" -ge "${n_local}" ]; then
    echo "OK: no_aerosols archived and verified"
else
    echo "WARN: no_aerosols file count mismatch (local=${n_local}, hpss=${n_hpss_files})"
fi

echo "Done."
