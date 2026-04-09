#!/bin/bash
#SBATCH --qos=xfer
#SBATCH --time=12:00:00
#SBATCH --job-name=htar_wind_u
#SBATCH --licenses=SCRATCH
#SBATCH --output=/pscratch/sd/m/mpowell/CASS_LES/logs/htar_wind_u_%j.out

cd /pscratch/sd/m/mpowell/CASS_LES/experiments/wind_u

for val in u_0p0 u_2p5 u_5p0 u_7p5 u_10p0; do
    ARCHIVE="/home/m/mpowell/CASS_LES/wind_u/${val}.tar"

    if htar -tf "${ARCHIVE}" > /dev/null 2>&1; then
        echo "SKIP: ${ARCHIVE} already exists on HPSS"
        continue
    fi

    if [ ! -d "${val}" ]; then
        echo "SKIP: ${val}/ not found on pscratch"
        continue
    fi

    echo "Archiving ${val}..."
    htar -Pcf "${ARCHIVE}" "${val}/"
    rc=$?

    if [ $rc -ne 0 ]; then
        echo "FAIL: ${val} (exit code ${rc})"
        continue
    fi

    n_local=$(find "${val}/" -type f | wc -l)
    n_hpss_files=$(htar -tf "${ARCHIVE}" 2>/dev/null | grep -v "^HTAR:" | wc -l)

    echo "  Local files: ${n_local}, HPSS members: ${n_hpss_files}"

    if [ "${n_hpss_files}" -ge "${n_local}" ]; then
        echo "OK: ${val} archived and verified"
    else
        echo "WARN: ${val} file count mismatch (local=${n_local}, hpss=${n_hpss_files})"
    fi
done

echo "Done."
