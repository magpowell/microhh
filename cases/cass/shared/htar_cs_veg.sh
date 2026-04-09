#!/bin/bash
#SBATCH --qos=xfer
#SBATCH --time=12:00:00
#SBATCH --job-name=htar_cs_veg
#SBATCH --licenses=SCRATCH
#SBATCH --output=/pscratch/sd/m/mpowell/CASS_LES/logs/htar_cs_veg_%j.out

cd /pscratch/sd/m/mpowell/CASS_LES/experiments/cs_veg

for val in cs_veg_0 cs_veg_41840 cs_veg_418400 cs_veg_4184000 cs_veg_41840000; do
    ARCHIVE="/home/m/mpowell/CASS_LES/cs_veg/${val}.tar"

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
