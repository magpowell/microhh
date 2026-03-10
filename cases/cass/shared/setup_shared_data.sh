#!/usr/bin/env bash
# setup_shared_data.sh
#
# Populate $SCRATCH/CASS_LES/shared_data/ and create symlinks in shared/data/.
#
# Run once (or re-run to repair broken symlinks).
# Source for van_genuchten_parameters.nc: microhh/misc/ (canonical repo copy).
# All other large data files live on scratch to preserve home quota.
#
# Usage:
#   cd cases/cass/shared/
#   bash setup_shared_data.sh

set -euo pipefail

SCRATCH_DATA="/pscratch/sd/m/mpowell/CASS_LES/shared_data"
DATA_DIR="$(cd "$(dirname "$0")/data" && pwd)"
MISC_DIR="/global/homes/m/mpowell/repos/microhh/misc"

echo "Scratch data dir : $SCRATCH_DATA"
echo "Shared data dir  : $DATA_DIR"
echo ""

mkdir -p "$SCRATCH_DATA"

# --- van_genuchten_parameters.nc ---
# Source: canonical copy in microhh/misc/
SRC="$MISC_DIR/van_genuchten_parameters.nc"
DST="$SCRATCH_DATA/van_genuchten_parameters.nc"
if [[ ! -f "$DST" ]]; then
    echo "Copying van_genuchten_parameters.nc from misc/ ..."
    cp "$SRC" "$DST"
else
    echo "van_genuchten_parameters.nc already on scratch — skipping copy"
fi

# --- CASS txt files ---
# Re-download from NERSC portal if not present on scratch.
for URL_PAIR in \
    "https://portal.nersc.gov/project/capt/CASS/snd cass_snd.txt" \
    "https://portal.nersc.gov/project/capt/CASS/sfc cass_sfc.txt" \
    "https://portal.nersc.gov/project/capt/CASS/lsf cass_lsf.txt"; do
    URL="${URL_PAIR%% *}"
    FNAME="${URL_PAIR##* }"
    if [[ ! -f "$SCRATCH_DATA/$FNAME" ]]; then
        echo "Downloading $FNAME ..."
        wget -q "$URL" -O "$SCRATCH_DATA/$FNAME"
    else
        echo "$FNAME already on scratch — skipping download"
    fi
done

# --- Create symlinks in shared/data/ ---
echo ""
echo "Creating symlinks in shared/data/ ..."
for FNAME in \
    van_genuchten_parameters.nc \
    cass_snd.txt \
    cass_sfc.txt \
    cass_lsf.txt; do
    LINK="$DATA_DIR/$FNAME"
    TARGET="$SCRATCH_DATA/$FNAME"
    if [[ -L "$LINK" ]]; then
        rm "$LINK"
    elif [[ -f "$LINK" ]]; then
        echo "WARNING: $LINK is a real file — moving to scratch before symlinking"
        mv "$LINK" "$TARGET"
    fi
    ln -s "$TARGET" "$LINK"
    echo "  $FNAME -> $TARGET"
done

# --- Generated nc files: symlink if already on scratch, warn if not ---
for FNAME in cass_ls2d_input.nc cass_cams_composite.nc; do
    TARGET="$SCRATCH_DATA/$FNAME"
    LINK="$DATA_DIR/$FNAME"
    if [[ -f "$TARGET" ]]; then
        if [[ -L "$LINK" ]]; then rm "$LINK"; fi
        ln -s "$TARGET" "$LINK"
        echo "  $FNAME -> $TARGET"
    else
        echo "  WARNING: $FNAME not found on scratch — run the appropriate preprocessing script first:"
        if [[ "$FNAME" == "cass_ls2d_input.nc" ]]; then
            echo "    python shared/preprocessing/cass_ls2d_input.py"
        else
            echo "    python shared/preprocessing/compute_cams_composite.py"
        fi
    fi
done

echo ""
echo "Done."
