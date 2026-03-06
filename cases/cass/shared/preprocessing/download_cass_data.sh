#!/bin/bash
#
# Download CASS case data from NERSC portal
# CASS: Composite Atmospheric Sounding for Shallow-cumulus
# Based on ARM SGP observations (July 24, DOY 205)
#

CASS_URL="https://portal.nersc.gov/project/capt/CASS"

echo "Downloading CASS case data..."

# Download atmospheric sounding (initial profiles)
echo "  - Atmospheric sounding (snd)..."
wget -q ${CASS_URL}/snd -O cass_snd.txt

# Download surface fluxes
echo "  - Surface fluxes (sfc)..."
wget -q ${CASS_URL}/sfc -O cass_sfc.txt

# Download large-scale forcings
echo "  - Large-scale forcings (lsf)..."
wget -q ${CASS_URL}/lsf -O cass_lsf.txt

echo "Download complete!"
echo ""
echo "Downloaded files:"
ls -lh cass_*.txt
