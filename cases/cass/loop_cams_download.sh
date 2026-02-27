#!/bin/bash
# Repeatedly runs download_cams.py with a 5-minute pause between runs.
# Each run submits at most one new ADS request, then exits.
# Stop with Ctrl+C once all 60 days are reported complete.

while true; do
    echo "--- $(date) ---"
    python download_cams.py
    echo "Sleeping 2.5 minutes..."
    sleep 150
done
