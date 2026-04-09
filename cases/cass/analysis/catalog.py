"""
CASS LES experiment catalog.

Single source of truth for all experiment RunSets, paths, labels, and sweep metadata.
All analysis notebooks import from here — no path hardcoding in notebooks.

Usage
-----
    from catalog import make_runset, list_group, ALL_EXPERIMENTS

    # One experiment
    rs = make_runset("no_aerosols_zero_wind")

    # All experiments in a sweep group (sorted by param)
    for key, meta in list_group("rs_scale"):
        rs = make_runset(key)
        label = meta["label"]
        param = meta["param"]
"""

import os
import sys
from pathlib import Path

# ── Repo / scratch roots ──────────────────────────────────────────────────────
_SCRATCH   = Path(os.environ.get("SCRATCH", "/pscratch/sd/m/mpowell"))
CASS_ROOT  = _SCRATCH / "CASS_LES"

_ANALYSIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_ANALYSIS_DIR))
from cass_analysis import RunSet

# ══════════════════════════════════════════════════════════════════════════════
# Experiment registry
# ══════════════════════════════════════════════════════════════════════════════
#
# Each entry:
#   label    : legend / title string (plain text, ASCII only)
#   root     : Path to directory containing {2stream,raytracer}/rep_{01..04}/
#   group    : experiment group name (used for sweep plots)
#   rt_types : radiation types present (default: both)
#   color    : matplotlib color for sweep-line plots
#   param    : sweep parameter numeric value for sorting (group-specific units)
#
ALL_EXPERIMENTS: dict[str, dict] = {
    # ── baseline ──────────────────────────────────────────────────────────────
    "base": dict(
        label  = "Base",
        root   = CASS_ROOT / "base",
        group  = "aerosol",
        color  = "C4",
    ),
    # ── no_aerosols (standard wind, aerosols off) ─────────────────────────────
    "no_aerosols": dict(
        label  = "No aerosols",
        root   = CASS_ROOT / "experiments/no_aerosols",
        group  = "aerosol",
        color  = "C0",
    ),
    # ── no_aerosols_zero_wind ─────────────────────────────────────────────────
    "no_aerosols_zero_wind": dict(
        label  = "No aerosols, zero wind",
        root   = CASS_ROOT / "experiments/no_aerosols_zero_wind",
        group  = "aerosol",
        color  = "C1",
    ),
}

# Archived experiments (data on HPSS, recoverable from git history):
# cs_veg, soil_moisture, mean_state_nudge, wind_u


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_runset(key: str, n_reps: int = 4) -> RunSet:
    """Return a RunSet for the given experiment key."""
    meta     = ALL_EXPERIMENTS[key]
    rt_types = meta.get("rt_types", ("2stream", "raytracer"))
    return RunSet(meta["label"], meta["root"], n_reps=n_reps, rt_types=rt_types)


def list_group(group: str) -> list[tuple[str, dict]]:
    """Return [(key, meta), ...] for all experiments in *group*, sorted by param."""
    items = [(k, v) for k, v in ALL_EXPERIMENTS.items() if v.get("group") == group]
    items.sort(key=lambda x: x[1].get("param", float("inf")))
    return items


def available_keys() -> list[str]:
    """Return experiment keys where at least one rep's stats file exists on disk."""
    out = []
    for key, meta in ALL_EXPERIMENTS.items():
        rt_types = meta.get("rt_types", ("2stream", "raytracer"))
        for rt in rt_types:
            p = Path(meta["root"]) / rt / "rep_01" / "cass.default.0000000.nc"
            if p.exists():
                out.append(key)
                break
    return out
