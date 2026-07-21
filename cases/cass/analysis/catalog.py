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
    # As of 2026-05, this key points at the v2 dataset (which now includes the
    # Couvreux passive tracer + SGS-flux split + extended sim to LST 19:20).
    # The v1 path (`experiments/no_aerosols_zero_wind`) is pending HPSS archive.
    "no_aerosols_zero_wind": dict(
        label  = "No aerosols, zero wind",
        root   = CASS_ROOT / "experiments/no_aerosols_zero_wind_v2",
        group  = "aerosol",
        color  = "C1",
    ),
    # Explicit v2 alias, kept so cells that reference V2_EXPT directly still resolve.
    "no_aerosols_zero_wind_v2": dict(
        label  = "No aerosols, zero wind",
        root   = CASS_ROOT / "experiments/no_aerosols_zero_wind_v2",
        group  = "aerosol",
        color  = "C1",
    ),
    # Explicit v1 alias — same physics minus the Couvreux passive tracer.
    # Kept on scratch only until HPSS archive; used for v1↔v2 sanity comparisons.
    "no_aerosols_zero_wind_v1": dict(
        label  = "No aerosols, zero wind (v1, no tracer)",
        root   = CASS_ROOT / "experiments/no_aerosols_zero_wind",
        group  = "aerosol",
        color  = "C7",
    ),
    # ── wind_geo (geostrophic wind sweep) ─────────────────────────────────────
    # u_g = 0 reference: aliases no_aerosols_zero_wind data root so it shows up
    # in list_group("wind_geo") as the starting point.
    "wind_geo_0": dict(
        label  = r"$u_g = 0\,\mathrm{m\,s^{-1}}$",
        root   = CASS_ROOT / "experiments/no_aerosols_zero_wind_v2",  # baseline → v2
        group  = "wind_geo",
        param  = 0.0,
        color  = "#dadaeb",
    ),
    "wind_geo_2p5": dict(
        label  = r"$u_g = 2.5\,\mathrm{m\,s^{-1}}$",
        root   = CASS_ROOT / "experiments/wind_geo/u_2p5",
        group  = "wind_geo",
        param  = 2.5,
        color  = "#9e9ac8",
    ),
    "wind_geo_5p0": dict(
        label  = r"$u_g = 5\,\mathrm{m\,s^{-1}}$",
        root   = CASS_ROOT / "experiments/wind_geo/u_5p0",
        group  = "wind_geo",
        param  = 5.0,
        color  = "#756bb1",
    ),
    "wind_geo_7p5": dict(
        label  = r"$u_g = 7.5\,\mathrm{m\,s^{-1}}$",
        root   = CASS_ROOT / "experiments/wind_geo/u_7p5",
        group  = "wind_geo",
        param  = 7.5,
        color  = "#54278f",
    ),
    "wind_geo_10p0": dict(
        label  = r"$u_g = 10\,\mathrm{m\,s^{-1}}$",
        root   = CASS_ROOT / "experiments/wind_geo/u_10p0",
        group  = "wind_geo",
        param  = 10.0,
        color  = "#3f007d",
    ),
    # ── wind_sun (U=5 wind whose direction tracks the solar azimuth, so ───────
    # clouds advect toward their own shadows). Standalone: same 5 m/s speed as
    # wind_geo_5p0 but shadow-tracking direction — its own group.
    "wind_sun": dict(
        label  = r"wind_sun ($U=5\,\mathrm{m\,s^{-1}}$, $\odot$-tracking)",
        root   = CASS_ROOT / "experiments/wind_sun",
        group  = "wind_sun",
        param  = 5.0,
        color  = "C1",
    ),
    # ── rs_scale (surface-resistance multiplier; Bowen-ratio sweep) ───────────
    # rs_scale = 1.0 reference: aliases no_aerosols_zero_wind data root so the
    # sweep has a baseline anchor (rs_scale=1 is the default, no scaling).
    "rs_scale_0p25": dict(
        label  = r"$f_{r_s} = 0.25$",
        root   = CASS_ROOT / "experiments/rs_scale/rs_0p25",
        group  = "rs_scale",
        param  = 0.25,
        color  = "#c7e9c0",
    ),
    "rs_scale_0p5": dict(
        label  = r"$f_{r_s} = 0.5$",
        root   = CASS_ROOT / "experiments/rs_scale/rs_0p5",
        group  = "rs_scale",
        param  = 0.5,
        color  = "#a1d99b",
    ),
    "rs_scale_1": dict(
        label  = r"$f_{r_s} = 1$",
        root   = CASS_ROOT / "experiments/no_aerosols_zero_wind_v2",  # baseline → v2
        group  = "rs_scale",
        param  = 1.0,
        color  = "#74c476",
    ),
    "rs_scale_2": dict(
        label  = r"$f_{r_s} = 2$",
        root   = CASS_ROOT / "experiments/rs_scale/rs_2",
        group  = "rs_scale",
        param  = 2.0,
        color  = "#31a354",
    ),
    "rs_scale_4": dict(
        label  = r"$f_{r_s} = 4$",
        root   = CASS_ROOT / "experiments/rs_scale/rs_4",
        group  = "rs_scale",
        param  = 4.0,
        color  = "#006d2c",
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
