"""
CASS LES experiment catalog.

Single source of truth for all experiment RunSets, paths, labels, and sweep metadata.
All analysis notebooks import from here — no path hardcoding in notebooks.

Usage
-----
    from catalog import make_runset, list_group, ALL_EXPERIMENTS

    # One experiment
    rs = make_runset("base")

    # All cs_veg sweep experiments (sorted by param)
    for key, meta in list_group("cs_veg"):
        rs = make_runset(key)
        label = meta["label"]
        param = meta["param"]   # cs_veg value (J m-2 K-1)
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
    # ── cs_veg sweep  (aerosols off, zero wind) ───────────────────────────────
    "cs_veg_0": dict(
        label  = r"$c_s = 0$",
        root   = CASS_ROOT / "experiments/cs_veg/cs_veg_0",
        group  = "cs_veg",
        param  = 0,
        color  = "#e8c99a",
    ),
    "cs_veg_41840": dict(
        label  = r"$c_s = 4.2\times10^4$",
        root   = CASS_ROOT / "experiments/cs_veg/cs_veg_41840",
        group  = "cs_veg",
        param  = 41_840,
        color  = "#c9903c",
    ),
    "cs_veg_418400": dict(
        label  = r"$c_s = 4.2\times10^5$",
        root   = CASS_ROOT / "experiments/cs_veg/cs_veg_418400",
        group  = "cs_veg",
        param  = 418_400,
        color  = "#a0522d",
    ),
    "cs_veg_4184000": dict(
        label  = r"$c_s = 4.2\times10^6$",
        root   = CASS_ROOT / "experiments/cs_veg/cs_veg_4184000",
        group  = "cs_veg",
        param  = 4_184_000,
        color  = "#7a3217",
    ),
    "cs_veg_41840000": dict(
        label  = r"$c_s = 4.2\times10^7$",
        root   = CASS_ROOT / "experiments/cs_veg/cs_veg_41840000",
        group  = "cs_veg",
        param  = 41_840_000,
        color  = "#4a1a08",
    ),
    # ── soil_moisture sweep  (aerosols off, zero wind) ────────────────────────
    # "theta_0p1": dict(
    #     label  = r"$\theta = 0.1$",
    #     root   = CASS_ROOT / "experiments/soil_moisture/theta_0p1",
    #     group  = "soil_moisture",
    #     param  = 0.1,
    #     color  = "#9ecae1",
    # ),
    "theta_0p155": dict(
        label  = r"$\theta = 0.155$",
        root   = CASS_ROOT / "experiments/soil_moisture/theta_0p155",
        group  = "soil_moisture",
        param  = 0.155,
        color  = "#7eb5d6",
    ),
    "theta_0p17": dict(
        label  = r"$\theta = 0.170$",
        root   = CASS_ROOT / "experiments/soil_moisture/theta_0p17",
        group  = "soil_moisture",
        param  = 0.170,
        color  = "#62a8cf",
    ),
    "theta_0p185": dict(
        label  = r"$\theta = 0.185$",
        root   = CASS_ROOT / "experiments/soil_moisture/theta_0p185",
        group  = "soil_moisture",
        param  = 0.185,
        color  = "#4d9bc8",
    ),
    "theta_0p2": dict(
        label  = r"$\theta = 0.2$",
        root   = CASS_ROOT / "experiments/soil_moisture/theta_0p2",
        group  = "soil_moisture",
        param  = 0.2,
        color  = "#4292c6",
    ),
    "theta_0p225": dict(
        label  = r"$\theta = 0.225$",
        root   = CASS_ROOT / "experiments/soil_moisture/theta_0p225",
        group  = "soil_moisture",
        param  = 0.225,
        color  = "#3a8ac2",
    ),
    "theta_0p25": dict(
        label  = r"$\theta = 0.25$",
        root   = CASS_ROOT / "experiments/soil_moisture/theta_0p25",
        group  = "soil_moisture",
        param  = 0.25,
        color  = "#3282be",
    ),
    # "theta_0p3": dict(
    #     label  = r"$\theta = 0.3$",
    #     root   = CASS_ROOT / "experiments/soil_moisture/theta_0p3",
    #     group  = "soil_moisture",
    #     param  = 0.3,
    #     color  = "#2171b5",
    # ),
    # "theta_0p4": dict(
    #     label  = r"$\theta = 0.4$",
    #     root   = CASS_ROOT / "experiments/soil_moisture/theta_0p4",
    #     group  = "soil_moisture",
    #     param  = 0.4,
    #     color  = "#084594",
    # ),
    # ── mean_state_nudge  (raytracer only; control is no_aerosols_zero_wind 2stream) ──
    "mean_state_nudge_3600s": dict(
        label    = r"Nudged RT ($\tau = 3600\,\mathrm{s}$)",
        root     = CASS_ROOT / "experiments/mean_state_nudge/nudge_3600s",
        group    = "mean_state_nudge",
        rt_types = ("raytracer",),
        param    = 3600,
        color    = "C1",
    ),
    # ── wind_u sweep  (aerosols off; swlspres=uflux) ─────────────────────────
    "wind_u_0p0": dict(
        label  = r"$u = 0\,\mathrm{m\,s^{-1}}$",
        root   = CASS_ROOT / "experiments/wind_u/u_0p0",
        group  = "wind_u",
        param  = 0.0,
        color  = "#dadaeb",
    ),
    "wind_u_2p5": dict(
        label  = r"$u = 2.5\,\mathrm{m\,s^{-1}}$",
        root   = CASS_ROOT / "experiments/wind_u/u_2p5",
        group  = "wind_u",
        param  = 2.5,
        color  = "#9e9ac8",
    ),
    "wind_u_5p0": dict(
        label  = r"$u = 5\,\mathrm{m\,s^{-1}}$",
        root   = CASS_ROOT / "experiments/wind_u/u_5p0",
        group  = "wind_u",
        param  = 5.0,
        color  = "#756bb1",
    ),
    "wind_u_7p5": dict(
        label  = r"$u = 7.5\,\mathrm{m\,s^{-1}}$",
        root   = CASS_ROOT / "experiments/wind_u/u_7p5",
        group  = "wind_u",
        param  = 7.5,
        color  = "#54278f",
    ),
    "wind_u_10p0": dict(
        label  = r"$u = 10\,\mathrm{m\,s^{-1}}$",
        root   = CASS_ROOT / "experiments/wind_u/u_10p0",
        group  = "wind_u",
        param  = 10.0,
        color  = "#3f007d",
    ),
}


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
