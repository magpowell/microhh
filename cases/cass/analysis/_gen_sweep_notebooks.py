"""Generate cs_veg_comparison.ipynb, soil_moisture_comparison.ipynb, wind_u_comparison.ipynb, wind_geo_comparison.ipynb.

Usage: python _gen_sweep_notebooks.py [group1 group2 ...]
If no groups given, regenerates all four. Valid groups: cs_veg, soil_moisture, wind_u, wind_geo.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
_ALL_GROUPS = ("cs_veg", "soil_moisture", "wind_u", "wind_geo")
_TARGETS = set(sys.argv[1:]) if len(sys.argv) > 1 else set(_ALL_GROUPS)


# ── Notebook building helpers ─────────────────────────────────────────────────

def md(src: str):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ── Shared preamble code (imports + config factory) ───────────────────────────

IMPORTS = """\
import os, sys, pickle, warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from cass_analysis import (
    load_stats, load_stats_ensemble,
    lwp_integral,
    RunSet, _to_plottime,
    rho, cp, Lv,
    LST_OFFSET,
)
from catalog import make_runset, list_group, CASS_ROOT, ALL_EXPERIMENTS
"""


def config_cell(group, title, sweep_filter="", param_unit="", extra_note=""):
    lines = [
        f'COMPOSITE_ROOT = CASS_ROOT / "analysis/cloud_root_composite"\n',
        f'N_REPS         = 4\n',
        f'LST_OFF        = LST_OFFSET   # from cass_analysis (5.5 h)\n',
        f'LST_MIN_H, LST_MAX_H = 12.5, 19.0\n',
        f'\n',
        f'SWEEP_GROUP = "{group}"\n',
        f'_all_sweep  = list_group(SWEEP_GROUP)\n',
    ]
    if sweep_filter:
        lines.append(sweep_filter + '\n')
    else:
        lines.append('sweep = _all_sweep\n')
    lines += [
        f'\n',
        f'PARAM_UNIT = "{param_unit}"\n',
        f'\n',
        f'# ── colour / style: colour per param value, solid=2stream, dashed=raytracer ─\n',
        f'COLORS  = {{key: meta["color"]  for key, meta in sweep}}\n',
        f'LABELS  = {{key: meta["label"]  for key, meta in sweep}}\n',
        f'PARAMS  = {{key: meta["param"]  for key, meta in sweep}}\n',
        f'\n',
        f'def lst_mask(s, lo=LST_MIN_H, hi=LST_MAX_H):\n',
        f'    t_local = s["t_local"].values if hasattr(s, "data_vars") else (s["t_local"] if isinstance(s, dict) else s)\n',
        f'    import pandas as _pd\n',
        f'    hours = _pd.DatetimeIndex(t_local).hour + _pd.DatetimeIndex(t_local).minute / 60\n',
        f'    return (hours >= lo) & (hours <= hi)\n',
        f'\n',
        f'def axvspan_window(ax, t_local, mask, **kwargs):\n',
        f'    t_num = _to_plottime(t_local)\n',
        f'    valid = t_num[mask]\n',
        f'    if len(valid):\n',
        f'        ax.axvspan(valid[0], valid[-1], **kwargs)\n',
        f'\n',
        f'# ── PDF output switch ────────────────────────────────────────────────────────\n',
        f'SAVE_PDF = False   # set True to write .pdf files\n',
        f'\n',
        f'def _savefig(fname, **kwargs):\n',
        f'    if SAVE_PDF:\n',
        f'        plt.savefig(fname, **kwargs)\n',
    ]
    if extra_note:
        lines.append(f'\n')
        lines.append(extra_note + '\n')
    return code("".join(lines))


DATA_LOAD = """\
data = {}  # key -> dict with all loaded arrays

for key, meta in sweep:
    rs = make_runset(key, n_reps=N_REPS)
    dirs_2s = rs.dirs.get("2stream",   [])
    dirs_rt = rs.dirs.get("raytracer", [])

    if not dirs_2s:
        print(f"  {key}: no 2stream reps found, skipping")
        continue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reps_2s = [load_stats(d) for d in dirs_2s]
        reps_rt = [load_stats(d) for d in dirs_rt] if dirs_rt else []

    # ── Ensemble means ──────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s2s_mean, s2s_std = load_stats_ensemble(dirs_2s)
        srt_mean, srt_std = (load_stats_ensemble(dirs_rt)
                              if dirs_rt else (None, None))

    # ── LWP per-rep (for σ across reps) ────────────────────────────────────
    nt_lwp  = min(s["qlqi_path"].sizes["time"] for s in reps_2s)
    if reps_rt:
        nt_lwp = min(nt_lwp, *(s["qlqi_path"].sizes["time"] for s in reps_rt))
    n_paired = min(len(reps_2s), len(reps_rt)) if reps_rt else len(reps_2s)

    lwp_1d_reps = np.stack([s["qlqi_path"].values[:nt_lwp] * 1e3 for s in reps_2s[:n_paired]])
    lwp_1d      = lwp_1d_reps.mean(axis=0)
    lwp_1d_std  = lwp_1d_reps.std(axis=0)

    if reps_rt:
        lwp_rt_reps = np.stack([s["qlqi_path"].values[:nt_lwp] * 1e3 for s in reps_rt[:n_paired]])
        lwp_rt      = lwp_rt_reps.mean(axis=0)
        lwp_rt_std  = lwp_rt_reps.std(axis=0)
        diff_reps   = lwp_rt_reps - lwp_1d_reps
        diff        = diff_reps.mean(axis=0)
        diff_std    = diff_reps.std(axis=0)
    else:
        lwp_rt = lwp_rt_std = diff = diff_std = diff_reps = None

    t_local_lwp = reps_2s[0]["t_local"].values[:nt_lwp]
    t_sec_lwp   = reps_2s[0]["t_sec"].values[:nt_lwp]
    dt_s        = float(t_sec_lwp[1] - t_sec_lwp[0]) if len(t_sec_lwp) > 1 else 60.0
    t_num_lwp   = _to_plottime(t_local_lwp)

    if diff is not None:
        cloudy_mask = (lwp_1d > 0.1) | (lwp_rt > 0.1)
        int_diff    = lwp_integral(diff, dt_s, cloudy_mask=cloudy_mask)
        int_1d      = lwp_integral(lwp_1d, dt_s, cloudy_mask=cloudy_mask)
        norm_int    = int_diff / int_1d * 100 if abs(int_1d) > 1e-6 else np.nan
        norm_int_std = float(np.std(
            [lwp_integral(diff_reps[r], dt_s, cloudy_mask=cloudy_mask) / int_1d * 100
             for r in range(n_paired)], ddof=0)) if abs(int_1d) > 1e-6 else np.nan
    else:
        norm_int = norm_int_std = None

    data[key] = dict(
        dirs_2s=dirs_2s, dirs_rt=dirs_rt,
        reps_2s=reps_2s, reps_rt=reps_rt,
        n_paired=n_paired,
        s2s_mean=s2s_mean, s2s_std=s2s_std,
        srt_mean=srt_mean, srt_std=srt_std,
        lwp_1d=lwp_1d, lwp_1d_std=lwp_1d_std,
        lwp_rt=lwp_rt, lwp_rt_std=lwp_rt_std,
        diff=diff, diff_std=diff_std,
        norm_int=norm_int, norm_int_std=norm_int_std,
        t_local_lwp=t_local_lwp, t_num_lwp=t_num_lwp, dt_s=dt_s,
    )
    print(f"  {key:25s}  2s={len(dirs_2s)}  rt={len(dirs_rt)}  "
          f"norm_int={norm_int:+.1f}%" if norm_int is not None
          else f"  {key:25s}  2s={len(dirs_2s)}  rt=0 (no raytracer)")

available = [k for k in data]
print(f"\\nLoaded {len(available)} sweep values: {available}")
"""

LWP_SUMMARY = """\
# ── LWP summary: mean difference + cloudy-period integral ────────────────────
keys_d = [k for k in available if data[k]["diff"] is not None]

fig, (ax_diff, ax_int) = plt.subplots(1, 2, figsize=(12, 4))

for key in keys_d:
    d   = data[key]
    c   = COLORS[key]
    lbl = LABELS[key]
    ax_diff.plot(d["t_num_lwp"], d["diff"], color=c, lw=1.5, label=lbl)
    ax_diff.fill_between(d["t_num_lwp"],
                         d["diff"] - d["diff_std"],
                         d["diff"] + d["diff_std"], color=c, alpha=0.15)

ax_diff.axhline(0, color="gray", lw=0.7, ls="--")
ax_diff.set_ylabel(r"3D − 1D LWP  (g m$^{-2}$)", fontsize=9)
ax_diff.set_title("(a) Mean difference")
ax_diff.legend(fontsize=8)
ax_diff.xaxis_date()

norm_ints = [data[k]["norm_int"] for k in keys_d]
norm_stds = [data[k]["norm_int_std"] for k in keys_d]
colors_v  = [COLORS[k]              for k in keys_d]
x = np.arange(len(keys_d))
ax_int.bar(x, norm_ints, color=colors_v, edgecolor="k", lw=0.8, alpha=0.85,
           yerr=norm_stds, capsize=5, error_kw=dict(lw=1.5))
ax_int.axhline(0, color="gray", lw=0.7)
ax_int.set_xticks(x)
ax_int.set_xticklabels([LABELS[k] for k in keys_d], rotation=15, ha="right", fontsize=9)
ax_int.set_xlabel(f"{SWEEP_GROUP}  ({PARAM_UNIT})")
ax_int.set_ylabel(r"$\int$(3D $-$ 1D) / $\int$(1D)  (%)", fontsize=9)
ax_int.set_title("(b) Normalised cloudy-period integral")

ax_diff.figure.autofmt_xdate()

fig.suptitle(f"{SWEEP_GROUP} sweep: 3D − 1D LWP  (ensemble mean ±1σ)", fontsize=12)
plt.tight_layout()
_savefig(f"{SWEEP_GROUP}_lwp_summary.pdf", bbox_inches="tight")
plt.show()
"""

THERMO_TIMEH = """\
# ── q_l time-height cross-sections per param value ───────────────────────────
# Load thermo profiles (from cass.default.0000000.nc)
thermo = {}
for key in available:
    d = data[key]
    reps = []
    for rt_key, rep_list in [("2stream", d["reps_2s"]), ("raytracer", d["reps_rt"])]:
        if not rep_list:
            continue
        _reps_t = []
        for _s in (d["reps_rt"] if rt_key == "raytracer" else d["reps_2s"]):
            _reps_t.append({
                "t": _s["t_sec"].values,
                "z": _s.coords["z"].values,
                "ql": _s["ql"].values * 1e3,
                "thl": _s["thl"].values,
                "qt":  _s["qt"].values * 1e3,
            })
        if not _reps_t:
            continue
        nt_ = min(r["t"].shape[0] for r in _reps_t)
        avg = {"t": _reps_t[0]["t"][:nt_], "z": _reps_t[0]["z"]}
        for _k in ["ql", "thl", "qt"]:
            avg[_k] = np.stack([r[_k][:nt_] for r in _reps_t]).mean(axis=0)
        reps.append((rt_key, avg))
    thermo[key] = reps

# 3 rows: 3D (raytracer), 1D (2stream), 3D − 1D diff.
# Shared abs scale for rows 0–1; symmetric diff scale for row 2.
n_vals = len(available)
fig, axes = plt.subplots(3, n_vals, figsize=(4.0 * n_vals, 10),
                         sharey=True, sharex=True, squeeze=False)
z_max = 4000

panels = []  # (col, key, t_h, z, ql_3d, ql_1d, diff) or None for missing
for col, key in enumerate(available):
    d_2s = d_rt = None
    for rt_key, avg in thermo[key]:
        if rt_key == "2stream":
            d_2s = avg
        elif rt_key == "raytracer":
            d_rt = avg
    if d_2s is None or d_rt is None:
        panels.append(None); continue
    z    = d_2s["z"]
    zmsk = z <= z_max
    nt   = min(d_2s["ql"].shape[0], d_rt["ql"].shape[0])
    ql_3 = d_rt["ql"][:nt, zmsk]
    ql_1 = d_2s["ql"][:nt, zmsk]
    t_h  = d_rt["t"][:nt] / 3600 + LST_OFF
    panels.append((col, key, t_h, z[zmsk], ql_3, ql_1, ql_3 - ql_1))

_qabs = [p[4] for p in panels if p] + [p[5] for p in panels if p]
_qdif = [p[6] for p in panels if p]
if _qabs:
    _pos = np.concatenate([q[q > 0].ravel() for q in _qabs if (q > 0).any()]) \\
            if any((q > 0).any() for q in _qabs) else np.array([0.01])
    vmax_abs  = max(float(np.nanpercentile(_pos, 99)), 0.01)
    _dif_flat = np.concatenate([np.abs(d).ravel() for d in _qdif])
    vmax_diff = max(float(np.nanpercentile(_dif_flat[np.isfinite(_dif_flat)], 99)), 0.01)
else:
    vmax_abs = vmax_diff = 0.01

pcm_abs = pcm_diff = None
for p in panels:
    if p is None:
        continue
    col, key, t_h, z_ax, ql_3, ql_1, diff = p
    pcm_abs  = axes[0, col].pcolormesh(t_h, z_ax, ql_3.T, cmap="Blues", vmin=0, vmax=vmax_abs)
    axes[1, col].pcolormesh(t_h, z_ax, ql_1.T, cmap="Blues", vmin=0, vmax=vmax_abs)
    pcm_diff = axes[2, col].pcolormesh(t_h, z_ax, diff.T, cmap="RdBu_r",
                                         vmin=-vmax_diff, vmax=vmax_diff)
    axes[0, col].set_title(LABELS[key], fontsize=9)
    axes[2, col].set_xlabel("LST  (h)")

for row, row_lbl in enumerate(["3D", "1D", "3D − 1D"]):
    axes[row, 0].set_ylabel(f"z (m)\\n{row_lbl}")

if pcm_abs is not None:
    fig.colorbar(pcm_abs,  ax=axes[:2, :], location="right", shrink=0.6,
                 label=r"$q_l$  (g kg$^{-1}$)")
    fig.colorbar(pcm_diff, ax=axes[2, :],  location="right", shrink=0.6,
                 label=r"$\\Delta q_l$  (g kg$^{-1}$)")
fig.suptitle(f"{SWEEP_GROUP}: q_l time-height (ensemble mean)")
_savefig(f"{SWEEP_GROUP}_ql_timeh.pdf", bbox_inches="tight")
plt.show()
"""

THERMO_PROFILES = """\
# ── Window-mean thermodynamic profiles ───────────────────────────────────────
import pandas as _pd
_T0 = _pd.Timestamp("2005-07-24 12:30")
_T1 = _pd.Timestamp("2005-07-24 17:00")

def _window_mean(avg_dict, key):
    t_idx = _pd.DatetimeIndex(
        _pd.to_datetime(avg_dict["t"], unit="s", origin=_pd.Timestamp("2005-07-24 05:30"))
    )
    msk = (t_idx >= _T0) & (t_idx <= _T1)
    return avg_dict[key][msk].mean(axis=0) if msk.any() else avg_dict[key].mean(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
ax_thl, ax_qt, ax_ql = axes
z_max_p = 4000

for key in available:
    c = COLORS[key]
    for rt_key, avg in thermo[key]:
        ls = "-" if rt_key == "2stream" else "--"
        lw = 1.5
        z  = avg["z"]
        z_msk = z <= z_max_p
        ax_thl.plot(_window_mean(avg, "thl")[z_msk], z[z_msk], color=c, ls=ls, lw=lw)
        ax_qt.plot( _window_mean(avg, "qt" )[z_msk], z[z_msk], color=c, ls=ls, lw=lw)
        ax_ql.plot( _window_mean(avg, "ql" )[z_msk], z[z_msk], color=c, ls=ls, lw=lw)

ax_thl.set_xlabel(r"$\\langle \\theta_l \\rangle$  (K)");  ax_thl.set_ylabel("z  (m)")
ax_qt.set_xlabel(r"$\\langle q_t \\rangle$  (g kg$^{-1}$)")
ax_ql.set_xlabel(r"$\\langle q_l \\rangle$  (g kg$^{-1}$)")
for ax in axes:
    ax.set_ylim(0, z_max_p)
_lh_c = [Line2D([0],[0], color=COLORS[k], lw=2, label=LABELS[k]) for k in available]
_lh_s = [Line2D([0],[0], color="k", ls="-",  lw=1.5, label="1D"),
          Line2D([0],[0], color="k", ls="--", lw=1.5, label="3D")]
ax_ql.legend(handles=_lh_c + _lh_s, fontsize=7)
fig.suptitle(f"{SWEEP_GROUP}: window-mean thermodynamic profiles  (LST 12:30–17:00)")
fig.tight_layout()
_savefig(f"{SWEEP_GROUP}_thermo_profiles.pdf", bbox_inches="tight")
plt.show()
"""

SEB_SECTION = """\
# ── SEB components, residual, EF ─────────────────────────────────────────────
fluxes = ["Rnet", "H", "LE", "G"]
flux_labels = {"Rnet": r"$R_\\mathrm{net}$", "H": "H", "LE": "LE", "G": "G"}

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

for key in available:
    d  = data[key]
    c  = COLORS[key]
    lbl = LABELS[key]
    for ax, flux in zip(axes.flat, fluxes):
        for rt_key, sm in [("2stream", d["s2s_mean"]), ("raytracer", d["srt_mean"])]:
            if sm is None:
                continue
            t_  = _to_plottime(sm["t_local"].values)
            ax.plot(t_, sm[flux].values,
                    color=c, ls="-" if rt_key == "2stream" else "--", lw=1.2)
        ax.set_ylabel(f"{flux_labels[flux]}  (W m⁻²)")
        ax.xaxis_date()

_lh_c = [Line2D([0],[0], color=COLORS[k], lw=2, label=LABELS[k]) for k in available]
_lh_s = [Line2D([0],[0], color="k", ls="-",  lw=1.5, label="1D"),
          Line2D([0],[0], color="k", ls="--", lw=1.5, label="3D")]
axes.flat[0].legend(handles=_lh_c + _lh_s, fontsize=6)

axes[0,0].set_title("$R_{net}$"); axes[0,1].set_title("H")
axes[1,0].set_title("LE");         axes[1,1].set_title("G")
fig.autofmt_xdate()
fig.tight_layout()
_savefig(f"{SWEEP_GROUP}_seb_components.pdf", bbox_inches="tight")
plt.show()
"""

SEB_BO = """\
# ── Bowen ratio: timeseries + vs sweep param ─────────────────────────────────
fig, (ax_bo, ax_bar) = plt.subplots(1, 2, figsize=(12, 4))

bo_1d_vals, bo_3d_vals = [], []

for key in available:
    d = data[key]
    c = COLORS[key]
    for rt_key, sm in [("2stream", d["s2s_mean"]), ("raytracer", d["srt_mean"])]:
        if sm is None:
            continue
        H   = sm["H"].values
        LE  = sm["LE"].values
        bo  = np.where(LE > 1.0, H / LE, np.nan)
        t_  = _to_plottime(sm["t_local"].values)
        day = (sm["Rnet"].values > 50) & (H > 0)
        ax_bo.plot(t_[day], bo[day], color=c,
                   ls="-" if rt_key == "2stream" else "--", lw=1.2)
        if day.any():
            bo_mean = float(np.nanmean(bo[day]))
            (bo_1d_vals if rt_key == "2stream" else bo_3d_vals).append(
                (PARAMS[key], bo_mean))

ax_bo.set_ylabel("Bo = H / LE  (–)")
ax_bo.set_title("(a) Bowen ratio  (daytime)")
_lh_c = [Line2D([0],[0], color=COLORS[k], lw=2, label=LABELS[k]) for k in available]
_lh_s = [Line2D([0],[0], color="k", ls="-",  lw=1.5, label="1D"),
          Line2D([0],[0], color="k", ls="--", lw=1.5, label="3D")]
ax_bo.legend(handles=_lh_c + _lh_s, fontsize=7, ncol=2)
ax_bo.xaxis_date()

if bo_1d_vals:
    p1d, b1d = zip(*sorted(bo_1d_vals))
    ax_bar.plot(p1d, b1d, "o-", color="C0", lw=1.5, label="1D")
if bo_3d_vals:
    p3d, b3d = zip(*sorted(bo_3d_vals))
    ax_bar.plot(p3d, b3d, "s--", color="C1", lw=1.5, label="3D")
ax_bar.set_xlabel(f"{SWEEP_GROUP}  ({PARAM_UNIT})")
ax_bar.set_ylabel("Daytime-mean Bo")
ax_bar.set_title("(b) Bo vs sweep param")
ax_bar.legend(fontsize=8)

fig.autofmt_xdate()
fig.tight_layout()
_savefig(f"{SWEEP_GROUP}_bowen_ratio.pdf", bbox_inches="tight")
plt.show()
"""

# ══════════════════════════════════════════════════════════════════════════════
# cs_veg_comparison.ipynb
# ══════════════════════════════════════════════════════════════════════════════

cs_veg_header = """\
# cs_veg sweep — comprehensive analysis

Skin heat capacity sweep (J&M replication).  All runs: aerosols off, zero wind.

| Key | cs_veg (J m⁻² K⁻¹) | Note |
|-----|-------------------|------|
| `cs_veg_0`      | 0              | no skin heat capacity |
| `cs_veg_41840`  | 4.2 × 10⁴     | 1 cm water equivalent |
| `cs_veg_418400` | 4.2 × 10⁵     | 10 cm water equivalent |

> Two highest values (cs_veg_4184000, cs_veg_41840000) excluded: clouds do not develop.

**Sections**
1. LWP overview
2. Thermodynamic structure
3. Surface energy balance
"""

cs_veg_config = config_cell(
    group="cs_veg",
    title="cs_veg",
    sweep_filter="sweep = _all_sweep[:3]   # drop two highest (no clouds develop)",
    param_unit=r"J m$^{-2}$ K$^{-1}$",
)

cs_veg_cells = [
    md(cs_veg_header),
    code(IMPORTS),
    cs_veg_config,
    md("---\n## Data loading"),
    code(DATA_LOAD),
    md("---\n## 1. LWP overview"),
    code(LWP_SUMMARY),
    md("---\n## 2. Thermodynamic structure"),
    code(THERMO_TIMEH),
    code(THERMO_PROFILES),
    md("---\n## 3. Surface energy balance"),
    code(SEB_SECTION),
    code(SEB_BO),
]

if "cs_veg" in _TARGETS:
    with open(HERE / "cs_veg_comparison.ipynb", "w") as f:
        json.dump(nb(cs_veg_cells), f, indent=1)
    print("Wrote cs_veg_comparison.ipynb")


# ══════════════════════════════════════════════════════════════════════════════
# soil_moisture_comparison.ipynb
# ══════════════════════════════════════════════════════════════════════════════

sm_header = """\
# soil_moisture sweep — comprehensive analysis

Soil moisture nudging sweep.  All runs: aerosols off, zero wind.

| Key | θ (m³ m⁻³) |
|-----|-----------|
| `theta_0p1` | 0.1 |
| `theta_0p2` | 0.2 |
| `theta_0p3` | 0.3 |
| `theta_0p4` | 0.4 |

**Sections**
1. LWP overview
2. Thermodynamic structure
3. Surface energy balance
"""

sm_config = config_cell(
    group="soil_moisture",
    title="soil_moisture",
    param_unit=r"m$^3$ m$^{-3}$",
)

sm_cells = [
    md(sm_header),
    code(IMPORTS),
    sm_config,
    md("---\n## Data loading"),
    code(DATA_LOAD),
    md("---\n## 1. LWP overview"),
    code(LWP_SUMMARY),
    md("---\n## 2. Thermodynamic structure"),
    code(THERMO_TIMEH),
    code(THERMO_PROFILES),
    md("---\n## 3. Surface energy balance"),
    code(SEB_SECTION),
    code(SEB_BO),
]

if "soil_moisture" in _TARGETS:
    with open(HERE / "soil_moisture_comparison.ipynb", "w") as f:
        json.dump(nb(sm_cells), f, indent=1)
    print("Wrote soil_moisture_comparison.ipynb")


# ══════════════════════════════════════════════════════════════════════════════
# wind_u_comparison.ipynb
# ══════════════════════════════════════════════════════════════════════════════

wu_header = """\
# wind_u sweep — comprehensive analysis

Constant u-wind sweep.  All runs: aerosols off; `swlspres=uflux` enforces exact mean wind.

| Key | u (m s⁻¹) | Status |
|-----|----------|--------|
| `wind_u_0p0`  | 0   | complete |
| `wind_u_2p5`  | 2.5 | complete |
| `wind_u_5p0`  | 5   | complete |
| `wind_u_7p5`  | 7.5 | complete |
| `wind_u_10p0` | 10  | 2stream complete; raytracer pending (job 50297781) |

> `u_10p0/raytracer` reps are skipped gracefully if not yet present on disk.

**Sections**
1. LWP overview
2. Thermodynamic structure
3. Surface energy balance
"""

wu_config = config_cell(
    group="wind_u",
    title="wind_u",
    param_unit=r"m s$^{-1}$",
    extra_note="# NOTE: wind_u_10p0/raytracer may be absent; RunSet skips missing reps automatically.",
)

wu_cells = [
    md(wu_header),
    code(IMPORTS),
    wu_config,
    md("---\n## Data loading"),
    code(DATA_LOAD),
    md("---\n## 1. LWP overview"),
    code(LWP_SUMMARY),
    md("---\n## 2. Thermodynamic structure"),
    code(THERMO_TIMEH),
    code(THERMO_PROFILES),
    md("---\n## 3. Surface energy balance"),
    code(SEB_SECTION),
    code(SEB_BO),
]

if "wind_u" in _TARGETS:
    with open(HERE / "wind_u_comparison.ipynb", "w") as f:
        json.dump(nb(wu_cells), f, indent=1)
    print("Wrote wind_u_comparison.ipynb")


# ══════════════════════════════════════════════════════════════════════════════
# wind_geo_comparison.ipynb
# ══════════════════════════════════════════════════════════════════════════════

wg_header = """\
# wind_geo sweep — comprehensive analysis

Geostrophic wind sweep (realistic Ekman shear via `swlspres=geo`).
All runs: aerosols off.

| Key | u_g (m s⁻¹) | Note |
|-----|------------|------|
| `wind_geo_0`    | 0   | reference (no_aerosols_zero_wind) |
| `wind_geo_2p5`  | 2.5 | |
| `wind_geo_5p0`  | 5   | |
| `wind_geo_7p5`  | 7.5 | |
| `wind_geo_10p0` | 10  | |

**Sections**
1. LWP overview
2. Thermodynamic structure
3. Surface energy balance
"""

wg_config = config_cell(
    group="wind_geo",
    title="wind_geo",
    param_unit=r"m s$^{-1}$",
)

wg_cells = [
    md(wg_header),
    code(IMPORTS),
    wg_config,
    md("---\n## Data loading"),
    code(DATA_LOAD),
    md("---\n## 1. LWP overview"),
    code(LWP_SUMMARY),
    md("---\n## 2. Thermodynamic structure"),
    code(THERMO_TIMEH),
    code(THERMO_PROFILES),
    md("---\n## 3. Surface energy balance"),
    code(SEB_SECTION),
    code(SEB_BO),
]

if "wind_geo" in _TARGETS:
    with open(HERE / "wind_geo_comparison.ipynb", "w") as f:
        json.dump(nb(wg_cells), f, indent=1)
    print("Wrote wind_geo_comparison.ipynb")

print("\nAll done.")
