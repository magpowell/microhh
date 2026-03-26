"""Generate cs_veg_comparison.ipynb, soil_moisture_comparison.ipynb, wind_u_comparison.ipynb."""

import json
from pathlib import Path

HERE = Path(__file__).parent


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
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from cass_analysis import (
    load_stats, load_stats_ensemble, load_3d_nc, load_xy_files,
    compute_normalized_cloud_root_profiles,
    lwp_integral, lwp_normalized_diff,
    seb_residual, evaporative_fraction,
    RunSet, _to_plottime,
    rho, cp, Lv,
)
from catalog import make_runset, list_group, ALL_EXPERIMENTS
"""


def config_cell(group, title, sweep_filter="", param_unit="", extra_note=""):
    lines = [
        f'SCRATCH        = Path("/pscratch/sd/m/mpowell/CASS_LES")\n',
        f'COMPOSITE_ROOT = SCRATCH / "analysis/cloud_root_composite"\n',
        f'N_REPS         = 4\n',
        f'LST_OFF        = 5.5   # t=0 → 05:30 LST\n',
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
        f'    t_local = s["t_local"] if isinstance(s, dict) else s\n',
        f'    hours = np.array([t.hour + t.minute / 60 for t in t_local])\n',
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
    nt_lwp  = min(s["qlqi_path"].shape[0] for s in reps_2s)
    if reps_rt:
        nt_lwp = min(nt_lwp, *(s["qlqi_path"].shape[0] for s in reps_rt))
    n_paired = min(len(reps_2s), len(reps_rt)) if reps_rt else len(reps_2s)

    lwp_1d_reps = np.stack([s["qlqi_path"][:nt_lwp] * 1e3 for s in reps_2s[:n_paired]])
    lwp_1d      = lwp_1d_reps.mean(axis=0)
    lwp_1d_std  = lwp_1d_reps.std(axis=0)

    if reps_rt:
        lwp_rt_reps = np.stack([s["qlqi_path"][:nt_lwp] * 1e3 for s in reps_rt[:n_paired]])
        lwp_rt      = lwp_rt_reps.mean(axis=0)
        lwp_rt_std  = lwp_rt_reps.std(axis=0)
        diff_reps   = lwp_rt_reps - lwp_1d_reps
        diff        = diff_reps.mean(axis=0)
        diff_std    = diff_reps.std(axis=0)
    else:
        lwp_rt = lwp_rt_std = diff = diff_std = diff_reps = None

    t_local_lwp = reps_2s[0]["t_local"][:nt_lwp]
    t_sec_lwp   = reps_2s[0]["t_sec"][:nt_lwp]
    dt_s        = float(t_sec_lwp[1] - t_sec_lwp[0]) if len(t_sec_lwp) > 1 else 60.0
    t_num_lwp   = _to_plottime(t_local_lwp)

    if diff is not None:
        cloudy_mask = (lwp_1d > 0.1) | (lwp_rt > 0.1)
        integral    = lwp_integral(diff, dt_s, cloudy_mask=cloudy_mask)
        int_std     = float(np.std(
            [lwp_integral(diff_reps[r], dt_s, cloudy_mask=cloudy_mask)
             for r in range(n_paired)], ddof=0))
    else:
        integral = int_std = None

    data[key] = dict(
        dirs_2s=dirs_2s, dirs_rt=dirs_rt,
        reps_2s=reps_2s, reps_rt=reps_rt,
        n_paired=n_paired,
        s2s_mean=s2s_mean, s2s_std=s2s_std,
        srt_mean=srt_mean, srt_std=srt_std,
        lwp_1d=lwp_1d, lwp_1d_std=lwp_1d_std,
        lwp_rt=lwp_rt, lwp_rt_std=lwp_rt_std,
        diff=diff, diff_std=diff_std,
        integral=integral, int_std=int_std,
        t_local_lwp=t_local_lwp, t_num_lwp=t_num_lwp, dt_s=dt_s,
    )
    print(f"  {key:25s}  2s={len(dirs_2s)}  rt={len(dirs_rt)}  "
          f"integral={integral:+.1f} g m⁻² h" if integral is not None
          else f"  {key:25s}  2s={len(dirs_2s)}  rt=0 (no raytracer)")

available = [k for k in data]
print(f"\\nLoaded {len(available)} sweep values: {available}")
"""

LWP_SECTION = """\
# ── LWP timeseries: absolute ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 3.5))

for key in available:
    d = data[key]
    c = COLORS[key]
    ax.plot(d["t_num_lwp"], d["lwp_1d"], color=c, ls="-",  lw=1.5)
    ax.fill_between(d["t_num_lwp"],
                    d["lwp_1d"] - d["lwp_1d_std"],
                    d["lwp_1d"] + d["lwp_1d_std"], color=c, alpha=0.10)
    if d["lwp_rt"] is not None:
        ax.plot(d["t_num_lwp"], d["lwp_rt"], color=c, ls="--", lw=1.5)
        ax.fill_between(d["t_num_lwp"],
                        d["lwp_rt"] - d["lwp_rt_std"],
                        d["lwp_rt"] + d["lwp_rt_std"], color=c, alpha=0.10)

ax.axhline(0, color="gray", lw=0.5)
ax.set_ylabel(r"LWP  (g m$^{-2}$)", fontsize=9)
_lh_c = [Line2D([0],[0], color=COLORS[k], lw=2, label=LABELS[k]) for k in available]
_lh_s = [Line2D([0],[0], color="k", ls="-",  lw=1.5, label="1D"),
          Line2D([0],[0], color="k", ls="--", lw=1.5, label="3D")]
ax.legend(handles=_lh_c + _lh_s, fontsize=7, ncol=2) #  + _lh_w
ax.xaxis_date()
fig.autofmt_xdate()
fig.suptitle(f"{SWEEP_GROUP} sweep:  Domain-mean LWP  (ensemble mean ±1σ)", fontsize=12)
plt.tight_layout()
_savefig(f"{SWEEP_GROUP}_lwp_timeseries.pdf", bbox_inches="tight")
plt.show()
"""

LWP_INTEGRAL_BAR = """\
# ── LWP summary: diff / normalised / integral ────────────────────────────────
keys_d = [k for k in available if data[k]["diff"] is not None]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
ax_diff, ax_norm, ax_int = axes

for key in keys_d:
    d   = data[key]
    c   = COLORS[key]
    lbl = LABELS[key]
    ax_diff.plot(d["t_num_lwp"], d["diff"], color=c, lw=1.5, label=lbl)
    ax_diff.fill_between(d["t_num_lwp"],
                         d["diff"] - d["diff_std"],
                         d["diff"] + d["diff_std"], color=c, alpha=0.15)
    norm     = d["diff"]     / np.maximum(d["lwp_1d"], 0.1) * 100
    norm_std = d["diff_std"] / np.maximum(d["lwp_1d"], 0.1) * 100
    ax_norm.plot(d["t_num_lwp"], norm, color=c, lw=1.5)
    ax_norm.fill_between(d["t_num_lwp"], norm - norm_std, norm + norm_std, color=c, alpha=0.15)

ax_diff.axhline(0, color="gray", lw=0.7, ls="--")
ax_diff.set_ylabel(r"3D − 1D LWP  (g m$^{-2}$)", fontsize=9)
ax_diff.set_title("(a) Mean difference")
ax_diff.legend(fontsize=8)
ax_diff.xaxis_date()

ax_norm.axhline(0, color="gray", lw=0.7, ls="--")
ax_norm.set_ylabel("(3D − 1D) / 1D  (%)", fontsize=9)
ax_norm.set_title("(b) Normalised difference")
ax_norm.xaxis_date()

integrals = [data[k]["integral"] for k in keys_d]
int_stds  = [data[k]["int_std"]  for k in keys_d]
colors_v  = [COLORS[k]           for k in keys_d]
x = np.arange(len(keys_d))
ax_int.bar(x, integrals, color=colors_v, edgecolor="k", lw=0.8, alpha=0.85,
           yerr=int_stds, capsize=5, error_kw=dict(lw=1.5))
ax_int.axhline(0, color="gray", lw=0.7)
ax_int.set_xticks(x)
ax_int.set_xticklabels([LABELS[k] for k in keys_d], rotation=15, ha="right", fontsize=9)
ax_int.set_xlabel(f"{SWEEP_GROUP}  ({PARAM_UNIT})")
ax_int.set_ylabel(r"$\int$(3D − 1D) LWP  (g m$^{-2}$ h)", fontsize=9)
ax_int.set_title("(c) Cloudy-period integral")

for ax in [ax_diff, ax_norm]:
    ax.figure.autofmt_xdate()

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
        dirs_ = d[f"dirs_{rt_key[:2].replace('ra','rt')}"] if rt_key == "raytracer" else d["dirs_2s"]
        # reload thermo directly
        _reps_t = []
        for _d in (d["dirs_rt"] if rt_key == "raytracer" else d["dirs_2s"]):
            ds = nc.Dataset(str(Path(_d) / "cass.default.0000000.nc"))
            thm = ds.groups["thermo"]
            _t  = np.array(ds.variables["time"][:])
            _z  = np.array(ds.variables["z"][:])
            _reps_t.append({
                "t": _t, "z": _z,
                "ql": np.array(thm.variables["ql"][:]) * 1e3,
                "thl": np.array(thm.variables["thl"][:]),
                "qt":  np.array(thm.variables["qt"][:]) * 1e3,
            })
            ds.close()
        if not _reps_t:
            continue
        nt_ = min(r["t"].shape[0] for r in _reps_t)
        avg = {"t": _reps_t[0]["t"][:nt_], "z": _reps_t[0]["z"]}
        for _k in ["ql", "thl", "qt"]:
            avg[_k] = np.stack([r[_k][:nt_] for r in _reps_t]).mean(axis=0)
        reps.append((rt_key, avg))
    thermo[key] = reps

# Plot ql time-height for each param, raytracer only (best available)
n_vals = len(available)
fig, axes = plt.subplots(1, n_vals, figsize=(4.5 * n_vals, 4.5), sharey=True)
if n_vals == 1:
    axes = [axes]
z_max = 4000

for ax, key in zip(axes, available):
    ql_data = None
    for rt_key, avg in thermo[key]:
        if rt_key == "raytracer":
            ql_data = avg
            break
    if ql_data is None and thermo[key]:
        _, ql_data = thermo[key][0]  # fall back to 2stream
    if ql_data is None:
        ax.set_title(LABELS[key]); continue

    z = ql_data["z"]
    z_mask = z <= z_max
    t_h = ql_data["t"] / 3600 + LST_OFF
    ql  = ql_data["ql"][:, z_mask]
    vmax = max(np.nanpercentile(ql[ql > 0], 99) if (ql > 0).any() else 0.01, 0.01)
    ax.pcolormesh(t_h, z[z_mask], ql.T, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xlabel("LST  (h)")
    ax.set_title(LABELS[key], fontsize=9)

axes[0].set_ylabel("z  (m)")
fig.suptitle(f"{SWEEP_GROUP}: q_l time-height (raytracer ensemble mean)")
fig.tight_layout()
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
            nt_ = sm["t_sec"].shape[0]
            t_  = _to_plottime(sm["t_local"][:nt_])
            ax.plot(t_, sm[flux][:nt_],
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

SEB_EF = """\
# ── Evaporative fraction + SEB residual ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
ax_ef, ax_res, ax_bar = axes  # ax_bar has param-value x-axis — do NOT sharex

param_vals_ef, ef_1d_vals, ef_3d_vals = [], [], []

for key in available:
    d   = data[key]
    c   = COLORS[key]
    lbl = LABELS[key]
    for rt_key, sm in [("2stream", d["s2s_mean"]), ("raytracer", d["srt_mean"])]:
        if sm is None:
            continue
        nt_ = sm["t_sec"].shape[0]
        t_  = _to_plottime(sm["t_local"][:nt_])
        ef  = evaporative_fraction({k: sm[k][:nt_] for k in sm})
        day = sm["Rnet"][:nt_] > 50
        ax_ef.plot(t_[day], ef[day], color=c,
                   ls="-" if rt_key == "2stream" else "--", lw=1.2)
        res = seb_residual({k: sm[k][:nt_] for k in sm})
        ax_res.plot(t_, res, color=c,
                    ls="-" if rt_key == "2stream" else "--", lw=1.2)
        if day.any():
            if rt_key == "2stream":
                ef_1d_vals.append((PARAMS[key], np.nanmean(ef[day])))
            else:
                ef_3d_vals.append((PARAMS[key], np.nanmean(ef[day])))

ax_ef.set_ylabel("EF  (–)")
ax_ef.set_title("Evaporative fraction  (daytime)")
ax_res.set_ylabel("SEB residual  (W m⁻²)")
ax_res.set_title("$R_{net} - (H+LE+G)$")
_lh_c = [Line2D([0],[0], color=COLORS[k], lw=2, label=LABELS[k]) for k in available]
_lh_s = [Line2D([0],[0], color="k", ls="-",  lw=1.5, label="1D"),
          Line2D([0],[0], color="k", ls="--", lw=1.5, label="3D")]
ax_res.legend(handles=_lh_c + _lh_s, fontsize=7)

# Bar: daytime-mean EF vs param
if ef_1d_vals:
    p1d, e1d = zip(*sorted(ef_1d_vals))
    ax_bar.plot(p1d, e1d, "o-", color="C0", lw=1.5, label="1D")
if ef_3d_vals:
    p3d, e3d = zip(*sorted(ef_3d_vals))
    ax_bar.plot(p3d, e3d, "s--", color="C1", lw=1.5, label="3D")
ax_bar.set_xlabel(f"{SWEEP_GROUP}  ({PARAM_UNIT})")
ax_bar.set_ylabel("Daytime-mean EF")
ax_bar.set_title("EF vs sweep param")
ax_bar.legend(fontsize=8)

for ax in [ax_ef, ax_res]:
    ax.xaxis_date()

fig.autofmt_xdate()
fig.tight_layout()
_savefig(f"{SWEEP_GROUP}_seb_ef.pdf", bbox_inches="tight")
plt.show()
"""

ENT_HELPERS = """\
# ── Entrainment helper functions ──────────────────────────────────────────────
FLUX_ZMIN = 500.0
FLUX_ZMAX = 3500.0
JUMP_DZ   = 250.0


def _find_bl_top(thv_flux_prof, zh):
    mask = (zh >= FLUX_ZMIN) & (zh <= FLUX_ZMAX)
    if mask.sum() == 0:
        return np.nan, np.nan
    idx_rel = np.argmin(thv_flux_prof[mask])
    return float(zh[mask][idx_rel]), float(thv_flux_prof[mask][idx_rel])


def _scalar_jump(mean_prof, z, z_top, dz=JUMP_DZ):
    above = mean_prof[(z >= z_top) & (z <= z_top + dz)]
    below = mean_prof[(z >= z_top - dz) & (z < z_top)]
    if len(above) == 0 or len(below) == 0:
        return np.nan
    return float(above.mean() - below.mean())


def _method_flux_jump(s):
    # we = -<w'thv'>_top / Delta_thv  (Lilly 1968)
    t, zh, z = s["t_sec"], s["zh"], s["z"]
    thv_flux_all, mean_all = s["thv_flux"], s["thv"]
    nt = len(t)
    we, z_top_ts, flux_top, delta = [np.full(nt, np.nan) for _ in range(4)]
    for i in range(nt):
        zt, _ = _find_bl_top(thv_flux_all[i], zh)
        if not np.isfinite(zt):
            continue
        z_top_ts[i] = zt
        f = float(np.interp(zt, zh, thv_flux_all[i]))
        flux_top[i] = f
        d = _scalar_jump(mean_all[i], z, zt)
        delta[i] = d
        if np.isfinite(d) and np.abs(d) > 1e-6:
            we[i] = -f / d
    return t, we * 1e3, z_top_ts, flux_top, delta


def _load_ext(run_dir):
    s = load_stats(run_dir)
    ds = nc.Dataset(str(Path(run_dir) / "cass.default.0000000.nc"))
    thm = ds.groups["thermo"]
    s["thv"]      = np.array(thm.variables["thv"][:])
    s["thv_flux"] = np.array(thm.variables["thv_flux"][:])
    s["zh"]       = np.array(ds.variables["zh"][:])
    ds.close()
    return s


def _load_ext_ensemble(rep_dirs):
    all_s = [_load_ext(d) for d in rep_dirs]
    nt_min = min(s["t_sec"].shape[0] for s in all_s)
    SPATIAL, TIMECOPY = {"z", "zh"}, {"t_local"}
    out = {}
    for key in all_s[0]:
        if key in SPATIAL:
            out[key] = all_s[0][key]
        elif key in TIMECOPY:
            out[key] = all_s[0][key][:nt_min]
        else:
            try:
                out[key] = np.stack([s[key][:nt_min] for s in all_s]).mean(axis=0)
            except Exception:
                out[key] = all_s[0][key]
    return out


# ── Compute entrainment for each sweep value + RT type ────────────────────────
ent_results = {}   # key -> {"2stream": {...}, "raytracer": {...}}
for key in available:
    d = data[key]
    ent_results[key] = {}
    for rt_key, dirs_ in [("2stream", d["dirs_2s"]), ("raytracer", d["dirs_rt"])]:
        if not dirs_:
            continue
        print(f"  {key}/{rt_key}: loading {len(dirs_)} reps …", end=" ", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_ens = _load_ext_ensemble(dirs_)
        t, we, z_top, flux_top, dthv = _method_flux_jump(s_ens)
        msk = lst_mask(s_ens)
        print(f"we = {np.nanmean(we[msk]):.1f} mm/s")
        ent_results[key][rt_key] = dict(
            t=t, t_local=s_ens["t_local"], we=we, z_top=z_top)
"""

ENT_PLOTS = """\
# ── Entrainment timeseries: z_top and we ──────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
first = True
for key in available:
    c = COLORS[key]
    for rt_key, r in ent_results[key].items():
        ls  = "-" if rt_key == "2stream" else "--"
        t_num_e = _to_plottime(r["t_local"])
        axes[0].plot(t_num_e, r["z_top"], color=c, ls=ls, lw=1.2)
        axes[1].plot(t_num_e, r["we"],    color=c, ls=ls, lw=1.2)
        if first:
            msk = lst_mask(r["t_local"])
            axvspan_window(axes[0], r["t_local"], msk, color="gold", alpha=0.12)
            axvspan_window(axes[1], r["t_local"], msk, color="gold", alpha=0.12)
            first = False

axes[0].set_ylabel("z_top  (m)")
axes[0].set_title("BL-top height  (min $\\\\langle w\'\\\\theta_v\'\\\\rangle$)")
_lh_c = [Line2D([0],[0], color=COLORS[k], lw=2, label=LABELS[k]) for k in available]
_lh_s = [Line2D([0],[0], color="k", ls="-",  lw=1.5, label="1D"),
          Line2D([0],[0], color="k", ls="--", lw=1.5, label="3D")]
_lh_w = [Line2D([0],[0], color="gold", lw=8, alpha=0.5, label="analysis window")]
axes[0].legend(handles=_lh_c + _lh_s + _lh_w, fontsize=7, ncol=2)
axes[0].set_ylim(0, 3500)
axes[1].axhline(0, color="k", lw=0.5, ls=":")
axes[1].set_ylabel(r"$w_e$  (mm s$^{-1}$)")
axes[1].set_title(r"$-\\langle w\'\\theta_v\'\\rangle_\\mathrm{top}/\\Delta\\theta_v$")
axes[1].xaxis_date()
fig.autofmt_xdate()
fig.tight_layout()
_savefig(f"{SWEEP_GROUP}_entrainment_ts.pdf", bbox_inches="tight")
plt.show()
"""

ENT_BAR = """\
# ── Window-mean entrainment summary: we vs param value ───────────────────────
param_vals_1d, we_1d_vals, we_1d_std = [], [], []
param_vals_3d, we_3d_vals, we_3d_std = [], [], []

for key in available:
    for rt_key, r in ent_results[key].items():
        msk = lst_mask(r["t_local"])[:len(r["we"])]
        val = np.nanmean(r["we"][msk])
        std = np.nanstd(r["we"][msk])
        if rt_key == "2stream":
            param_vals_1d.append(PARAMS[key])
            we_1d_vals.append(val); we_1d_std.append(std)
        else:
            param_vals_3d.append(PARAMS[key])
            we_3d_vals.append(val); we_3d_std.append(std)

fig, ax = plt.subplots(figsize=(7, 4))
if we_1d_vals:
    ax.errorbar(param_vals_1d, we_1d_vals, we_1d_std, fmt="o-",
                color="C0", lw=1.5, capsize=4, label="1D (2stream)")
if we_3d_vals:
    ax.errorbar(param_vals_3d, we_3d_vals, we_3d_std, fmt="s--",
                color="C1", lw=1.5, capsize=4, label="3D (raytracer)")
ax.set_xlabel(f"{SWEEP_GROUP}  ({PARAM_UNIT})")
ax.set_ylabel(r"$\\langle w_e \\rangle$  (mm s$^{-1}$)  [LST 12:30–19:00]")
ax.set_title(f"{SWEEP_GROUP}: window-mean entrainment rate")
ax.axhline(0, color="k", lw=0.5)
ax.legend()
fig.tight_layout()
_savefig(f"{SWEEP_GROUP}_entrainment_bar.pdf", bbox_inches="tight")
plt.show()
"""


def cloud_root_section(group, sweep_keys_expr, expt_subpath_expr):
    """
    group: e.g. "cs_veg"
    sweep_keys_expr: Python expression to get list of keys (subset to include)
    expt_subpath_expr: lambda key -> relative path under SCRATCH/experiments/, as a Python f-string
    """
    return f"""\
# ── Cloud-root turbulent flux profiles ───────────────────────────────────────
# Loads from cloud_root_composite cache built by cloud_roots/submit_cloud_root_pipeline.sh.
# If 3D composite data are missing for a case, that case is silently skipped.

_RHO_CP = rho * cp
_RHO_LV = rho * Lv * 1e-3   # g/kg m/s → W m⁻²
LST_MIN_CR, LST_MAX_CR = 10.5, 17.5

CASES_CR = []
for _key in {sweep_keys_expr}:
    _meta = ALL_EXPERIMENTS[_key]
    _rs = make_runset(_key, n_reps=N_REPS)
    for _rt, _ls in [("2stream", "-"), ("raytracer", "--")]:
        _dirs = _rs.dirs.get(_rt, [])
        if not _dirs:
            continue
        _run_root  = str(_dirs[0].parent)   # .../subdir/2stream  (parent of rep_XX)
        _comp_expt = str(Path(_run_root).parent.relative_to(SCRATCH / "experiments"))
        CASES_CR.append(dict(
            key=f"{{_key}}/{{_rt}}",
            label=_meta["label"] + (" 1D" if _rt == "2stream" else " 3D"),
            color=_meta["color"], ls=_ls,
            comp_expt=_comp_expt,
            comp_rt=_rt,
            run_root=_run_root,
        ))

def _compute_and_cache_profiles(case, n_reps=N_REPS, lst_min=LST_MIN_CR, lst_max=LST_MAX_CR):
    run_root = Path(case["run_root"])
    H_dom_reps, H_root_reps, H_free_reps   = [], [], []
    LE_dom_reps, LE_root_reps, LE_free_reps = [], [], []
    cloud_frac_reps = []
    zeta = None
    for i in range(1, n_reps + 1):
        rep_dir = run_root / f"rep_{{i:02d}}"
        cache = (COMPOSITE_ROOT / case["comp_expt"] / case["comp_rt"]
                 / f"rep_{{i:02d}}" / "raw_flux_profile_cache.nc")
        # Invalidate stale cache
        if cache.exists():
            with xr.open_dataset(str(cache)) as _chk:
                if "f_free_thl" not in _chk or "t_hours" not in _chk:
                    print(f"  {{case['key']}} rep_{{i:02d}}: cache outdated, recomputing")
                    cache.unlink()
        if not cache.exists():
            if not (rep_dir / "thl.nc").exists():
                print(f"  SKIP {{case['key']}} rep_{{i:02d}}: thl.nc not found")
                continue
            print(f"  {{case['key']}} rep_{{i:02d}}: computing from 3D fields ...", flush=True)
            s     = load_stats(rep_dir)
            ds_3d = load_3d_nc(rep_dir, variables=["thl", "qt", "ql", "w"])
            ds_xy = load_xy_files(rep_dir, variables=["qlqi_path"])
            res   = compute_normalized_cloud_root_profiles(ds_3d, ds_xy, s)
            ds_3d.close()
            if res["f_cloud_thl"].shape[0] == 0:
                print(f"    WARNING: no valid timesteps, skipping")
                continue
            cache.parent.mkdir(parents=True, exist_ok=True)
            xr.Dataset({{
                "f_cloud_thl":  xr.DataArray(res["f_cloud_thl"],  dims=["time", "zeta"]),
                "f_cloud_qv":   xr.DataArray(res["f_cloud_qv"],   dims=["time", "zeta"]),
                "f_free_thl":   xr.DataArray(res["f_free_thl"],   dims=["time", "zeta"]),
                "f_free_qv":    xr.DataArray(res["f_free_qv"],    dims=["time", "zeta"]),
                "f_domain_thl": xr.DataArray(res["f_domain_thl"], dims=["time", "zeta"]),
                "f_domain_qv":  xr.DataArray(res["f_domain_qv"],  dims=["time", "zeta"]),
                "z_sl":         xr.DataArray(res["z_sl"],         dims=["time"]),
                "cloud_frac":   xr.DataArray(res["cloud_frac"],   dims=["time"]),
                "t_hours":      xr.DataArray(res["t_hours"],      dims=["time"]),
            }}, coords={{"zeta": res["zeta"]}}).to_netcdf(str(cache))
            print(f"    cached ({{res['f_cloud_thl'].shape[0]}} timesteps)")
        else:
            print(f"  {{case['key']}} rep_{{i:02d}}: cache hit")
        ds = xr.open_dataset(str(cache))
        zeta   = ds["zeta"].values
        t_hrs  = ds["t_hours"].values
        mask_t = (t_hrs >= lst_min) & (t_hrs <= lst_max)
        if mask_t.sum() == 0:
            ds.close(); continue
        import warnings as _w
        with _w.catch_warnings():
            _w.filterwarnings("ignore", category=RuntimeWarning)
            H_dom_reps.append( np.nanmean(ds["f_domain_thl"].values[mask_t] * _RHO_CP, axis=0))
            H_root_reps.append(np.nanmean(ds["f_cloud_thl"].values[mask_t]  * _RHO_CP, axis=0))
            H_free_reps.append(np.nanmean(ds["f_free_thl"].values[mask_t]   * _RHO_CP, axis=0))
            LE_dom_reps.append( np.nanmean(ds["f_domain_qv"].values[mask_t] * _RHO_LV, axis=0))
            LE_root_reps.append(np.nanmean(ds["f_cloud_qv"].values[mask_t]  * _RHO_LV, axis=0))
            LE_free_reps.append(np.nanmean(ds["f_free_qv"].values[mask_t]   * _RHO_LV, axis=0))
            cloud_frac_reps.append(float(np.nanmean(ds["cloud_frac"].values[mask_t])))
        ds.close()
    n = len(H_dom_reps)
    def _s(lst): return np.stack(lst) if lst else np.empty((0,))
    return (dict(zeta=zeta, n_reps=n,
                 H_dom=_s(H_dom_reps),  H_root=_s(H_root_reps),  H_free=_s(H_free_reps),
                 LE_dom=_s(LE_dom_reps), LE_root=_s(LE_root_reps), LE_free=_s(LE_free_reps)),
            np.array(cloud_frac_reps))

abs_profiles, cloud_fracs = {{}}, {{}}
for _case in CASES_CR:
    print(f"\\n{{_case['label']}}")
    _ap, _cf = _compute_and_cache_profiles(_case)
    abs_profiles[_case["key"]] = _ap
    cloud_fracs[_case["key"]]  = _cf
    _n = _ap["n_reps"]
    if _n > 0:
        _sd = _cf.std(ddof=1) if _n > 1 else 0.0
        print(f"  -> {{_n}} reps  alpha = {{_cf.mean():.3f}} +/- {{_sd:.3f}}")
    else:
        print(f"  NO COMPOSITE DATA")

cases_with_data = [c for c in CASES_CR if abs_profiles[c["key"]]["n_reps"] > 0]
print(f"\\n{{len(cases_with_data)}} / {{len(CASES_CR)}} cases have composite data")
"""


CLOUD_ROOT_PLOT = """\
# ── Absolute H and LE flux profiles: cloud-root vs cloud-free ────────────────
if not cases_with_data:
    print("No cloud-root composite data available yet — run submit_cloud_root_pipeline.sh")
else:
    CONDS = [("H_root", "LE_root", "cloud-root", "--"),
             ("H_free", "LE_free", "cloud-free",  "-")]

    RT_ROWS = [("3D", "--"), ("1D", "-")]   # (row title, case["ls"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, (rt_label, rt_ls) in enumerate(RT_ROWS):
        rt_cases = [c for c in cases_with_data if c["ls"] == rt_ls]
        for col, var in enumerate(["H", "LE"]):
            ax = axes[row, col]
            ax.axhline(1.0, color="gray", ls="--", lw=0.8)
            ax.axvline(0,   color="gray", ls=":",  lw=0.5)
            for case in rt_cases:
                ap   = abs_profiles[case["key"]]
                zeta = ap["zeta"]
                for h_key, le_key, lbl, ls_c in CONDS:
                    arr = ap[h_key if var == "H" else le_key]
                    mu  = np.nanmean(arr, axis=0)
                    sd  = np.nanstd(arr, axis=0, ddof=1) if ap["n_reps"] > 1 else np.zeros_like(mu)
                    ax.plot(mu, zeta, color=case["color"], ls=ls_c, lw=1.5)
                    ax.fill_betweenx(zeta, mu-sd, mu+sd, alpha=0.10, color=case["color"])
            ax.set_ylim(0, 1)
            ax.set_title(f"{rt_label} — {var}  (W m⁻², resolved)")
            if col == 0:
                ax.set_ylabel("z / z_sl")
            if row == 1:
                ax.set_xlabel(f"{var}  (W m⁻²)")

    _lh_cond = [Line2D([0],[0], color="k", ls="-",  lw=1.5, label="cloud-free"),
                Line2D([0],[0], color="k", ls="--", lw=1.5, label="cloud-root")]
    _seen_cols = set()
    _lh_params = []
    for _c in cases_with_data:
        if _c["color"] not in _seen_cols:
            _lbl = _c["label"].replace(" 1D", "").replace(" 3D", "")
            _lh_params.append(Line2D([0],[0], color=_c["color"], lw=2, label=_lbl))
            _seen_cols.add(_c["color"])
    axes[1,1].legend(handles=_lh_cond + _lh_params, fontsize=7, loc="lower right")
    fig.suptitle(f"{SWEEP_GROUP}: cloud-root vs cloud-free resolved turbulent flux profiles")
    fig.tight_layout()
    _savefig(f"{SWEEP_GROUP}_cloudroot_profiles.pdf", bbox_inches="tight")
    plt.show()
"""


CLOUD_ROOT_COMP_DIFF = """# ── 3D − 1D cloud-root composite difference across sweep ───────────────────────
# Rows: w’θ_l’ and w’q_v’; Cols: one per sweep value; shared colour scale per row.
# Requires: cloud_root_composite cache (events_{orient}.nc per rep)
import sys as _sys
_sys.path.insert(0, str(Path.cwd() / "cloud_roots"))
from cloud_root_composite_average import average_reps

import warnings as _warnings
print("Loading 2D composites ...")
_comp2d = {}
for _case in CASES_CR:
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", category=RuntimeWarning)
        _comp2d[_case["key"]] = average_reps(
            COMPOSITE_ROOT, _case["comp_expt"], _case["comp_rt"],
            n_reps=N_REPS, verbose=False,
        )
    _tag = ", ".join(
        f"{o}:{_comp2d[_case['key']][o].attrs['n_reps']}reps"
        for o in ("xz", "yz") if o in _comp2d[_case["key"]]
    )
    print(f"  {_case['label']}: {_tag}")

# Build 3D/1D pairs per sweep key and orientation
_pairs = {}   # sweep_key -> {orient: {"1D": ds, "3D": ds}}
for _k, _ in sweep:
    _k_2s = f"{_k}/2stream"
    _k_rt = f"{_k}/raytracer"
    if _k_2s not in _comp2d or _k_rt not in _comp2d:
        continue
    _pairs[_k] = {}
    for _o in ("xz", "yz"):
        if _o in _comp2d[_k_2s] and _o in _comp2d[_k_rt]:
            _pairs[_k][_o] = {"1D": _comp2d[_k_2s][_o], "3D": _comp2d[_k_rt][_o]}

_VARS_DIFF = [
    ("w_thl_mean", "RdBu_r", "3D−1D  Δw’θₗ’  (K m s⁻¹)"),
    ("w_qv_mean",  "BrBG",   "3D−1D  Δw’q_v’  (g kg⁻¹ m s⁻¹)"),
]

for orient in ("xz", "yz"):
    _keys = [k for k, _ in sweep if k in _pairs and orient in _pairs[k]]
    if not _keys:
        print(f"No {orient} composite diff data available — skipping")
        continue
    n_cols = len(_keys)
    hl = "x/L" if orient == "xz" else "y/L"

    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 7), squeeze=False,
                                constrained_layout=True)
    for row, (var, cmap, clabel) in enumerate(_VARS_DIFF):
        _all_diffs = np.concatenate([
            (_pairs[k][orient]["3D"][var].values -
             _pairs[k][orient]["1D"][var].values).ravel()
            for k in _keys
        ])
        vmax = float(np.nanpercentile(np.abs(_all_diffs), 98))

        _pcms = []
        for col, k in enumerate(_keys):
            ax   = axes[row, col]
            ds3  = _pairs[k][orient]["3D"]
            ds1  = _pairs[k][orient]["1D"]
            diff = ds3[var].values - ds1[var].values
            pcm  = ax.pcolormesh(ds3.xL.values, ds3.z_nd.values, diff,
                                  cmap=cmap, shading="auto", vmin=-vmax, vmax=vmax)
            _pcms.append(pcm)
            ax.axvline(-0.5, color="0.4", lw=0.8, ls="--")
            ax.axvline( 0.5, color="0.4", lw=0.8, ls="--")
            ax.axhline( 1.0, color="0.4", lw=0.8, ls=":")
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
            if col == 0:
                ax.set_ylabel("z / z_sl")
            if row == 0:
                ax.set_title(f"{LABELS[k]}")
            if row == 1:
                ax.set_xlabel(hl)
        fig.colorbar(_pcms[-1], ax=axes[row, :], location="right",
                     shrink=0.8, label=clabel)

    fig.suptitle(f"{SWEEP_GROUP}: 3D − 1D cloud-root composite  ({orient})")
    _savefig(f"{SWEEP_GROUP}_cloudroot_composite_diff_{orient}.pdf", bbox_inches="tight")
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
4. Entrainment
5. Cloud-root turbulent structure
"""

cs_veg_config = config_cell(
    group="cs_veg",
    title="cs_veg",
    sweep_filter="sweep = _all_sweep[:3]   # drop two highest (no clouds develop)",
    param_unit=r"J m$^{-2}$ K$^{-1}$",
)

cs_veg_cr = cloud_root_section(
    group="cs_veg",
    sweep_keys_expr="[k for k in available]",
    expt_subpath_expr="cs_veg/{key}",
)

cs_veg_cells = [
    md(cs_veg_header),
    code(IMPORTS),
    cs_veg_config,
    md("---\n## Data loading"),
    code(DATA_LOAD),
    md("---\n## 1. LWP overview"),
    code(LWP_SECTION),
    code(LWP_INTEGRAL_BAR),
    md("---\n## 2. Thermodynamic structure"),
    code(THERMO_TIMEH),
    code(THERMO_PROFILES),
    md("---\n## 3. Surface energy balance"),
    code(SEB_SECTION),
    code(SEB_EF),
    md("---\n## 4. Entrainment\n\nMethod: θ_v flux-jump, `wₑ = −⟨w′θ_v′⟩_top / Δθ_v`"),
    code(ENT_HELPERS),
    code(ENT_PLOTS),
    code(ENT_BAR),
    md("---\n## 5. Cloud-root turbulent structure\n\nLoads from `cloud_root_composite` cache. Skips missing cases."),
    code(cs_veg_cr),
    code(CLOUD_ROOT_PLOT),
    code(CLOUD_ROOT_COMP_DIFF),
]

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
4. Entrainment
5. Cloud-root turbulent structure
"""

sm_config = config_cell(
    group="soil_moisture",
    title="soil_moisture",
    param_unit=r"m$^3$ m$^{-3}$",
)

sm_cr = cloud_root_section(
    group="soil_moisture",
    sweep_keys_expr="[k for k in available]",
    expt_subpath_expr="soil_moisture/{key}",
)

sm_cells = [
    md(sm_header),
    code(IMPORTS),
    sm_config,
    md("---\n## Data loading"),
    code(DATA_LOAD),
    md("---\n## 1. LWP overview"),
    code(LWP_SECTION),
    code(LWP_INTEGRAL_BAR),
    md("---\n## 2. Thermodynamic structure"),
    code(THERMO_TIMEH),
    code(THERMO_PROFILES),
    md("---\n## 3. Surface energy balance"),
    code(SEB_SECTION),
    code(SEB_EF),
    md("---\n## 4. Entrainment\n\nMethod: θ_v flux-jump, `wₑ = −⟨w′θ_v′⟩_top / Δθ_v`"),
    code(ENT_HELPERS),
    code(ENT_PLOTS),
    code(ENT_BAR),
    md("---\n## 5. Cloud-root turbulent structure\n\nLoads from `cloud_root_composite` cache. Skips missing cases."),
    code(sm_cr),
    code(CLOUD_ROOT_PLOT),
    code(CLOUD_ROOT_COMP_DIFF),
]

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
4. Entrainment
5. Cloud-root turbulent structure
"""

wu_config = config_cell(
    group="wind_u",
    title="wind_u",
    param_unit=r"m s$^{-1}$",
    extra_note="# NOTE: wind_u_10p0/raytracer may be absent; RunSet skips missing reps automatically.",
)

wu_cr = cloud_root_section(
    group="wind_u",
    sweep_keys_expr="[k for k in available]",
    expt_subpath_expr="wind_u/{key}",
)

wu_cells = [
    md(wu_header),
    code(IMPORTS),
    wu_config,
    md("---\n## Data loading"),
    code(DATA_LOAD),
    md("---\n## 1. LWP overview"),
    code(LWP_SECTION),
    code(LWP_INTEGRAL_BAR),
    md("---\n## 2. Thermodynamic structure"),
    code(THERMO_TIMEH),
    code(THERMO_PROFILES),
    md("---\n## 3. Surface energy balance"),
    code(SEB_SECTION),
    code(SEB_EF),
    md("---\n## 4. Entrainment\n\nMethod: θ_v flux-jump, `wₑ = −⟨w′θ_v′⟩_top / Δθ_v`"),
    code(ENT_HELPERS),
    code(ENT_PLOTS),
    code(ENT_BAR),
    md("---\n## 5. Cloud-root turbulent structure\n\nLoads from `cloud_root_composite` cache. Skips missing cases."),
    code(wu_cr),
    code(CLOUD_ROOT_PLOT),
    code(CLOUD_ROOT_COMP_DIFF),
]

with open(HERE / "wind_u_comparison.ipynb", "w") as f:
    json.dump(nb(wu_cells), f, indent=1)
print("Wrote wind_u_comparison.ipynb")

print("\nAll done.")
