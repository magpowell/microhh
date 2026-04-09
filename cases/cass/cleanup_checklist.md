# CASS LES Cleanup Checklist

One-time tasks. Delete this file when complete.

## Scratch archival (independent of repo cleanup)

| Experiment | Scratch path | HPSS path | Status |
|---|---|---|---|
| soil_moisture | (deleted) | `/home/m/mpowell/CASS_LES/soil_moisture/` | ARCHIVED + VERIFIED |
| mean_state_nudge | (deleted) | -- (re-derivable from no_aero_zero_wind 2stream) | DELETED, not archived |
| cs_veg | `experiments/cs_veg/` | `/home/m/mpowell/CASS_LES/cs_veg/` | TODO: htar + verify |
| wind_u | `experiments/wind_u/` | `/home/m/mpowell/CASS_LES/wind_u/` | TODO: htar + verify |
| analysis/timescale/ | `analysis/timescale/` | -- (deletable, 1.5 GB) | TODO: delete |

After HPSS verification: delete cs_veg and wind_u from scratch.

## Repo cleanup (can proceed immediately)

- [ ] Delete experiment dirs for archived experiments:
      `experiments/cs_veg/`, `experiments/soil_moisture/`,
      `experiments/mean_state_nudge/`, `experiments/wind_u/`
- [ ] Clean `cass_input.py`: remove `--theta-nudge`, `--nudge-thermo`; keep `--zero-winds`, `--wind-u`, `--geo-wind`; add `--rs-scale`
- [ ] Create `experiments/rs_scale/` scripts (setup, sbatch, submit, submit_debug)
- [ ] Create `experiments/wind_azi/` placeholder (README with design TODO)
- [ ] Update `README.md` to reflect only active experiments
- [ ] Commit: "Streamline project: archive old experiments, add rs_scale sweep"

## Analysis revamp (`analysis/`)

Full audit required — interactive session with user to triage what's critical vs exploratory.

- [ ] Inventory all notebooks and scripts in `analysis/`; classify as critical / exploratory / dead
- [ ] Refactor critical analysis into modular xarray-based code (current code is too verbose)
- [ ] Extract reusable utilities into `cass_analysis.py` (loading, alpha/gamma_s computation, etc.)
- [ ] Remove or archive notebooks that only reference archived experiments
- [ ] Ensure analysis pipeline generalizes cleanly to rs_scale, sw_scale, wind_azi, wind_geo
