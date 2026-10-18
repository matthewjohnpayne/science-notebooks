# orbunc

A small toolkit for exploring the uncertainty of minor-planet orbits and of their predicted positions. It is used by the orbital-uncertainty notebooks in this repository (see `docs/orbit_uncertainty/PLAN.md`).

```bash
pip install -e ./orbunc          # from the repository root
python -m pytest orbunc/tests    # tests needing the ephemerides are skipped if they are absent
```

Propagation uses [ASSIST](https://github.com/matthewholman/assist), which needs two JPL files (about 720 MB): `linux_p1550p2650.440` and `sb441-n16.bsp`. Set `ASSIST_DIR` to a folder that has them; otherwise `orbunc.ephem.get_ephem()` downloads them to `~/.orbunc` (or `$ORBUNC_DATA`).

Units: au, days, barycentric ICRF; time `t` is TDB days since J2000, as in ASSIST.

| Module | Contents |
|---|---|
| `ephem` | Ephemeris files; `propagate` (many test particles in one ASSIST run, batched by distance from the Earth; Earth-striking orbits return NaN); `body_state` |
| `frames` | Time scales, sky directions, ecliptic rotation, osculating elements, H-G magnitudes |
| `mpc` | MPC API access (observations in ADES, mpc_orb orbits, observatory codes), cached as gzip JSON; `Observations`; station positions with astropy; `orbit_state` (an mpc_orb orbit as a barycentric state and covariance) |
| `predict` | Light-time-corrected astrometric positions, residuals, chi-squared |
| `ranging` | Attributables, the admissible region, and valley-following systematic ranging with sampling |
| `fit` | Six-parameter least-squares fits and Gaussian sampling |
| `sky` | Sky positions from a station, offsets, the linearised sky covariance, orbit drawing, orbit classes |
