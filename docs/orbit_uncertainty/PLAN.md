# Orbital uncertainty: plan of the notebook series

A series of notebooks on how the uncertainty of a minor planet's orbit, and of its position as seen by an observer, grows, shrinks and changes shape, and on how it looks from different vantage points. Started 2026-10-18 on the `orbit-uncertainty` branch. It sits between research and explanation: it may lead to a paper (most likely from scenario 4), and otherwise serves as explainers here and on matthewjohnpayne.github.io/explore/. It may move to `generic_mpcsoftware_dev/mpc-research/` later; the toolkit is self-contained (`orbunc/`) to make that easy.

Unlike the other notebooks in this repository, these are not each tied to one of Matthew's papers.

## Unifying idea

Each scenario has the same uncertainty region in six-dimensional orbit space. What changes is where it is projected:

| Projection | Scenario |
|---|---|
| The observer's sky (variant orbits; on-sky ellipse or banana) | 5, 4 |
| The target plane at an Earth encounter (b-plane) | 1 |
| The Earth's surface (impact corridor) | 2 |
| The occultation fundamental plane (perpendicular to the star) | 3, 4 |

The hypothesis to test in scenario 4 is the following. Angular uncertainties seen from Roman (at Sun–Earth L2) and from the ground differ greatly, because the poorly known distance projects through the roughly 1.5e6 km baseline. The occultation shadow-plane uncertainty is the same for every observer, though; the vantage point only selects which star and which time give an event.

## Toolkit: `orbunc/`

ASSIST and REBOUND propagation, MPC API data (ADES observations, mpc_orb orbits, observatory codes), and astropy station positions. On top of these:

- attributables and the admissible region (Milani et al. 2004);
- systematic ranging that follows the chi-squared valley (Farnocchia et al. 2015 style);
- least-squares fits, Gaussian sampling, and sky projections.

**Validated:**
- The MPC orbits of 2024 YR4 and (10606) Crocco reproduce the real astrometry, with median |O−C| of about 0.07–0.12″.
- `orbunc/tests` (11 tests).
- Ranging and the least-squares Gaussian agree wherever the latter should be valid.

**Lessons so far:**
- ASSIST leaves IAS15 in "global" adaptive mode. Near the Earth this is about 1e5 times slower than `prs23`, so `prs23` is now set.
- Orbits that would hit the Earth must be filtered out before integration.
- Reported ADES uncertainties can be too small: a D29 tracklet reported at 0.07″ was offset about 0.1″ from the true orbit. Use a floor (0.2″ for now), or a proper weighting scheme (Vereš et al. 2017).
- digest2's trial orbits (from `collect_orbits=True`) carry only (q, e, i, H, d, angle), not full states, and are a census rather than a posterior. Ranging supplies the variant orbits; digest2 is used for scores and comparison.

layup (github.com/Smithsonian/layup) is optional, for cross-checks. It must be built from source; the 0.0.1 release on PyPI is not this package.

## Scenarios

| # | Notebook | Status | Notes |
|---|---|---|---|
| 5 | NEOCP: orbits from 1, 2, 3 tracklets (`notebooks/neocp_uncertainty`) | **Draft done** (2026-10-18) | 2024 YR4 (real discovery tracklets) and (10606) Crocco. Admissible region, systematic ranging, diurnal-parallax distance from one tracklet, digest2, orbit spray, sky predictions, prior dependence, Gaussian vs ranging. |
| 1 | Near-miss NEO: impact probability rises then falls (`notebooks/impact_probability`) | **Draft done** (2026-10-18) | 2024 YR4. Fits use the observations *submitted* before each cut-off date (submission times decoded from the obsid; JWST excluded), with a 0.1" floor (0.05" and 0.2" as a sensitivity test), Vereš et al. (2017) nightly sqrt(N/4) down-weighting, and Carpino et al. (2003) outlier rejection (chi2 8/7). Each fit is projected onto the 2032-12-22 b-plane by Monte Carlo through ASSIST (`orbunc.encounter`; Earth-striking samples are removed during integration) and by a linear map; a 2-D toy model comes first. Compared qualitatively with NASA's published values: >1% (Jan 29), 2.3% (Feb 7), 3.1% (Feb 18, the peak), 1.5% (Feb 19), 0.004% (Feb 24). Truth: closest approach 2032-12-22 08:24 TDB at 0.00185 au. |
| 2 | Impactor: the evolving impact corridor | Planned | 2008 TC3, 2023 CX1 or 2024 BX1. The corridor on the globe narrows as observations arrive. Check against the known fall sites. Needs propagation to impact (`propagate` currently returns NaN for Earth-striking orbits, by design; add an impact-time option). |
| 3 | Occultations: 0, 1, 2+ events | Planned | Matthew (2026-10-18): a numbered main-belt asteroid with no occultation data yet; his guess is a Cartesian position uncertainty of ~1000 km (to be measured). Picture to test: one ground-based occultation collapses the uncertainty in the plane of the sky at that epoch but leaves the topocentric radial direction untouched; a second, well-separated event collapses it further. Also show the along-track (timing) versus across-track (about one radius for a single chord) asymmetry, and the regrowth of along-track uncertainty from the semi-major axis uncertainty between events. |
| 4 | Roman GBTDS: TNOs, vantage points | Planned (research) | Matthew: focus on TNOs and on how the spread depends on distance and semi-major axis. Colleagues' estimate: δa/a ~ 1e-3 (fractional, Matthew 2026-10-18), from simulated Roman bulge astrometry, probably over a few weeks rather than the full 72-day season (the idea is to fit partial-season data and predict later in the season). Reproduce this. Giovinazzi et al. 2021 find Δa/a ~ 1e-2 from one L2 epoch plus ground epochs. Survey parameters, Roman's orbit (Horizons −211) and the literature are in `roman_research_notes.md`. Likely paper candidate: no prior work found on Roman-derived TNO orbits for occultation prediction. |

## Conventions

These follow the repository's AGENTS.md:
- Notebooks are generated by builders in `tools/`, executed by `tools/execute.py`, and committed with outputs.
- MPC data are cached (gzip JSON) in each notebook's `data/`.
- ASSIST ephemerides go in `$ASSIST_DIR`, or are downloaded to `~/.orbunc`.
- The figure palette was validated for colour-blind safety: categorical #2a78d6, #eb6834, #1baf7a, with direct labels or legends.
