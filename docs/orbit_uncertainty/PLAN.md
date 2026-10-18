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
| 3 | Occultations (`notebooks/occultations`) | **Draft done** (2026-10-18) | Measured first: numbered MBAs with long arcs are known to ~3-40 km (formal), not ~1000 km; ~1000s of km applies to ~4-opposition arcs. Part A (Matthew: prefer purely synthetic, idealised observations): (500000)'s orbit as truth, simulated 0.3" history (4 or 26 oppositions), simulated single-chord occultations (1 km along-track, R/sqrt3 across) at the 2027 opposition, second 14 d or 1 yr later; linear (Kalman) updates (`orbunc.fit.update`). Long arc: sky-plane collapse, line of sight untouched; short arc: line of sight constrained too (line of variations). Part B: real occultation astrometry (ADES mode OCC) of (2000), (4000), (1000): optical-only orbits miss by 2-3x formal sigma (factor 2.5); (4000) Hipparchus stage by stage with formal and inflated priors. |
| 4 | Roman GBTDS: TNOs, vantage points | Planned (research) | Matthew: focus on TNOs and on how the spread depends on distance and semi-major axis. Colleagues' estimate: δa/a ~ 1e-3 (fractional, Matthew 2026-10-18), from simulated Roman bulge astrometry, probably over a few weeks rather than the full 72-day season (the idea is to fit partial-season data and predict later in the season). Reproduce this. Giovinazzi et al. 2021 find Δa/a ~ 1e-2 from one L2 epoch plus ground epochs. Survey parameters, Roman's orbit (Horizons −211) and the literature are in `roman_research_notes.md`. Likely paper candidate: no prior work found on Roman-derived TNO orbits for occultation prediction. |

## Conventions

These follow the repository's AGENTS.md:
- Notebooks are generated by builders in `tools/`, executed by `tools/execute.py`, and committed with outputs.
- MPC data are cached (gzip JSON) in each notebook's `data/`.
- ASSIST ephemerides go in `$ASSIST_DIR`, or are downloaded to `~/.orbunc`.
- The figure palette was validated for colour-blind safety: categorical #2a78d6, #eb6834, #1baf7a, with direct labels or legends.

## To do in `neocp_uncertainty` (Matthew, 2026-10-18)

- **Open Section 6.1 with a simple diagram of a tracklet on the sky:** a few positions moving across a star field over an hour, with the fitted rate of motion, before introducing the line of sight. This tells the reader what a "tracklet" is.
- **Bring back the whole-orbit "spray"** (old Figure 7b, orbits consistent with one tracklet), tied explicitly to the line-of-sight view (Figure 8a):
  - use the same sampled orbits and the same colouring by distance along the ray;
  - mark each orbit's crossing point on the ray, so each full orbit can be traced back to where it meets the line of sight;
  - for example, side-by-side panels with shared colours, or the full orbits drawn faintly behind the ray and arcs.
- **Cosmetic, Figure 8:** the legend in (c) overlaps some points, and (c)'s title crowds (a)'s x-axis label.

## Further directions (Matthew, 2026-10-18; not started)

These are possible additions to the NEO/NEOCP side of the project.

**(a) More "standard" examples.**
- Add typical NEOCP objects alongside, or instead of, the unusual 2024 YR4: an ordinary NEO, a main-belt object posted to the NEOCP, a slow distant object.
- They might be chosen systematically, for example from NEOCP postings, by digest2 score and discovery geometry.

**(b) When can the covariance of a designated short-arc orbit be trusted?**
- The question concerns orbits with a few tracklets, good enough to be designated and to enter `orbfit_results`.
- Figure 8 of `neocp_uncertainty` shows the Gaussian and ranging disagreeing after two tracklets.
- Is there a cheap diagnostic other than direct N-body sampling? Ideas to test (references to be checked before use):
  - nonlinearity indices, comparing first- and second-order terms of the orbit-to-prediction map;
  - comparing the Δχ² actually reached at ±kσ along the covariance's principal axes (especially the weakest direction, the line of variations) with the quadratic prediction;
  - sigma-point (unscented) propagation of 13 orbits against the linear map;
  - comparing the size of the uncertainty with the scale on which the predictions curve.
- Calibrate any such diagnostic against ranging or Monte Carlo results.

**(c) How often does reality fall inside the predicted envelopes?**
- Motivated by the NEOCP development about to start in mpc-software.
- Method: for many real objects, fit each stage of their early history (1, 2, 3 tracklets, ...) with each method (linear covariance, ranging, MCMC, ...). Predict their later positions, and compare with the positions later observed or with the final orbit.
- Measure the coverage: how often the truth falls within the 1, 2 and 3σ (or 68, 95, 99.7%) regions. Show it as calibration curves or probability-integral-transform histograms.
- Use observations by submission time (as in `impact_probability`), and expect formal uncertainties to be optimistic (see `occultations`, Part B).
- This is the large-sample version of Figure 8.
