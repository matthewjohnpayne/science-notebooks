#!/usr/bin/env python3
"""Build notebooks/roman_tno/roman_tno.ipynb from the cells below.

The notebook is generated so that its prose and code can be reviewed as one
plain-text file. Edit here, then rebuild and execute:

    python tools/build_roman_tno.py
    python tools/execute.py notebooks/roman_tno/roman_tno.ipynb
"""
import pathlib

import nbformat as nbf

OUT = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "roman_tno" / "roman_tno.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Seen from Roman, Seen from the Ground: Orbits of Distant Objects from the Galactic Bulge Survey

*Matthew Payne*

From 2027, the Nancy Grace Roman Space Telescope will stare at a few small fields towards the centre of the Milky Way for about 70 days at a time, imaging each one every 12 minutes, to find planets by gravitational microlensing: the Galactic Bulge Time Domain Survey. The fields lie only 5 to 6.5 degrees from the ecliptic, so distant Solar System objects, trans-Neptunian objects (TNOs) in particular, will drift through them, measured thousands of times with milliarcsecond precision (Gould 2014; Holler et al. 2018).

Such data should give remarkably good orbits from only a part of a season. But an orbit is only useful if it predicts where the object will be, and Roman sits about 1.5 million km from the Earth, near the Sun-Earth L2 point. Predictions for observers on the ground, for example of stellar occultations, which need a TNO's position to tens of km, are then made from a different place from the observations. How does the uncertainty of an orbit determined from Roman spread out when seen from Roman, and when seen from the Earth? How does that change with the object's distance, with the length of the stretch of data used, and with time?

We will:

1. Look at Roman's orbit and the survey's geometry.
2. Simulate Roman's observations of synthetic objects at 15 to 80 au, with idealised, known uncertainties.
3. Measure how well their orbits, and in particular their semi-major axes, are determined from 1 to 10 weeks of data.
4. Look at the uncertainty in position from two vantage points, Roman and the Earth, and see why they differ.
5. Separate what matters for predicting occultations: where the shadow path falls (across the motion) and when (along it), from Roman and from the ground, through the season and into the next one.

This is the fourth notebook in a series on orbital uncertainty; see also [NEOCP tracklets](../neocp_uncertainty/neocp_uncertainty.ipynb), [impact probabilities](../impact_probability/impact_probability.ipynb) and [occultations](../occultations/occultations.ipynb). Following the approach of the occultation notebook, the observations here are entirely synthetic, so that the geometry is not confused by imperfect real data (and Roman's survey has not yet begun).

**Requirements:** as for the other notebooks in the series (`orbunc` with ASSIST and its ephemeris files); Roman's trajectory is fetched from JPL Horizons and cached in `data/`.

**Run time:** about two minutes.
""")

code(r"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from orbunc import ephem, frames, spacecraft
import roman   # helper module in this folder

warnings.filterwarnings("ignore", module="erfa")
rng = np.random.default_rng(211)
KM, JD = frames.AU_KM, ephem.JD_REF
ephem.get_ephem()
TRAJ = spacecraft.Trajectory(*spacecraft.fetch_vectors("-211", "2026-12-01", "2028-10-01", "6 h", cache_dir="data"))
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
""")

md(r"""
## 1. Roman and the bulge survey

The survey's design was set by the Roman Observations Time Allocation Committee (ROTAC 2025). The numbers we use, with their sources:

| | |
|---|---|
| Fields | five contiguous fields plus one at the Galactic Centre, each about 0.28 deg², around RA 267.5°, Dec −29° (ROTAC 2025, Appendix C.3, Table 9) |
| Ecliptic latitude | about −5° to −6.5° |
| Seasons | about 72 days around each equinox, set by the Sun-angle limits; provisionally 2027 Feb 13 – Apr 25 and 2027 Aug 14 – Oct 30 (STScI Roman documentation) |
| Cadence | each field every 12.1 minutes in the wide F146 filter, 66.9-s exposures (ROTAC 2025) |
| Astrometry | about 87 mas / (signal-to-noise ratio) per exposure, with a floor of about 1.1 mas (McKinnon & van der Marel 2026; WFIRST Astrometry Working Group 2019) |
| Orbit | around the Sun-Earth L2 point, about 1.5 million km from the Earth; trajectory from JPL Horizons (target −211), a prediction made before arrival at L2 |

Roman launched on 2026 August 30; its trajectory, and the survey schedule, will change in detail. Here is where Roman is relative to the Earth during 2027, in a frame that keeps the Sun in a fixed direction, with the direction to the bulge fields.
""")

code(r"""
t = Time("2027-01-01", scale="tdb").jd + np.arange(0, 365, 0.5) - JD
geo = TRAJ(t + JD)                                          # km, geocentric ICRF
sun = ephem.body_state("sun", t)[:, :3] - ephem.body_state("earth", t)[:, :3]
ev = ephem.body_state("earth", t)[:, 3:] - ephem.body_state("sun", t)[:, 3:]
xh = -sun / np.linalg.norm(sun, axis=1)[:, None]           # away from the Sun
zh = np.cross(xh, ev); zh /= np.linalg.norm(zh, axis=1)[:, None]
yh = np.cross(zh, xh)
X, Y, Z = (np.sum(geo * v, 1) / 1e6 for v in (xh, yh, zh))
u_field = frames.radec_to_unit(*np.radians(roman.FIELDS["3"]))
B_perp = np.linalg.norm(geo - np.outer(geo @ u_field, u_field), axis=1) / 1e6   # baseline across the line of sight

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axs[0]
ax.plot(X, Y, color=roman.MUTED, lw=1)
for (a, b), c, lab in [(roman.SPRING_2027, roman.ROMAN_C, "spring season"), (roman.AUTUMN_2027, roman.COLOURS[1], "autumn season")]:
    k = (t + JD >= Time(a).jd) & (t + JD <= Time(b).jd)
    ax.plot(X[k], Y[k], color=c, lw=3, label=lab)
    m = np.where(k)[0][len(np.where(k)[0]) // 2]
    d = np.array([u_field @ xh[m], u_field @ yh[m]]); d /= np.linalg.norm(d)
    ax.annotate("", (X[m] + 0.5 * d[0], Y[m] + 0.5 * d[1]), (X[m], Y[m]), arrowprops=dict(arrowstyle="-|>", color=c))
ax.plot(0, 0, "o", color=roman.INK, ms=8)
ax.text(0.05, 0.05, "Earth", fontsize=9)
ax.annotate("to the Sun", (-0.4, 0), (0.2, 0.0), arrowprops=dict(arrowstyle="-|>", color="#eda100"), fontsize=8, va="center")
ax.set_aspect("equal")
ax.set_xlabel("away from the Sun (million km)")
ax.set_ylabel("along the Earth's orbit (million km)")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("(a) Roman's 2027 path relative to the Earth; arrows: towards the bulge", fontsize=10)
ax = axs[1]
ax.plot(Time(t + JD, format="jd").datetime, np.linalg.norm(geo, axis=1) / 1e6, color=roman.INK, lw=1.5, label="Earth-Roman distance")
ax.plot(Time(t + JD, format="jd").datetime, B_perp, color=roman.ROMAN_C, lw=1.5, label="its component across the line of sight to the bulge")
for a, b in (roman.SPRING_2027, roman.AUTUMN_2027):
    ax.axvspan(Time(a).datetime, Time(b).datetime, color=roman.MUTED, alpha=0.15, lw=0)
ax.set_ylabel("million km")
ax.legend(fontsize=8, frameon=False)
fig.autofmt_xdate()
ax.set_title("(b) the Earth-Roman baseline (shaded: survey seasons)", fontsize=10)
fig.suptitle("Figure 1. Roman's orbit in 2027 (JPL Horizons prediction)", fontsize=11)
plt.tight_layout()
plt.show()
for (a, b), lab in [(roman.SPRING_2027, "spring"), (roman.AUTUMN_2027, "autumn")]:
    k = (t + JD >= Time(a).jd) & (t + JD <= Time(b).jd)
    print(f"{lab} season: baseline across the line of sight {B_perp[k].min():.2f} to {B_perp[k].max():.2f} million km")
""")

md(r"""
The bulge is observed near quadrature, about 90° from the Sun, so the line of sight runs roughly perpendicular to the Sun-Earth line, and Roman's offset from the Earth, mostly away from the Sun, lies largely *across* it. That offset is a baseline of more than a million kilometres between the place where the observations are made and the place where many predictions will be used.

## 2. Synthetic distant objects and their observations

We place four synthetic objects on the line of sight from Roman to the centre of field 3 in the middle of the spring 2027 season (March 21), on near-circular orbits inclined by about 8°: a Centaur at 15 au, a plutino-like object at 39 au, a classical TNO at 44 au and a distant object at 80 au.

Roman measures each object every 12.1 minutes; we bin the exposures in groups of five, one point an hour. For the per-exposure precision we take a source of F146 = 24 AB, for which the formula above gives about 11 mas (S/N about 8; for an albedo of 0.1, a TNO roughly 100 km across at 40 au would be about this faint, while one of 300 km would be some 2.5 magnitudes brighter; Roman's single-exposure detection limit is near 25.4). The errors are independent and Gaussian, so the orbit fits' uncertainties are right by construction.
""")

code(r"""
MAG = 24.0
SIG_EXP = roman.precision_mas(MAG)
T_MID = Time("2027-03-21", scale="tdb").jd - JD
OBJECTS = {"Centaur, 15 au": 15.0, "plutino, 39 au": 39.4, "classical TNO, 44 au": 44.0, "distant, 80 au": 80.0}
truth = {}
for name, r in OBJECTS.items():
    x, rho = roman.place(TRAJ, T_MID, *roman.FIELDS["3"], r, r, 8.0)
    truth[name] = x
    print(f"{name:22s}: distance from Roman {rho:5.1f} au")
print(f"precision per exposure at F146 = {MAG}: {SIG_EXP:.1f} mas; per hourly point {SIG_EXP / np.sqrt(5):.1f} mas")

fig, ax = plt.subplots(figsize=(7, 4.5))
season = Time(roman.SPRING_2027[0]).jd + np.arange(0, 71, 0.5)
ts = Time(season, format="jd", scale="utc").tdb.jd - JD
o = roman.observer(TRAJ, ts, "Roman")
for (name, x), c in zip(truth.items(), roman.COLOURS):
    P = ephem.propagate(x[None], T_MID, ts)[:, 0, :3] - o
    ra, dec = frames.unit_to_radec(P)
    ax.plot(np.degrees(ra), np.degrees(dec), color=c, lw=2, label=name)
    ax.plot(np.degrees(ra[0]), np.degrees(dec[0]), "o", color=c, ms=4)
for f, (ra_f, dec_f) in roman.FIELDS.items():
    ax.plot(ra_f, dec_f, "s", color=roman.MUTED, ms=5)
    ax.text(ra_f + 0.05, dec_f + 0.05, f, fontsize=7, color=roman.MUTED)
ax.invert_xaxis()
ax.set_xlabel("RA (deg)"); ax.set_ylabel("Dec (deg)")
ax.legend(fontsize=8, frameon=False)
ax.set_title("Figure 2. Paths of the synthetic objects through the 2027 spring season, seen from Roman\n(dots: season start; squares: field centres)", fontsize=10)
plt.show()
for name, x in truth.items():
    P = ephem.propagate(x[None], T_MID, ts)[:, 0, :3] - o
    u = P / np.linalg.norm(P, axis=1)[:, None]
    rate = np.degrees(np.arccos(np.clip(np.sum(u[1:] * u[:-1], 1), -1, 1))) * 3600 / 12   # arcsec per hour (0.5-d steps)
    k = np.argmin(rate)
    print(f"{name:22s}: apparent rate from {rate.max():.2f} down to {rate[k]:.3f} arcsec/h, slowest on "
          f"{Time(season[k], format='jd').iso[:10]} (its stationary point)")
""")

md(r"""
Over a 70-day season a TNO moves less than a degree, so it can cross a field (about 0.8° by 0.4°) in a few weeks; the Centaur moves faster. The paths also turn round: the bulge is observed near quadrature, where distant objects reach their *stationary points*, when the observer's own motion around the Sun cancels their east-west motion and they turn round in right ascension; their apparent motion is then slowest (late March here, printed above). This will matter in Section 5. A TNO that passes through the survey will typically be measured over a part of a season, not all of it: the natural question is what a *partial* stare gives.

## 3. How well is the orbit determined from part of a season?

We fit each object's orbit to arcs of 1 to 10 weeks starting on the first day of the season, and measure the uncertainty of the semi-major axis, $\sigma_a/a$, from the fit's covariance (sampled, and converted to orbital elements).
""")

code(r"""
ARCS = [7, 10, 14, 21, 28, 42, 70]
fits, da = {}, {}
for name, x in truth.items():
    da[name] = []
    for arc in ARCS:
        O = roman.observe(TRAJ, x, T_MID, roman.SPRING_2027[0], arc, SIG_EXP, rng)
        f = roman.fit_from_truth(O, x, T_MID)
        S = rng.multivariate_normal(f.state, f.cov, 2000)
        el = frames.elements(frames.bary_eq_to_helio_ec(S, f.t))
        da[name].append(np.std(el["a"]) / np.mean(el["a"]))
        fits[(name, arc)] = (O, f, S)
    print(f"{name:22s} sigma_a/a: " + ", ".join(f"{arc} d {v:.1e}" for arc, v in zip(ARCS, da[name])))

fig, ax = plt.subplots(figsize=(7.5, 4.2))
for (name, v), c in zip(da.items(), roman.COLOURS):
    ax.plot(ARCS, v, "o-", color=c, lw=2, label=name)
ax.axhline(1e-3, color=roman.INK, ls="--", lw=1)
ax.text(70, 1.2e-3, r"$\sigma_a/a = 10^{-3}$", ha="right", fontsize=8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(ARCS); ax.set_xticklabels(ARCS); ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("length of the Roman arc (days)")
ax.set_ylabel(r"$\sigma_a / a$")
ax.legend(fontsize=8, frameon=False)
ax.set_title("Figure 3. Semi-major axis uncertainty from part of a season", fontsize=11)
plt.show()
""")

md(r"""
The uncertainty falls steeply with the length of the arc, roughly as a high power of it: a longer arc both adds measurements and, more importantly, lets the curvature of the path, which is what reveals the distance and the orbit, grow. An uncertainty of $\sigma_a/a \approx 10^{-3}$ is reached with two to four weeks of data for a TNO with these measurement errors, and sooner for the nearer Centaur; the full season does far better. (For ground-based data, by contrast, it takes years of observations to reach this; Bernstein & Khushalani 2000 describe the short-arc degeneracies of TNO orbits.) Closer objects are better determined for the same arc, because their apparent path curves more.

## 4. The uncertainty from two vantage points

Take the plutino-like object after a 14-day arc. Its orbit predicts a position at any time, with an uncertainty that is a three-dimensional cloud. Roman's data measure directions from Roman very well, but the distance from Roman less well: so the cloud is long and thin, pointing along Roman's line of sight. Figure 4 shows it at the end of the arc, side on, and face on from Roman and from the Earth.
""")

code(r"""
NAME, ARC = "plutino, 39 au", 14
O, f, S = fits[(NAME, ARC)]
t_end = O.t[-1]
sp = roman.spread(S, f.t, f.state, t_end, TRAJ)
cR, cE = sp["Roman"]["cloud"], sp["Earth"]["cloud"]
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
ax = axs[0]
ax.scatter(cR[:, 2] / 1e3, cR[:, 0], s=2, color=roman.MUTED, alpha=0.4, lw=0, rasterized=True)
L = np.abs(cR[:, 2]).max() / 1e3
oR, oE = roman.observer(TRAJ, t_end, "Roman")[0], roman.observer(TRAJ, t_end, "Earth")[0]
base = (oR - oE) * KM
FR = roman.apparent_frame(ephem.propagate(f.state[None], f.t, [t_end])[0, 0], t_end, oR, roman.observer(TRAJ, t_end + 0.01, "Roman")[0])[0]
dist_km = sp["Roman"]["dist_au"] * KM
slope = (base @ FR[0]) / dist_km      # change of the Earth's sightline, along-track, per km along Roman's
xs = np.array([-L, L])
ax.plot(xs, 0 * xs, color=roman.ROMAN_C, lw=2, label="Roman's line of sight")
ax.plot(xs, slope * xs * 1e3, color=roman.EARTH_C, lw=2, label="the Earth's line of sight")
ax.set_xlabel("along Roman's line of sight (thousand km)")
ax.set_ylabel("along-track (km)")
ax.legend(fontsize=8, frameon=False)
ax.set_title("(a) side on (note the very different scales)", fontsize=10)
lim = 4 * max(sp["Earth"]["along"], sp["Earth"]["across"])
for ax, who, c in [(axs[1], "Roman", roman.ROMAN_C), (axs[2], "Earth", roman.EARTH_C)]:
    d = sp[who]["cloud"]
    ax.scatter(d[:, 0], d[:, 1], s=2, color=c, alpha=0.4, lw=0, rasterized=True)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("along the apparent motion (km)")
    ax.set_ylabel("across it (km)")
    ax.set_title(f"(c) seen from the Earth" if who == "Earth" else "(b) seen from Roman", fontsize=10)
fig.suptitle(f"Figure 4. The {NAME} after a {ARC}-day Roman arc: its position uncertainty at the end of the arc", fontsize=11)
plt.tight_layout()
plt.show()
B = np.linalg.norm(base - (base @ FR[2]) * FR[2])
print(f"distance uncertainty along Roman's line of sight: {sp['Roman']['los']:,.0f} km ({sp['Roman']['los'] / dist_km:.1e} of the distance)")
print(f"baseline Earth-Roman across the line of sight: {B / 1e6:.2f} million km; expected extra spread "
      f"B x sigma_rho / rho = {B * sp['Roman']['los'] / dist_km:,.0f} km")
for who in ("Roman", "Earth"):
    print(f"seen from {who:5s}: 1-sigma along the motion {sp[who]['along']:6.0f} km, across it {sp[who]['across']:5.0f} km")
""")

md(r"""
Seen from Roman (b), the cloud is compact: Roman measured these directions directly. Seen from the Earth (c) it is several times longer. Nothing about the cloud has changed: the Earth simply looks at the long, thin cloud from a slightly different direction, and sees some of its length (a). The angle is tiny, the baseline divided by the distance, about $1.6\times10^6\,{\rm km}/5.9\times10^9\,{\rm km} \approx 3\times10^{-4}$ rad, but the cloud is more than a million km long, so the extra spread is about

$$\Delta \approx B_\perp \, \frac{\sigma_\rho}{\rho},$$

the baseline across the line of sight times the *fractional* uncertainty of the distance, here a few hundred km. It does not depend on the distance itself, only on how well it is known in proportion.

## 5. Predicting occultations, from Roman and from the ground

For a stellar occultation the two directions on the sky play different roles. The asteroid's shadow sweeps across the observer's plane along the direction of the apparent motion, so an error **along** the motion only shifts the *time* of the event (by the error divided by the shadow's speed); an error **across** the motion shifts the *path*, and decides whether a given observer sees the event at all. A TNO a few hundred km across can be observed only if the path is known to a fraction of its size.

Here is how both evolve after the end of 14- and 28-day arcs, for predictions made for observers at Roman and on the Earth (the Earth's centre; observers on its surface differ by at most its radius, negligible here), through the rest of the season and into the autumn season.
""")

code(r"""
LATER = [0, 3, 7, 14, 28, 56]
autumn = Time(roman.AUTUMN_2027[0], scale="utc").tdb.jd - JD
rows = {}
for name in truth:
    for arc in (14, 28):
        O, f, S = fits[(name, arc)]
        times = list(O.t[-1] + np.array(LATER)) + [autumn]
        rows[(name, arc)] = [roman.spread(S, f.t, f.state, tt, TRAJ) for tt in times]

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for row, arc in enumerate((14, 28)):
    for (name, x), c in zip(truth.items(), roman.COLOURS):
        sp = rows[(name, arc)][:-1]
        for who, ls in (("Roman", "-"), ("Earth", "--")):
            axs[row, 0].plot(LATER, [s[who]["across"] for s in sp], ls, color=c, lw=2, label=f"{name}, {who}" if row == 0 else None)
            axs[row, 1].plot(LATER, [s[who]["along"] / s[who]["rate_km_s"] for s in sp], ls, color=c, lw=2)
    axs[row, 0].set_ylabel(f"after a {arc}-day arc\npath uncertainty, across (km)")
    axs[row, 1].set_ylabel("timing uncertainty (s)")
    for ax in axs[row]:
        ax.set_yscale("log")
axs[1, 0].set_xlabel("days after the end of the arc"); axs[1, 1].set_xlabel("days after the end of the arc")
axs[0, 0].legend(fontsize=7, frameon=False, ncol=2)
fig.suptitle("Figure 5. Occultation predictions from a Roman arc: solid, for Roman; dashed, for the Earth", fontsize=11)
plt.tight_layout()
plt.show()

for name in truth:
    O, f, S = fits[(name, 70)]
    s70 = roman.spread(S, f.t, f.state, autumn, TRAJ)
    print(f"{name:22s} after the full 70-day arc, at the start of the autumn season: across {s70['Earth']['across']:,.0f} km, "
          f"timing {s70['Earth']['along'] / s70['Earth']['rate_km_s']:,.0f} s (for the Earth)")
print()
print(f"{'object':22s} {'arc':>4s} {'when':>16s} {'across (km): Roman / Earth':>28s} {'timing (s): Roman / Earth':>27s}")
for (name, arc), sp in rows.items():
    for lab, s in zip(["end of arc", "+7 d", "+28 d", "autumn season"], [sp[0], sp[2], sp[4], sp[-1]]):
        R, E = s["Roman"], s["Earth"]
        print(f"{name:22s} {arc:4d} {lab:>16s} {R['across']:12,.0f} / {E['across']:<12,.0f} {R['along'] / R['rate_km_s']:12,.0f} / {E['along'] / E['rate_km_s']:<10,.0f}")
""")

md(r"""
What stands out:

* **At first, the vantage point matters.** At the end of a 14-day arc the TNOs' paths are predicted to a few tens of km for Roman, but to about 100 to 200 km for the Earth: the extra $B_\perp \sigma_\rho/\rho$ of Section 4. After 28 days of data the difference is about a factor of two. Within a few weeks the difference fades, because other parts of the uncertainty grow and dominate.
* **The path uncertainty grows within days to weeks.** After a 14-day arc, the TNOs' paths are uncertain by a few hundred km after a week and a few thousand km after a month; a 28-day arc keeps them near 100 km for a month, enough for occultations by TNOs of a few hundred km. The curves are not monotonic: the uncertainty grows mainly *along the orbit*, and how that divides between "along" and "across" the apparent motion changes as the direction of the apparent motion swings around the stationary point (Figure 2).
* **Timing uncertainties are seconds at first, minutes after a month.** Near the stationary point the shadow moves slowly, so a given along-track error means a long timing error; a timing uncertainty, though, can be covered by observing for longer.
* **The gap between seasons is the real limit.** By the autumn season, five months later, the paths from a 14-day arc are uncertain by tens of thousands of km, and from a 28-day arc by thousands, from either vantage point. Only the full season's data (printed above) carry a TNO's path into the next season, with an uncertainty of one to a few hundred km.

## 6. Summary

* From part of a 70-day stare, Roman's milliarcsecond astrometry determines a TNO's semi-major axis to $\sigma_a/a \sim 10^{-3}$ in two to four weeks (for objects of about F146 = 24), and the full season to $\sim 10^{-5}$. Nearer objects need shorter arcs.
* The resulting uncertainty is a long, thin cloud along Roman's line of sight. Roman sees it end on; observers on the Earth, 1.5 million km away and largely across the line of sight, see part of its length: an extra spread of about $B_\perp\,\sigma_\rho/\rho$, a few hundred km after two weeks, independent of the object's distance.
* For occultation predictions, the path stays within about 100 km for a month after a 28-day arc but degrades within a week or two after a 14-day one; timing uncertainties grow from seconds to minutes; and the five-month gap between seasons is the main obstacle for any arc short of the full season. Near quadrature the objects pass their stationary points, which complicates the split between path and timing errors.

## 7. Going further

* **Brightness.** Change `MAG`: each magnitude changes the precision by a factor of 2.5, and the arc needed for a given $\sigma_a/a$ accordingly.
* **Combining Roman with the ground.** A single ground-based position taken during the season adds a baseline of 1.5 million km to Roman's data and directly measures the distance (Giovinazzi, Blake & Bernardinelli 2021); add one with `orbunc.simulate.astrometry` and refit.
* **Real orbits.** Replace the synthetic objects by known TNOs whose paths cross the fields (Porter et al. 2022 used a far longer baseline, from New Horizons, for the same purpose).

## References

* Bernstein, G. & Khushalani, B. (2000), *Orbit fitting and uncertainties for Kuiper Belt objects*, AJ 120, 3323. [ADS](https://ui.adsabs.harvard.edu/abs/2000AJ....120.3323B/abstract)
* Giovinazzi, M. R., Blake, C. H. & Bernardinelli, P. H. (2021), *Enhancing ground-based observations of trans-Neptunian objects using a single-epoch parallax measurement from L2*, PASP 133, 114401. [ADS](https://ui.adsabs.harvard.edu/abs/2021PASP..133k4401G/abstract)
* Gould, A. (2014), *WFIRST ultra-precise astrometry I: Kuiper Belt objects*, JKAS 47, 279. [ADS](https://ui.adsabs.harvard.edu/abs/2014JKAS...47..279G/abstract)
* Holler, B. J., Milam, S. N., Bauer, J. M., et al. (2018), *Solar system science with the Wide-Field Infrared Survey Telescope*, JATIS 4, 034003. [ADS](https://ui.adsabs.harvard.edu/abs/2018JATIS...4c4003H/abstract)
* McKinnon, K. A. & van der Marel, R. P. (2026), *Simulating Roman+Gaia combined astrometry, parallaxes, and proper motions*, PASP 138, 044507. [ADS](https://ui.adsabs.harvard.edu/abs/2026PASP..138d4507M/abstract)
* Porter, S. B., Spencer, J. R., Verbiscer, A., et al. (2022), *Orbits and occultation opportunities of 15 TNOs observed by New Horizons*, PSJ 3, 23. [ADS](https://ui.adsabs.harvard.edu/abs/2022PSJ.....3...23P/abstract)
* Roman Observations Time Allocation Committee (2025), *Final report and recommendations*, arXiv:2505.10574. [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv250510574O/abstract)
* WFIRST Astrometry Working Group, Sanderson, R. E., Bellini, A., et al. (2019), *Astrometry with the Wide-Field Infrared Space Telescope*, JATIS 5, 044005. [ADS](https://ui.adsabs.harvard.edu/abs/2019JATIS...5d4005W/abstract)
* STScI Roman documentation (RDox), Galactic Bulge Time Domain Survey and the provisional schedule: https://roman-docs.stsci.edu/
* JPL Horizons: https://ssd.jpl.nasa.gov/horizons/
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Wrote {OUT}")
