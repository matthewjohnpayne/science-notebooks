"""Helpers for the NEOCP uncertainty notebook: data, the tracklet-by-tracklet sequence, figures.

The orbit machinery (ASSIST propagation, attributables, systematic ranging, least squares,
projections) is in the `orbunc` toolkit at the top of this repository; this module only
organises it for the notebook.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import requests
from astropy.time import Time

from orbunc import ephem, fit, frames, mpc, ranging, sky

# Categorical colours for 1, 2 and 3 tracklets (validated as a set, colour-blind safe), and
# for three orbit-class groups. Text and reference marks stay in ink colours.
STAGE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
GROUPS = {"NEO": ["Atira", "Aten", "Apollo", "Amor"], "Main belt or Mars-crosser": ["Mars-crosser", "Main belt"],
          "Further out or unbound": ["Outer", "Unbound"]}
GROUP_COLOURS = {"NEO": "#eb6834", "Main belt or Mars-crosser": "#2a78d6", "Further out or unbound": "#1baf7a"}
INK, MUTED = "#0b0b0b", "#8a8984"


def load(desig, cache_dir="data", start=None, end=None):
    """Observations (optionally only between ISO dates start and end) and the MPC's current orbit
    (the 'truth' for this exercise)."""
    obs = mpc.from_ades(mpc.fetch_ades(desig, cache_dir), start=start, end=end)
    t0, x0, cov = mpc.orbit_state(mpc.fetch_orbit(desig, cache_dir))
    return obs, (t0, x0, cov)


def tracklet_table(obs, groups):
    rows = []
    for g in groups:
        t = Time(obs.jd_utc[g], format="jd")
        rows.append((t[0].iso[:16], obs.stn[g[0]], len(g), 24 * np.ptp(obs.jd_utc[g]),
                     float(np.median(np.hypot(obs.sig_ra[g], obs.sig_dec[g]) / np.sqrt(2)))))
    print(f"{'first observation (UTC)':24s} {'station':>7s} {'obs':>4s} {'span (h)':>9s} {'sigma (arcsec)':>15s}")
    for r in rows:
        print(f"{r[0]:24s} {r[1]:>7s} {r[2]:4d} {r[3]:9.2f} {r[4]:15.2f}")


def sequence(attr, obs, groups, **kw):
    """Systematic ranging with the first 1, 2, ... tracklets; returns one dict per stage."""
    out = []
    for n in range(1, len(groups) + 1):
        O = obs.select(np.concatenate(groups[:n]))
        nodes = ranging.systematic_ranging(attr, O, **kw)
        w = ranging.weights(nodes, attr)
        el = frames.elements(frames.bary_eq_to_helio_ec(np.nan_to_num(nodes.states), attr.t))
        out.append(dict(n=n, obs=O, nodes=nodes, w=w, el=el))
    return out


def draw(stage, attr, n, rng):
    """n orbits drawn from a ranging stage, with their elements."""
    X, rho, rhodot = ranging.sample(stage["nodes"], attr, stage["obs"], n, rng)
    return X, frames.elements(frames.bary_eq_to_helio_ec(X, attr.t))


def group_of(el):
    cls = sky.orbit_class(el)
    g = np.empty(len(cls), dtype=object)
    for name, members in GROUPS.items():
        g[np.isin(cls, members)] = name
    return g


def group_probabilities(el, w):
    g = group_of(el)
    return {name: float(w[g == name].sum()) for name in GROUPS}


def wmean_sd(x, w):
    m = np.sum(w * x)
    return m, np.sqrt(np.sum(w * (x - m) ** 2))


def wquantile(x, w, q):
    k = np.argsort(x)
    c = np.cumsum(w[k])
    return np.interp(q, c / c[-1], x[k])


# --- digest2 ----------------------------------------------------------------

def digest2_tracklet(obs, idx):
    """digest2 scores and trial orbits for one tracklet (needs the digest2 package)."""
    from digest2 import Digest2, Observation
    oc = os.path.join(ephem.data_dir(), "digest2.obscodes")
    if not os.path.exists(oc):
        r = requests.get("https://data.minorplanetcenter.net/api/obscodes", json={"format": "ObsCodes.html"}, timeout=120)
        r.raise_for_status()
        open(oc, "w").write(r.text)
    trk = [Observation(mjd=jd - 2400000.5, ra=np.degrees(ra), dec=np.degrees(de),
                       mag=(m if np.isfinite(m) else 20.0), band=(b or "V"), obscode=s)
           for jd, ra, de, m, b, s in zip(obs.jd_utc[idx], obs.ra[idx], obs.dec[idx], obs.mag[idx], obs.band[idx], obs.stn[idx])]
    with Digest2(obscodes_path=oc, repeatable=True) as d2:
        return d2.classify_tracklet(trk, collect_orbits=True)


# --- figures ----------------------------------------------------------------

def plot_admissible(ax, attr, nodes, truth_rr=None, title="", vmax=50):
    """The admissible region in (rho, rho-dot), nodes coloured by delta-chi2."""
    rr = np.geomspace(1e-4, 100, 800)
    lo, hi = ranging.rhodot_interval(attr, rr)
    ok = np.isfinite(lo)
    ax.fill_between(rr[ok], lo[ok], hi[ok], color="#f1f0ec", zorder=0, lw=0)
    ax.plot(rr[ok], lo[ok], color=MUTED, lw=1)
    ax.plot(rr[ok], hi[ok], color=MUTED, lw=1)
    d = nodes.chi2 - np.nanmin(nodes.chi2)
    good = np.isfinite(d)
    sc = ax.scatter(nodes.rho[good], nodes.rhodot[good], c=np.minimum(d[good], vmax), cmap="Blues_r",
                    vmin=0, vmax=vmax, s=6, lw=0, rasterized=True)
    if truth_rr is not None:
        ax.plot(*truth_rr, marker="*", color=INK, ms=13, mec="white", mew=1, ls="none", zorder=5)
        ax.annotate("true orbit", truth_rr, xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel(r"distance from the observer, $\rho$ (au)")
    ax.set_ylabel(r"radial velocity, $\dot\rho$ (au/day)")
    ax.set_title(title, fontsize=11)
    return sc


def plot_spray(ax, el_samples, el_truth=None, lim=4.0, title="", n_max=200):
    """Top-down (ecliptic) view of sampled orbits, coloured by orbit-class group."""
    paths = sky.orbit_paths(el_samples)[:n_max]
    groups = group_of(el_samples)[:n_max]
    th = np.linspace(0, 2 * np.pi, 200)
    for name in GROUPS:
        k = groups == name
        for p in paths[k]:
            ax.plot(p[:, 0], p[:, 1], color=GROUP_COLOURS[name], lw=0.5, alpha=0.35)
    ax.plot(np.cos(th), np.sin(th), color=INK, lw=1.2)
    ax.text(0.72, -0.86, "Earth", fontsize=9, color=INK)
    if el_truth is not None:
        p = sky.orbit_paths(el_truth)[0]
        ax.plot(p[:, 0], p[:, 1], color=INK, lw=1.4, ls="--")
    ax.plot(0, 0, marker="o", color="#eda100", ms=6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (au)")
    ax.set_title(title, fontsize=11)


def group_legend(ax, loc="lower left"):
    handles = [plt.Line2D([], [], color=GROUP_COLOURS[g], lw=2, label=g) for g in GROUPS]
    handles.append(plt.Line2D([], [], color=INK, lw=1.4, ls="--", label="true orbit"))
    ax.legend(handles=handles, loc=loc, fontsize=8, frameon=False)
