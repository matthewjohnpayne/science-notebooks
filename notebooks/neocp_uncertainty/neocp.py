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


# --- the line of sight ------------------------------------------------------------

def _helio_ec(states, t):
    """Heliocentric ecliptic states from barycentric ones, shape (len(t), N, 6)."""
    sun = ephem.body_state("sun", t)
    return frames.rotate_states(np.asarray(states) - sun[:, None, :], frames.EQ_TO_EC)


def ray_geometry(attr, X, arc_days=15.0, n_arc=31):
    """Heliocentric ecliptic geometry of orbits X (states at attr.t) that fit one tracklet.

    Returns dict: observer (3,), ray direction (3,), rho (N,), positions at attr.t (N, 3),
    arcs (n_arc, N, 3) over +-arc_days, and the Earth's orbit (200, 3) for reference.
    """
    times = attr.t + np.linspace(-arc_days, arc_days, n_arc)
    P = ephem.propagate(X, attr.t, times)                                   # (n_arc, N, 6)
    arcs = _helio_ec(P, times)[..., :3]
    now6 = _helio_ec(X[None], np.array([attr.t]))[0]
    obs6 = _helio_ec(attr.observer[None, None], np.array([attr.t]))[0, 0]
    now, obs = now6[:, :3], obs6[:3]
    u = frames.radec_to_unit(attr.x[0], attr.x[1]) @ frames.EQ_TO_EC.T
    rho = np.linalg.norm(now - obs, axis=1)
    year = attr.t + np.linspace(0, 365.25, 200)
    earth = frames.rotate_states(ephem.body_state("earth", year) - ephem.body_state("sun", year), frames.EQ_TO_EC)[:, :3]
    v_rel = now6[:, 3:] - obs6[3:]
    v_across = np.linalg.norm(v_rel - np.outer(v_rel @ u, u), axis=1)
    return dict(observer=obs, observer_v=obs6[3:], u=u, rho=rho, now=now, arcs=arcs, earth=earth,
                v_helio=np.linalg.norm(now6[:, 3:], axis=1), v_across=v_across)


def plot_ray_views(g, cmap="Blues", vmin=None, vmax=None, rho_true=None, zoom_au=0.6):
    """Three views of a family of orbits crossing one line of sight (see ray_geometry)."""
    from matplotlib.colors import LogNorm
    import matplotlib as mpl
    rho = g["rho"]
    norm = LogNorm(vmin or rho.min(), vmax or rho.max())
    cm = mpl.colormaps[cmap]
    col = lambda r: cm(0.3 + 0.7 * norm(r))
    o, u = g["observer"], g["u"]
    ray = o[None] + np.outer([0, rho.max() * 1.05], u)
    fig = plt.figure(figsize=(13, 9))
    axs = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), fig.add_subplot(2, 1, 2)]
    for ax, lim, title in [(axs[0], None, "(a) seen from above the ecliptic"),
                           (axs[1], zoom_au, f"(b) the same, within {zoom_au} au of the observer")]:
        ax.plot(g["earth"][:, 0], g["earth"][:, 1], color=MUTED, lw=1)
        ax.plot(0, 0, "o", color="#eda100", ms=7)
        ax.plot(*ray[:, :2].T, color=INK, lw=1, ls="--")
        for k in np.argsort(-rho):
            a = g["arcs"][:, k]
            ax.plot(a[:, 0], a[:, 1], color=col(rho[k]), lw=1, alpha=0.8)
            ax.annotate("", a[-1, :2], a[-4, :2], arrowprops=dict(arrowstyle="-|>", color=col(rho[k]), lw=0.8))
        ax.scatter(g["now"][:, 0], g["now"][:, 1], c=[col(r) for r in rho], s=10, zorder=3, edgecolors="none")
        ax.plot(*o[:2], "o", color=INK, ms=6, zorder=4)
        ax.annotate("observer (Earth)", o[:2], xytext=(6, -12), textcoords="offset points", fontsize=8)
        if lim:
            ax.set_xlim(o[0] - lim, o[0] + lim)
            ax.set_ylim(o[1] - lim, o[1] + lim)
        ax.set_aspect("equal", adjustable="datalim" if lim is None else "box")
        ax.set_xlabel("x, ecliptic (au)")
        ax.set_ylabel("y, ecliptic (au)")
        ax.set_title(title, fontsize=10)
    # (c) speeds: what the tracklet fixes (the speed across the line of sight) and what it does not
    ax = axs[2]
    kms = frames.AU_KM / 86400
    rr = np.geomspace(rho.min() * 0.7, rho.max() * 1.3, 200)
    pos = o[None] + np.outer(rr, u)
    v_esc = np.sqrt(2 * frames.GM_SUN / np.linalg.norm(pos, axis=1)) * kms
    order = np.argsort(rho)
    ax.plot(rho[order], g["v_across"][order] * kms, color=INK, lw=1.5, label="speed across the line of sight, relative to the observer")
    ax.scatter(rho, g["v_helio"] * kms, c=[col(r) for r in rho], s=12, edgecolors="none", zorder=3,
               label="speed around the Sun (one dot per orbit)")
    ax.plot(rr, v_esc, color="#e34948", lw=1.5, ls="--", label="escape speed from the Sun at that point")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("distance from the observer at the time of the tracklet (au)")
    ax.set_ylabel("speed (km/s)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("(c) every orbit has the same angular rate, so its speed across the line of sight grows with distance", fontsize=10)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colors.ListedColormap([col(r) for r in np.geomspace(norm.vmin, norm.vmax, 256)]))
    fig.colorbar(sm, ax=axs, label="distance from the observer at the time of the tracklet (au)", shrink=0.6)
    return fig, axs


def plot_sky_fans(attr, X, rho, days, stn, x_true=None, t_true=None, cmap="Blues", vmin=None, vmax=None):
    """The observer's view: where each orbit puts the asteroid on the sky `days` after the tracklet,
    relative to the true position (or to the median if no truth is given), coloured by distance."""
    from matplotlib.colors import LogNorm
    import matplotlib as mpl
    norm = LogNorm(vmin or rho.min(), vmax or rho.max())
    cm = mpl.colormaps[cmap]
    order = np.argsort(-rho)
    fig, axs = plt.subplots(1, len(days), figsize=(4.3 * len(days), 4.3))
    for ax, d in zip(np.atleast_1d(axs), days):
        jd = attr.jd_utc + d
        ra, dec = sky.sky_positions(X, attr.t, [jd], stn)
        if x_true is not None:
            ra0, dec0 = sky.sky_positions(x_true[None], t_true, [jd], stn)
            ra0, dec0 = ra0[0, 0], dec0[0, 0]
        else:
            ra0, dec0 = sky.weighted_centre(ra[:, 0], dec[:, 0])
        e, n = sky.offsets(ra[:, 0], dec[:, 0], ra0, dec0, unit=np.pi / 180)
        ax.scatter(e[order], n[order], c=cm(0.3 + 0.7 * norm(rho[order])), s=6, edgecolors="none")
        ax.plot(0, 0, marker="*", color=INK, ms=12, mec="white", ls="none")
        ax.invert_xaxis()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("east offset (deg)")
        ax.set_ylabel("north offset (deg)")
        ax.set_title(f"{d:g} days after the tracklet", fontsize=10)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colors.ListedColormap([cm(0.3 + 0.7 * norm(r)) for r in np.geomspace(norm.vmin, norm.vmax, 256)]))
    fig.colorbar(sm, ax=axs, label="distance at the time of the tracklet (au)", shrink=0.8)
    return fig, axs


def plot_sky_tracks(attr, X, rho, stn, days=30.0, x_true=None, t_true=None, cmap="Blues", vmin=None, vmax=None, pct=90):
    """The observer's view: each orbit's track on the sky for `days` after the tracklet, relative to the
    straight-line (constant-rate) extrapolation of the tracklet, coloured by distance. Axis limits
    contain the central `pct` percent of the end points; the fraction outside is reported."""
    from matplotlib.colors import LogNorm
    import matplotlib as mpl
    norm = LogNorm(vmin or rho.min(), vmax or rho.max())
    cm = mpl.colormaps[cmap]
    dt = np.linspace(0, days, 61)
    jd = attr.jd_utc + dt
    ra, dec = sky.sky_positions(X, attr.t, jd, stn)
    a0, d0, ad, dd = attr.x
    ra_l, dec_l = a0 + ad * (attr.t + dt - attr.t), d0 + dd * dt          # the straight-line extrapolation
    e, n = sky.offsets(ra, dec, ra_l[None], dec_l[None], unit=np.pi / 180)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for k in np.argsort(-rho):
        ax.plot(e[k], n[k], color=cm(0.3 + 0.7 * norm(rho[k])), lw=0.8, alpha=0.8)
    if x_true is not None:
        rt, dt_ = sky.sky_positions(x_true[None], t_true, jd, stn)
        et, nt = sky.offsets(rt, dt_, ra_l[None], dec_l[None], unit=np.pi / 180)
        ax.plot(et[0], nt[0], color=INK, lw=2, ls="--", label="the true track")
        ax.plot(et[0, -1], nt[0, -1], marker="*", color=INK, ms=12, mec="white", ls="none")
        ax.legend(fontsize=8, frameon=False)
    r = np.hypot(e[:, -1], n[:, -1])
    lim = np.percentile(r, pct) * 1.1
    ax.set_xlim(lim, -lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("east offset from the straight-line extrapolation (deg)")
    ax.set_ylabel("north offset (deg)")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colors.ListedColormap([cm(0.3 + 0.7 * norm(v)) for v in np.geomspace(norm.vmin, norm.vmax, 256)]))
    fig.colorbar(sm, ax=ax, label="distance at the time of the tracklet (au)", shrink=0.8)
    return fig, ax, float(np.mean(r > lim))
