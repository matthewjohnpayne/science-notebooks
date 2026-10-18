"""Helpers for the impact-probability notebook: the toy model, the orbit fits by date, figures.

The orbit machinery (ASSIST propagation, fits, encounters and the b-plane) is in the `orbunc`
toolkit at the top of this repository.
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from orbunc import encounter, ephem, fit, frames, mpc

KM = frames.AU_KM
STAGE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, EARTH = "#0b0b0b", "#8a8984", "#2a78d6"


# --- the toy model ----------------------------------------------------------

def toy_probability(offset, sigma_long, sigma_short, radius=1.0, n=400_000, rng=None):
    """Probability that a point drawn from a 2-D Gaussian (centred at (0, offset), standard
    deviations sigma_short across and sigma_long along the offset) lands within `radius` of
    the origin."""
    rng = rng or np.random.default_rng(0)
    x = rng.normal(0, sigma_short, n)
    y = offset + rng.normal(0, sigma_long, n)
    return float(np.mean(x * x + y * y < radius * radius))


# --- data and fits ------------------------------------------------------------

def load(desig="2024 YR4", cache_dir="data"):
    obs = mpc.from_ades(mpc.fetch_ades(desig, cache_dir))
    t0, x0, cov = mpc.orbit_state(mpc.fetch_orbit(desig, cache_dir))
    return obs, mpc.submission_times(obs), (t0, x0, cov)


def closest_approach(x0, t0, around_iso, span=5.0):
    """Time (TDB) and distance (au) of the closest approach to the Earth near a date."""
    tt = Time(around_iso, scale="tdb").jd - ephem.JD_REF
    grid = tt + np.arange(-span, span, 0.005)
    X = ephem.propagate(x0[None], t0, grid)[:, 0]
    d = np.linalg.norm(X[:, :3] - ephem.body_state("earth", grid)[:, :3], axis=1)
    k = np.argmin(d)
    return grid[k], d[k]


def fit_as_of(obs, submitted, cut_iso, x_guess, t_guess, sigma_floor=0.1):
    """The orbit fitted to the observations submitted before cut_iso (UTC).

    Weights: a floor on the uncertainties and the sqrt(N/4) down-weighting of large nightly
    batches (Vereš et al. 2017); outliers rejected automatically (Carpino et al. 2003).
    """
    O = obs.before(Time(cut_iso, scale="utc").jd, submitted=submitted).with_floor(sigma_floor).deweight_nights()
    x_start = ephem.propagate(x_guess[None], t_guess, [O.t[0]])[0, 0]
    return O, fit.fit_orbit_robust(x_start, O.t[0], O)


def bplane_sequence(obs, submitted, cuts, truth, t_ca, n_mc, cache, rng, sigma_floor=0.1):
    """For each cut-off date: the fit, a Monte Carlo encounter, and the linearised b-plane.

    Results are cached in `cache` (an .npz per date), so the notebook re-runs quickly.
    """
    t0, x0, _ = truth
    cache = pathlib.Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for cut in cuts:
        path = cache / f"bplane_{cut}_n{n_mc}_floor{sigma_floor}.npz"
        if path.exists():
            d = dict(np.load(path))
        else:
            O, f = fit_as_of(obs, submitted, cut, x0, t0, sigma_floor)
            mean, C, b_e = encounter.linear_bplane(f.state, f.cov, f.t, t_ca)
            e = encounter.encounter(fit.sample_gaussian(f, n_mc, rng), f.t, t_ca)
            d = dict(nobs=len(O), nrej=int(np.sum(~f.used)), arc=float(O.t[-1] - O.t[0]), chi2=f.chi2,
                     dof=int(2 * f.used.sum() - 6),
                     xi=e.xi.astype(np.float32), zeta=e.zeta.astype(np.float32), impact=e.impact,
                     b_earth=e.b_earth.astype(np.float32),
                     lin_mean=mean, lin_cov=C, lin_b=b_e,
                     ip_lin=encounter.impact_probability_gaussian(mean, C, b_e))
            np.savez_compressed(path, **d)
        d["cut"] = cut
        d["ip_mc"] = float(np.mean(d["impact"]))
        n = len(d["impact"])
        # 95% upper limit when no sample hits (rule of three), else the binomial standard error
        d["ip_mc_err"] = 3.0 / n if d["impact"].sum() == 0 else float(np.sqrt(d["ip_mc"] * (1 - d["ip_mc"]) / n))
        out.append(d)
    return out


# --- figures --------------------------------------------------------------------

def plot_bplane(ax, d, xi_lim_km, zeta_lim_km, title="", s=2, truth=None):
    """A b-plane cloud (km) with the Earth's capture disc (shaded) and the Earth (outline).

    The cloud is thousands of km wide but up to millions of km long, so the two axes have
    different scales and the circles appear as ellipses.
    """
    from matplotlib.patches import Ellipse
    xi, zeta = d["xi"] * KM / 1e3, d["zeta"] * KM / 1e6
    hit = d["impact"].astype(bool)
    ax.scatter(xi[~hit], zeta[~hit], s=s, color=MUTED, alpha=0.35, lw=0, rasterized=True)
    ax.scatter(xi[hit], zeta[hit], s=s + 6, color="#e34948", lw=0, rasterized=True)
    b = float(np.median(d["b_earth"])) * KM
    ax.add_patch(Ellipse((0, 0), 2 * b / 1e3, 2 * b / 1e6, color=EARTH, alpha=0.6, lw=0))
    if truth is not None:
        ax.plot(truth[0] / 1e3, truth[1] / 1e6, marker="*", color=INK, ms=11, mec="white", ls="none")
    ax.set_xlim(-xi_lim_km / 1e3, xi_lim_km / 1e3)
    ax.set_ylim(-zeta_lim_km / 1e6, zeta_lim_km / 1e6)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4, symmetric=True))
    ax.set_xlabel(r"$\xi$ (thousand km)")
    ax.set_ylabel(r"$\zeta$ (million km)")
    ax.set_title(title, fontsize=10)
