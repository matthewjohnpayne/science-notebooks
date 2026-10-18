"""Close encounters with the Earth: the target plane (b-plane) and impacts.

The b-plane (Öpik; Valsecchi et al. 2003) is the plane through the Earth's centre perpendicular to
the asteroid's incoming geocentric velocity. Where the incoming asymptote of the hyperbolic
geocentric orbit crosses it is the point B; the asteroid hits the Earth if |B| is less than the
capture radius b_E = R_E sqrt(1 + v_esc^2 / v_inf^2), which allows for gravitational focusing.
Coordinates: eta along the incoming velocity; zeta along minus the projection of the Earth's
heliocentric velocity (so zeta measures whether the asteroid arrives early or late); xi completes
the right-handed set (it measures the minimum distance the orbits can reach).
"""
from dataclasses import dataclass

import numpy as np

from .ephem import GM_EARTH, R_EARTH_AU, body_state, propagate


@dataclass
class Encounter:
    xi: np.ndarray        # b-plane coordinates (au)
    zeta: np.ndarray
    b_earth: np.ndarray   # capture radius for each orbit (au)
    v_inf: np.ndarray     # geocentric hyperbolic excess speed (au/day)
    t_ca: np.ndarray      # time of closest approach (TDB days since J2000)
    d_min: np.ndarray     # sampled minimum geocentric distance (au)
    impact: np.ndarray    # True for orbits that strike the Earth

    @property
    def b(self):
        return np.hypot(self.xi, self.zeta)


def bplane(geo, v_earth_helio):
    """b-plane coordinates from geocentric states (N, 6) and the Earth's heliocentric velocity (N, 3)."""
    r, v = geo[:, :3], geo[:, 3:]
    rn = np.linalg.norm(r, axis=1)
    v_inf2 = np.sum(v * v, axis=1) - 2 * GM_EARTH / rn
    v_inf = np.sqrt(np.maximum(v_inf2, 0))
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    hhat = h / hn[:, None]
    evec = np.cross(v, h) / GM_EARTH - r / rn[:, None]
    e = np.linalg.norm(evec, axis=1)
    P = evec / e[:, None]
    Q = np.cross(hhat, P)
    nu_inf = np.arccos(np.clip(-1 / e, -1, 1))
    # incoming velocity direction: opposite to the incoming asymptote's position direction
    u = -(np.cos(nu_inf)[:, None] * P - np.sin(nu_inf)[:, None] * Q)
    B = np.cross(u, hhat) * (hn / v_inf)[:, None]
    vp = v_earth_helio - np.sum(v_earth_helio * u, axis=1)[:, None] * u
    zeta_hat = -vp / np.linalg.norm(vp, axis=1)[:, None]
    xi_hat = np.cross(u, zeta_hat)
    b_earth = R_EARTH_AU * np.sqrt(1 + 2 * GM_EARTH / (R_EARTH_AU * np.maximum(v_inf2, 1e-30)))
    return np.sum(B * xi_hat, axis=1), np.sum(B * zeta_hat, axis=1), b_earth, v_inf


def encounter(states, t0, t_ca, window=15.0, dt=0.25, r_entry=0.02):
    """Propagate orbits through an encounter with the Earth near time t_ca.

    Orbits are propagated (ephem.propagate, which removes orbits that strike the Earth) to a grid
    of times every dt days within +-window of t_ca. Each orbit's b-plane point is computed from
    its geocentric state at the last grid time before closest approach at which it was still at
    least r_entry from the Earth, so that the two-body asymptote describes the incoming orbit well.
    """
    states = np.atleast_2d(states)
    n = len(states)
    times = t_ca - window + dt * np.arange(int(round(2 * window / dt)) + 1)
    hist = propagate(states, t0, times)
    struck = ~np.isfinite(hist[-1, :, 0])
    earth = body_state("earth", times)
    sun = body_state("sun", times)
    geo = hist - earth[:, None, :]
    d = np.linalg.norm(geo[..., :3], axis=-1)
    d_filled = np.where(np.isfinite(d), d, np.inf)
    # for orbits removed on striking the Earth, their last finite state stands for the approach
    last = np.max(np.where(np.isfinite(d), np.arange(len(times))[:, None], -1), axis=0)
    kmin = np.where(struck, np.maximum(last, 0), np.argmin(d_filled, axis=0))
    idx = np.arange(len(times))[:, None]
    ok = (d_filled >= r_entry) & (idx <= kmin[None]) & np.isfinite(d)
    k_entry = np.where(ok.any(axis=0), np.max(np.where(ok, idx, -1), axis=0), np.maximum(last, 0))
    g = geo[k_entry, np.arange(n)]
    ve = (earth - sun)[k_entry, 3:]
    xi, zeta, b_e, v_inf = bplane(g, ve)
    impact = struck | (np.hypot(xi, zeta) < b_e)
    d_min = np.where(struck, 0.0, d_filled[kmin, np.arange(n)])
    return Encounter(xi, zeta, b_e, v_inf, times[kmin], d_min, impact)


def linear_bplane(state, cov, t0, t_ca, h=(1e-8, 1e-10), **kw):
    """Mean (xi, zeta) and 2x2 covariance of the b-plane point, by linear mapping of a fit."""
    step = np.array([h[0]] * 3 + [h[1]] * 3)
    X = np.vstack([state] + [state + step[k] * np.eye(6)[k] for k in range(6)]
                  + [state - step[k] * np.eye(6)[k] for k in range(6)])
    enc = encounter(X, t0, t_ca, **kw)
    p = np.stack([enc.xi, enc.zeta], axis=1)
    J = ((p[1:7] - p[7:13]) / (2 * step)[:, None]).T
    return p[0], J @ cov @ J.T, enc.b_earth[0]


def impact_probability_gaussian(mean, cov, b_earth, n=4_000_000, rng=None):
    """Probability that a 2-D Gaussian in the b-plane falls inside the capture disc (Monte Carlo)."""
    rng = rng or np.random.default_rng(0)
    p = rng.multivariate_normal(mean, cov, size=n)
    return float(np.mean(np.hypot(p[:, 0], p[:, 1]) < b_earth))
