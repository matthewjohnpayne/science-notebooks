"""Astrometric positions predicted from orbits, residuals and chi-squared."""
import numpy as np

from .ephem import body_state, propagate
from .frames import C_AU_DAY, GM_SUN, unit_to_radec, wrap_pi


def _sun_accel(r, sun):
    d = r - sun
    return -GM_SUN * d / np.linalg.norm(d, axis=-1, keepdims=True) ** 3


def light_time_shift(states, sun, dt):
    """Shift states by dt (small, days) with a second-order Taylor step under solar gravity."""
    r, v = states[..., :3], states[..., 3:]
    a = _sun_accel(r, sun)
    dt = np.asarray(dt)[..., None]
    return np.concatenate([r + v * dt + 0.5 * a * dt ** 2, v + a * dt], axis=-1)


def topocentric(states_at_t, sun, observer_pos):
    """Light-time-corrected astrometric line of sight from an observer.

    states_at_t : (..., 6) object states at the observation (reception) time.
    Returns (unit vector, distance), the direction to where the object was when the light left it.
    """
    r = states_at_t
    tau = np.zeros(r.shape[:-1])
    for _ in range(3):
        emit = light_time_shift(r, sun, -tau)
        rho = emit[..., :3] - observer_pos
        dist = np.linalg.norm(rho, axis=-1)
        tau = dist / C_AU_DAY
    return rho / dist[..., None], dist


def predict(states, t0, obs_t, observer):
    """RA, Dec (radians) and distance of each orbit at each observation.

    states   : (N, 6) barycentric ICRF states at TDB time t0.
    obs_t    : (M,) TDB times of observation;  observer : (M, 6) or (M, 3) barycentric positions.
    Returns arrays of shape (N, M).
    """
    obs_t = np.atleast_1d(obs_t)
    uniq, inv = np.unique(obs_t, return_inverse=True)
    prop = propagate(states, t0, uniq)[inv]                       # (M, N, 6)
    sun = body_state("sun", uniq)[inv][:, None, :3]               # (M, 1, 3)
    u, dist = topocentric(prop, sun, np.asarray(observer)[:, None, :3])
    ra, dec = unit_to_radec(u)
    return ra.T, dec.T, dist.T


def residuals(states, t0, obs):
    """Observed minus computed (N, M, 2) in radians, as (RA*cos Dec, Dec)."""
    ra, dec, _ = predict(states, t0, obs.t, obs.observer)
    return np.stack([wrap_pi(obs.ra - ra) * np.cos(obs.dec), obs.dec - dec], axis=-1)


def chi2(states, t0, obs):
    res = residuals(states, t0, obs)
    return np.sum(obs.whiten(res) ** 2, axis=(-1, -2))
