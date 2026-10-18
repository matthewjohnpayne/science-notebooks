"""Projecting sets of orbits onto the sky, onto the ecliptic plane, and into orbit classes."""
import numpy as np

from .frames import ARCSEC, utc_jd_to_tdb
from .mpc import observer_states
from .predict import predict


def sky_positions(states, t0, jd_utc, stn):
    """Astrometric RA, Dec (radians), each (N, M), of N orbits seen from station `stn` at UTC dates."""
    jd_utc = np.atleast_1d(jd_utc)
    obs = observer_states([stn] * len(jd_utc), jd_utc)
    ra, dec, _ = predict(states, t0, utc_jd_to_tdb(jd_utc), obs)
    return ra, dec


def offsets(ra, dec, ra0, dec0, unit=ARCSEC):
    """Offsets (east, north) of positions from (ra0, dec0), by default in arcsec.

    Azimuthal equidistant projection: the distance from the centre is the true angular
    distance, so it stays meaningful even for points far from the centre (it agrees with the
    tangent-plane projection for small offsets).
    """
    cosc = np.clip(np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0), -1, 1)
    c = np.arccos(cosc)
    k = np.where(c > 1e-12, c / np.sin(np.maximum(c, 1e-12)), 1.0)
    x = k * np.cos(dec) * np.sin(ra - ra0)
    y = k * (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0))
    return x / unit, y / unit


def weighted_centre(ra, dec, w=None):
    """Mean direction of a cloud of positions (ra, dec in radians)."""
    u = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)
    m = np.average(u, axis=0, weights=w)
    return np.arctan2(m[1], m[0]) % (2 * np.pi), np.arcsin(m[2] / np.linalg.norm(m))


def linear_sky_covariance(state, cov, t0, jd_utc, stn, h=(1e-8, 1e-10)):
    """2x2 covariance (arcsec^2, east/north) of the predicted position, from a linearised map.

    The state covariance is mapped through the finite-difference Jacobian of (RA cos Dec, Dec)
    with respect to the state, the usual linear approximation behind an ephemeris uncertainty.
    """
    step = np.array([h[0]] * 3 + [h[1]] * 3)
    X = np.vstack([state] + [state + step[k] * np.eye(6)[k] for k in range(6)]
                  + [state - step[k] * np.eye(6)[k] for k in range(6)])
    ra, dec = sky_positions(X, t0, [jd_utc], stn)
    e, n = offsets(ra[:, 0], dec[:, 0], ra[0, 0], dec[0, 0])
    J = np.stack([(e[1:7] - e[7:13]) / (2 * step), (n[1:7] - n[7:13]) / (2 * step)])
    return J @ cov @ J.T, (ra[0, 0], dec[0, 0])


def ellipse(cov2, nsig=3.0, n=200):
    """Points (x, y) on the nsig contour of a 2-D Gaussian with covariance cov2."""
    vals, vecs = np.linalg.eigh(cov2)
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)])
    return vecs @ (nsig * np.sqrt(np.maximum(vals, 0))[:, None] * circle)


def orbit_paths(el, n=241):
    """Heliocentric ecliptic (x, y, z) of two-body orbits from element dicts (bound orbits only).

    el: dict of arrays as returned by frames.elements. Returns array (N, n, 3); NaN where e >= 1.
    """
    a, e = np.atleast_1d(el["a"]), np.atleast_1d(el["e"])
    i, O, w = (np.radians(np.atleast_1d(el[k])) for k in ("i", "node", "argperi"))
    E = np.linspace(0, 2 * np.pi, n)[None]
    with np.errstate(invalid="ignore"):
        xp = a[:, None] * (np.cos(E) - e[:, None])
        yp = a[:, None] * np.sqrt(1 - e[:, None] ** 2) * np.sin(E)
    cO, sO, cw, sw, ci, si = np.cos(O), np.sin(O), np.cos(w), np.sin(w), np.cos(i), np.sin(i)
    P = np.stack([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si], -1)
    Q = np.stack([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si], -1)
    out = xp[..., None] * P[:, None] + yp[..., None] * Q[:, None]
    out[e >= 1] = np.nan
    return out


# Orbit classes, by perihelion q, aphelion Q and semi-major axis a (au), following the
# usual MPC/CNEOS boundaries for near-Earth objects.
CLASSES = ["Atira", "Aten", "Apollo", "Amor", "Mars-crosser", "Main belt", "Outer", "Unbound"]


def orbit_class(el):
    a, e, q = np.atleast_1d(el["a"]), np.atleast_1d(el["e"]), np.atleast_1d(el["q"])
    with np.errstate(invalid="ignore"):
        Q = a * (1 + e)
    cls = np.full(a.shape, "Outer", dtype=object)
    cls[(q >= 1.666) & (a < 3.7)] = "Main belt"
    cls[(q >= 1.3) & (q < 1.666)] = "Mars-crosser"
    cls[(q >= 1.017) & (q < 1.3)] = "Amor"
    cls[(q < 1.017) & (a >= 1)] = "Apollo"
    cls[(a < 1) & (Q >= 0.983)] = "Aten"
    cls[(a < 1) & (Q < 0.983)] = "Atira"
    cls[e >= 1] = "Unbound"
    return cls


def class_probabilities(el, w):
    cls = orbit_class(el)
    return {c: float(np.sum(w[cls == c])) for c in CLASSES}
