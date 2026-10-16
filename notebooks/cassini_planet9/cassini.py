"""Helper module for the notebook "Planet Nine and the Cassini Ranging Data".

It follows Holman & Payne (2016, AJ 152, 94; HP16): a distant planet ("Planet Nine") exerts a
tidal acceleration on Saturn,

    F = G (M / r_X^3) [ rho - 3 (rho . n) n ]                     (HP16 Eq. 7),

where rho is Saturn's heliocentric position, r_X Planet Nine's distance and n the unit vector
towards it. Saturn's displacement Delta r from its ephemeris position obeys the linearised
Encke equation

    d^2 Delta r / dt^2 = -(mu / rho^3) [ Delta r - 3 (rho_hat . Delta r) rho_hat ] + F

(HP16 Eq. 4 linearised, as in Danby's form of Encke's method; HP16 print the Delta r term with a
plus sign, but expanding mu (rho/rho^3 - r/r^3) to first order gives the minus sign used here),

and the Earth-Saturn range changes by delta R = R_hat . Delta r, with R_hat the unit vector from
the Earth to Saturn. An ephemeris fit would absorb any part of delta R that a change in Saturn's
initial position and velocity can mimic, so we remove that part by linear least squares; what
is left is the signature Planet Nine would leave in the Cassini range residuals.

Everything is linear in F, and F is linear in the symmetric tensor T = I - 3 n n^T, so the
signature for any direction and strength is a combination of six basis responses. That is what
makes all-sky maps fast.

Units: au and days; positions are heliocentric, in the ICRF (equatorial) frame, from JPL Horizons.
"""
import json
import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

GM_SUN = 2.959122082855911e-4          # au^3 / day^2  (k^2)
M_SATURN_SYS = 2.858860e-4             # Saturn system / Sun
M_EARTH = 3.003489e-6                  # Sun = 1
AU_M = 1.495978707e11
OBLIQUITY = np.radians(23.4392911)     # J2000 mean obliquity (IAU 1980), for ecliptic -> equatorial
JD_2004 = 2453005.5                    # 2004 January 1.0


# ------------------------------------------------------------------ ephemerides
def horizons_vectors(command, start, stop, step, path):
    """Heliocentric ICRF positions (au) of a Horizons body, cached to path (.npz)."""
    if os.path.exists(path):
        z = np.load(path)
        return z["jd"], z["xyz"]
    import requests
    p = dict(format="json", COMMAND=f"'{command}'", CENTER="'500@10'", EPHEM_TYPE="'VECTORS'", REF_PLANE="'FRAME'",
             REF_SYSTEM="'ICRF'", START_TIME=f"'{start}'", STOP_TIME=f"'{stop}'", STEP_SIZE=f"'{step}'", VEC_TABLE="'1'",
             OUT_UNITS="'AU-D'", CSV_FORMAT="'YES'", VEC_LABELS="'NO'")
    r = requests.get("https://ssd.jpl.nasa.gov/api/horizons.api", params=p, timeout=120)
    r.raise_for_status()
    txt = r.json()["result"]
    body = txt[txt.index("$$SOE") + 5:txt.index("$$EOE")]
    rows = [l.split(",") for l in body.strip().splitlines()]
    jd = np.array([float(x[0]) for x in rows])
    xyz = np.array([[float(v) for v in x[2:5]] for x in rows])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, jd=jd, xyz=xyz)
    return jd, xyz


class Ephemeris:
    """Saturn (system barycentre) and the Earth (geocentre), heliocentric, interpolated."""

    def __init__(self, jd, saturn, earth):
        self.saturn = CubicSpline(jd, saturn)
        self.earth = CubicSpline(jd, earth)
        self.jd0, self.jd1 = jd[0], jd[-1]

    def range_unit(self, jd):
        d = self.saturn(jd) - self.earth(jd)
        return d / np.linalg.norm(d, axis=-1)[..., None]

    def elongation(self, jd):
        """Sun-Earth-Saturn angle (degrees)."""
        e, s = self.earth(jd), self.saturn(jd)
        u1, u2 = -e, s - e
        c = np.sum(u1 * u2, -1) / np.linalg.norm(u1, axis=-1) / np.linalg.norm(u2, axis=-1)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))


def observation_times(eph, start_jd, end_jd, n, min_elong=20.0):
    """n range-measurement times spread evenly over [start, end], leaving out times when
    Saturn is within min_elong degrees of the Sun (no tracking near solar conjunction)."""
    t = np.linspace(start_jd, end_jd, 20 * n)
    t = t[eph.elongation(t) > min_elong]
    return t[np.linspace(0, len(t) - 1, n).round().astype(int)]


# ------------------------------------------------------------------ the linear model
BASIS = [np.diag([1.0, 0, 0]), np.diag([0, 1.0, 0]), np.diag([0, 0, 1.0]),
         np.array([[0, 1.0, 0], [1, 0, 0], [0, 0, 0]]), np.array([[0, 0, 1.0], [0, 0, 0], [1, 0, 0]]),
         np.array([[0, 0, 0], [0, 0, 1.0], [0, 1, 0]])]


def tensor_coeffs(n):
    """Coefficients of T = I - 3 n n^T on BASIS (n a unit vector)."""
    T = np.eye(3) - 3 * np.outer(n, n)
    return np.array([T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[0, 2], T[1, 2]])


def responses(eph, t_obs, t0):
    """Range responses at the observation times (au):
    forced[b] for F = G * BASIS[b] @ rho with unit tidal parameter (M/r^3 = 1 Msun/au^3), starting
    from Delta r = 0 at t0; and free[k] for the homogeneous solution with unit k-th initial
    condition (Delta r0, Delta v0). Returns (forced 6 x N, free 6 x N)."""
    mu = GM_SUN * (1 + M_SATURN_SYS)
    R = eph.range_unit(t_obs)

    def rhs(t, y, B):
        rho = eph.saturn(t)
        r = np.linalg.norm(rho)
        rh = rho / r
        out = []
        for k in range(len(y) // 6):
            dr, dv = y[6 * k:6 * k + 3], y[6 * k + 3:6 * k + 6]
            acc = -mu / r**3 * (dr - 3 * np.dot(rh, dr) * rh)
            if B is not None and k < len(B):
                acc = acc + GM_SUN * (B[k] @ rho)
            out.append(np.concatenate([dv, acc]))
        return np.concatenate(out)

    def run(y0, B):
        sol = solve_ivp(rhs, (t0, t_obs[-1]), y0, args=(B,), t_eval=t_obs, method="DOP853", rtol=1e-11, atol=1e-30)
        Y = sol.y.reshape(len(y0) // 6, 6, -1)
        return np.einsum("kin,ni->kn", Y[:, :3, :], R)

    forced = run(np.zeros(36), BASIS)
    free = run(np.eye(6).ravel(), None)
    return forced, free


def projector(free):
    """Least-squares projector that removes everything a change of Saturn's initial state can
    absorb: P = I - A (A^T A)^-1 A^T with A = free^T."""
    A = free.T
    Q, _ = np.linalg.qr(A)
    return np.eye(A.shape[0]) - Q @ Q.T


def signature(n, k, forced_perp):
    """Post-fit range signature (metres) for tidal parameter k (Msun/au^3) towards unit vector n."""
    return k * (tensor_coeffs(n) @ forced_perp) * AU_M


def radec_to_unit(ra, dec):
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def ecliptic_orbit_position(a, e, inc, node, argp, nu):
    """Heliocentric position (au) on an orbit with ecliptic elements (radians), rotated to the
    equatorial frame; nu is the true anomaly (array)."""
    r = a * (1 - e**2) / (1 + e * np.cos(nu))
    u = argp + nu
    x = r * (np.cos(node) * np.cos(u) - np.sin(node) * np.sin(u) * np.cos(inc))
    y = r * (np.sin(node) * np.cos(u) + np.cos(node) * np.sin(u) * np.cos(inc))
    z = r * np.sin(u) * np.sin(inc)
    ce, se = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    return np.stack([x, ce * y - se * z, se * y + ce * z], -1)
