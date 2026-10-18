"""Constants, time scales, directions on the sky, and orbital elements."""
import numpy as np
from astropy.time import Time

from .ephem import JD_REF, body_state

GM_SUN = 0.01720209895 ** 2          # au^3 / day^2 (Gaussian constant)
C_AU_DAY = 173.1446326846693         # speed of light, au / day
AU_KM = 149597870.7
R_EARTH_KM = 6378.137
OBLIQUITY = np.radians(84381.448 / 3600)  # J2000 obliquity used by the MPC
ARCSEC = np.pi / 180 / 3600

_c, _s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
EQ_TO_EC = np.array([[1, 0, 0], [0, _c, _s], [0, -_s, _c]])
EC_TO_EQ = EQ_TO_EC.T


# --- time ---------------------------------------------------------------

def utc_jd_to_tdb(jd_utc):
    """UTC Julian date -> TDB days since J2000 (ASSIST's time variable)."""
    return Time(np.asarray(jd_utc), format="jd", scale="utc").tdb.jd - JD_REF


def tdb_to_utc_jd(t):
    return Time(np.asarray(t) + JD_REF, format="jd", scale="tdb").utc.jd


def mjd_tt_to_tdb(mjd):
    return Time(np.asarray(mjd), format="mjd", scale="tt").tdb.jd - JD_REF


# --- directions ---------------------------------------------------------

def radec_to_unit(ra, dec):
    ra, dec = np.asarray(ra), np.asarray(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def unit_to_radec(u):
    u = np.asarray(u)
    r = np.linalg.norm(u, axis=-1)
    return np.mod(np.arctan2(u[..., 1], u[..., 0]), 2 * np.pi), np.arcsin(u[..., 2] / r)


def gnomonic(ra, dec, ra0, dec0):
    """Tangent-plane coordinates (xi, eta), in radians, of (ra, dec) about (ra0, dec0)."""
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cosc
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cosc
    return xi, eta


def wrap_pi(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


# --- frames and elements ------------------------------------------------

def rotate_states(states, R):
    states = np.asarray(states, float)
    return np.concatenate([states[..., :3] @ R.T, states[..., 3:] @ R.T], axis=-1)


def bary_eq_to_helio_ec(states, t):
    """Barycentric ICRF states at TDB time t -> heliocentric J2000-ecliptic states."""
    sun = body_state("sun", t)[0]
    return rotate_states(np.asarray(states) - sun, EQ_TO_EC)


def helio_ec_to_bary_eq(states, t):
    sun = body_state("sun", t)[0]
    return rotate_states(states, EC_TO_EQ) + sun


def elements(states_helio_ec, gm=GM_SUN):
    """Osculating heliocentric elements of (..., 6) states.

    Returns a dict of arrays: a, e, q, i, node, argperi, M (angles in degrees; M only for e < 1)
    and the specific orbital energy E.
    """
    s = np.asarray(states_helio_ec, float)
    r, v = s[..., :3], s[..., 3:]
    rn = np.linalg.norm(r, axis=-1)
    v2 = np.sum(v * v, axis=-1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=-1)
    E = 0.5 * v2 - gm / rn
    a = -gm / (2 * E)
    evec = np.cross(v, h) / gm - r / rn[..., None]
    e = np.linalg.norm(evec, axis=-1)
    q = hn ** 2 / gm / (1 + e)
    inc = np.arccos(np.clip(h[..., 2] / hn, -1, 1))
    nvec = np.stack([-h[..., 1], h[..., 0], np.zeros_like(hn)], axis=-1)
    nn = np.linalg.norm(nvec, axis=-1)
    node = np.mod(np.arctan2(nvec[..., 1], nvec[..., 0]), 2 * np.pi)
    with np.errstate(invalid="ignore", divide="ignore"):   # undefined for circular or equatorial orbits
        argp = np.arccos(np.clip(np.sum(nvec * evec, -1) / (nn * e), -1, 1))
        argp = np.where(evec[..., 2] < 0, 2 * np.pi - argp, argp)
        nu = np.arccos(np.clip(np.sum(evec * r, -1) / (e * rn), -1, 1))
        nu = np.where(np.sum(r * v, -1) < 0, 2 * np.pi - nu, nu)
    with np.errstate(invalid="ignore"):
        Ea = 2 * np.arctan2(np.sqrt(np.clip(1 - e, 0, None)) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
        M = np.where(e < 1, np.mod(Ea - e * np.sin(Ea), 2 * np.pi), np.nan)
    deg = np.degrees
    return dict(a=a, e=e, q=q, i=deg(inc), node=deg(node), argperi=deg(argp), M=deg(M), E=E)


def absolute_magnitude(mag, r_helio, delta, phase_rad, G=0.15):
    """IAU H,G absolute magnitude (Bowell et al. 1989) from an apparent magnitude."""
    p1 = np.exp(-3.33 * np.tan(phase_rad / 2) ** 0.63)
    p2 = np.exp(-1.87 * np.tan(phase_rad / 2) ** 1.22)
    return mag - 5 * np.log10(r_helio * delta) + 2.5 * np.log10((1 - G) * p1 + G * p2)
