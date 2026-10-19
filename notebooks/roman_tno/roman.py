"""Helpers for the Roman TNO notebook: the survey, synthetic objects and observations, and the
uncertainty seen from two vantage points.

The orbit machinery is in the `orbunc` toolkit at the top of this repository.
"""
import numpy as np
from astropy.time import Time

from orbunc import ephem, fit, frames, simulate, spacecraft

KM = frames.AU_KM
JD = ephem.JD_REF
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
ROMAN_C, EARTH_C = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#8a8984"

# Galactic Bulge Time Domain Survey (overguide design, ROTAC 2025 Appendix C.3, Table 9; the
# RA/Dec conversion is ours). Field centres are provisional to ~0.1-0.2 deg.
FIELDS = {  # name: (RA, Dec) in degrees
    "1": (267.3354, -29.9149), "2": (267.5772, -29.5640), "3": (267.8173, -29.2127),
    "4": (268.0558, -28.8610), "5": (268.2928, -28.5089), "GC": (266.5270, -29.0013)}
SPRING_2027 = ("2027-02-13", "2027-04-25")   # provisional schedule (STScI RDox)
AUTUMN_2027 = ("2027-08-14", "2027-10-30")
CADENCE_MIN = 12.1                           # F146 cadence per field, minutes
WFI_DEG = (0.8, 0.4)                         # approximate footprint of one pointing (deg)


def precision_mas(mag_f146):
    """Approximate astrometric precision (mas) of one 66-s F146 exposure of a point source: 87 mas / SNR
    with a 1.1 mas floor (McKinnon & van der Marel 2026), with SNR = 100 at F146 = 21.2 AB (STScI)
    scaled in proportion to flux (the background-limited regime appropriate to faint sources)."""
    snr = 100 * 10 ** (-0.4 * (mag_f146 - 21.2))
    return np.hypot(87.0 / snr, 1.1)


def roman_trajectory():
    return spacecraft.roman()


def observer(traj, t_tdb, who):
    """Barycentric position (au) of 'Roman', 'Earth' (its centre) at TDB time(s) t."""
    t = np.atleast_1d(t_tdb)
    e = ephem.body_state("earth", t)[:, :3]
    return e + traj(t + JD) / KM if who == "Roman" else e


def place(traj, t_tdb, ra_deg, dec_deg, r_helio, a, incl_deg):
    """A synthetic object on Roman's line of sight (ra, dec) at time t: heliocentric distance r_helio,
    semi-major axis a (speed from the vis-viva equation), moving perpendicular to the Sun-object line
    on a prograde orbit inclined by about incl_deg to the ecliptic. Returns (barycentric state, rho)."""
    o = observer(traj, t_tdb, "Roman")[0]
    sun = ephem.body_state("sun", t_tdb)[0]
    u = frames.radec_to_unit(np.radians(ra_deg), np.radians(dec_deg))
    w = o - sun[:3]
    b = w @ u
    rho = -b + np.sqrt(b * b - (w @ w - r_helio ** 2))
    r = w + rho * u
    v = np.sqrt(frames.GM_SUN * (2 / r_helio - 1 / a))
    z = frames.EC_TO_EQ @ np.array([0.0, 0.0, 1.0])
    t1 = np.cross(z, r)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(r, t1)
    t2 /= np.linalg.norm(t2)
    th = np.radians(incl_deg)
    return np.r_[r + sun[:3], v * (np.cos(th) * t1 + np.sin(th) * t2) + sun[3:]], rho


def observe(traj, x, t_x, start_iso, days, sigma_exp_mas, rng, per_bin=5):
    """Simulated Roman astrometry: one exposure every CADENCE_MIN minutes for `days` days from
    start_iso (UTC), binned `per_bin` at a time (so sigma per binned point = sigma_exp/sqrt(per_bin))."""
    step = CADENCE_MIN * per_bin / 1440
    jd = Time(start_iso, scale="utc").jd + np.arange(0, days, step)
    geo = traj(Time(jd, format="jd", scale="utc").tdb.jd)
    return simulate.astrometry_from_space(x, t_x, jd, geo, sigma_exp_mas / 1e3 / np.sqrt(per_bin), rng,
                                          stn="Roman")


def fit_from_truth(O, x_true, t_true):
    x0 = ephem.propagate(x_true[None], t_true, [O.t[0]])[0, 0]
    return fit.fit_orbit(x0, O.t[0], O)


def apparent_frame(x_mean, t, obs_pos, obs_pos_later):
    """Unit vectors (along the apparent motion, across it, line of sight) of an object seen from an
    observer, from the mean orbit at t and t + 0.01 d."""
    X = ephem.propagate(x_mean[None], t, [t, t + 0.01])[:, 0, :3]
    u0 = X[0] - obs_pos
    u0 /= np.linalg.norm(u0)
    u1 = X[1] - obs_pos_later
    u1 /= np.linalg.norm(u1)
    along = u1 - u0 - ((u1 - u0) @ u0) * u0
    along /= np.linalg.norm(along)
    return np.array([along, np.cross(u0, along), u0]), np.linalg.norm(X[0] - obs_pos)


def spread(S, t_fit, x_mean, t, traj):
    """1-sigma uncertainty (km, at the object) along and across the apparent motion and along the line
    of sight, as seen from Roman and from the Earth's centre at TDB time t, for orbit samples S
    (states at t_fit). Also the apparent rate (km/s at the object) to convert along-track to timing."""
    P = ephem.propagate(S, t_fit, [t])[0, :, :3]
    Xm = ephem.propagate(x_mean[None], t_fit, [t])[0, 0]
    out = {}
    for who in ("Roman", "Earth"):
        o0, o1 = observer(traj, t, who)[0], observer(traj, t + 0.01, who)[0]
        F, dist = apparent_frame(Xm, t, o0, o1)
        d = (P - P.mean(0)) @ F.T * KM
        X1 = ephem.propagate(Xm[None], t, [t + 0.01])[0, 0, :3]
        rate = np.linalg.norm(((X1 - o1) - (Xm[:3] - o0)) - (((X1 - o1) - (Xm[:3] - o0)) @ F[2]) * F[2]) * KM / 864.0
        out[who] = dict(along=d[:, 0].std(), across=d[:, 1].std(), los=d[:, 2].std(), rate_km_s=rate, cloud=d, dist_au=dist)
    return out


def spread_series(S, t_fit, x_mean, times, traj):
    """As `spread`, at many TDB times at once (one integration). Returns, for 'Roman' and 'Earth',
    arrays over time of the sky-plane uncertainty ellipse's major and minor axes, the 1-sigma
    uncertainty along and across the apparent motion (all km at the object), and the apparent
    rate (km/s at the object)."""
    times = np.asarray(times, float)
    P = ephem.propagate(S, t_fit, times)[..., :3]                       # (M, N, 3)
    Xm = ephem.propagate(x_mean[None], t_fit, np.r_[times, times + 0.01])[:, 0, :3]
    Xm0, Xm1 = Xm[:len(times)], Xm[len(times):]
    out = {}
    for who in ("Roman", "Earth"):
        o0, o1 = observer(traj, times, who), observer(traj, times + 0.01, who)
        res = {k: np.empty(len(times)) for k in ("major", "minor", "along", "across", "rate_km_s")}
        for k in range(len(times)):
            u0 = Xm0[k] - o0[k]
            u0 /= np.linalg.norm(u0)
            rel = (Xm1[k] - o1[k]) - (Xm0[k] - o0[k])
            along = rel - (rel @ u0) * u0
            res["rate_km_s"][k] = np.linalg.norm(along) * KM / 864.0
            along /= np.linalg.norm(along)
            F = np.array([along, np.cross(u0, along)])
            d = (P[k] - P[k].mean(0)) @ F.T * KM
            w = np.linalg.eigvalsh(np.cov(d.T))
            res["minor"][k], res["major"][k] = np.sqrt(w)
            res["along"][k], res["across"][k] = d.std(0)
        out[who] = res
    return out


def sci_ticks(ax, axes="xy"):
    """Label linear-axis ticks in scientific notation: 0, 10^3, -2x10^6, ..."""
    import matplotlib.ticker as mt

    def fmt(v, _):
        if v == 0:
            return "0"
        e = int(np.floor(np.log10(abs(v)) + 1e-9))
        m = v / 10 ** e
        m_txt = "" if np.isclose(abs(m), 1) else (f"{abs(m):.1f}".rstrip("0").rstrip(".") + r"\times")
        return f"${'-' if v < 0 else ''}{m_txt}10^{{{e}}}$"
    for a in axes:
        axis = ax.xaxis if a == "x" else ax.yaxis
        axis.set_major_locator(mt.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        axis.set_major_formatter(mt.FuncFormatter(fmt))


EXPOSURE_S = 66.9                            # F146 exposure (ROTAC 2025)
CADENCE_S = CADENCE_MIN * 60


def shadow_speed_km_s(x, t_x, times, traj, who):
    """Speed (km/s) of an object's shadow across the observer, i.e. its apparent motion on the sky
    times its distance, from the orbit x (state at t_x), at TDB times."""
    times = np.asarray(times, float)
    P = ephem.propagate(x[None], t_x, np.r_[times, times + 0.01])[:, 0, :3]
    o = observer(traj, np.r_[times, times + 0.01], who)
    v0, v1 = P[:len(times)] - o[:len(times)], P[len(times):] - o[len(times):]
    u0 = v0 / np.linalg.norm(v0, axis=1)[:, None]
    rel = v1 - v0
    perp = rel - np.sum(rel * u0, 1)[:, None] * u0
    return np.linalg.norm(perp, axis=1) * KM / 864.0


def capture_probability(duration_s):
    """Chance that an event of the given duration overlaps one of the field's exposures, for an event
    at a random time: (exposure + duration) / cadence, capped at 1."""
    return np.minimum((EXPOSURE_S + np.asarray(duration_s)) / CADENCE_S, 1.0)


def occultation_probability(offset_km, radius_km, sigma_km):
    """Probability that the star is occulted (as seen by the observer) when the predicted centre line
    passes offset_km from it, for an object of radius_km and a Gaussian cross-track uncertainty."""
    from scipy.stats import norm
    return norm.cdf((offset_km + radius_km) / sigma_km) - norm.cdf((offset_km - radius_km) / sigma_km)


def sky_track(x, t_x, times, traj, who, ref=None):
    """Directions (RA, Dec, radians) of the object seen from `who` at TDB times; optionally offsets
    (arcsec, east and north) from reference directions ref = (ra, dec)."""
    P = ephem.propagate(x[None], t_x, times)[:, 0, :3] - observer(traj, times, who)
    ra, dec = frames.unit_to_radec(P)
    if ref is None:
        return ra, dec
    from orbunc import sky
    e, n = sky.offsets(ra, dec, ref[0], ref[1])
    return e, n
