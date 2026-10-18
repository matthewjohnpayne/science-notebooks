"""Synthetic observations with idealised, known uncertainties.

To demonstrate how orbital uncertainty behaves, it is often clearest to observe a realistic orbit
(for example a real asteroid's, taken as the truth) with simulated observations whose errors are
exactly as stated, rather than to wrestle with real data and their imperfectly known errors.
"""
import numpy as np

from . import ephem, frames, predict
from .frames import ARCSEC, AU_KM as KM
from .mpc import Observations, observer_states


def _sky_motion_axes(state, t, observer):
    """Unit vectors along and across the apparent motion on the sky, seen from `observer`."""
    los = state[:3] - observer[:3]
    los /= np.linalg.norm(los)
    v = state[3:] - observer[3:]
    along = v - (v @ los) * los
    along /= np.linalg.norm(along)
    return along, np.cross(los, along)


def astrometry(x_true, t0, jd_utc, stn, sigma_arcsec, rng, trk=None):
    """Simulated optical astrometry of orbit x_true (state at t0) from ground stations.

    Each position has independent Gaussian errors of sigma_arcsec in RA*cos(Dec) and in Dec.
    jd_utc: UTC Julian dates; stn: one station code or one per date. Returns Observations.
    """
    jd_utc = np.atleast_1d(np.asarray(jd_utc, float))
    stn = np.broadcast_to(np.asarray(stn), jd_utc.shape).copy()
    site = observer_states(stn, jd_utc)
    ra, dec, _ = predict.predict(x_true[None], t0, frames.utc_jd_to_tdb(jd_utc), site)
    ra, dec = ra[0], dec[0]
    e = rng.normal(0, sigma_arcsec, (len(jd_utc), 2)) * ARCSEC
    n = len(jd_utc)
    return Observations(
        jd_utc=jd_utc, ra=np.mod(ra + e[:, 0] / np.cos(dec), 2 * np.pi), dec=dec + e[:, 1],
        sig_ra=np.full(n, float(sigma_arcsec)), sig_dec=np.full(n, float(sigma_arcsec)), stn=stn,
        trk=np.asarray(trk if trk is not None else [f"sim{k}" for k in range(n)]),
        mag=np.full(n, np.nan), band=np.full(n, ""), obsid=np.full(n, ""))


def astrometry_from_space(x_true, t0, jd_utc, geo_km, sigma_arcsec, rng, stn="SPACE", trk=None):
    """Simulated astrometry of orbit x_true (state at t0) from a spacecraft at geocentric positions
    geo_km (M, 3), with independent Gaussian errors of sigma_arcsec. Returns Observations."""
    jd_utc = np.atleast_1d(np.asarray(jd_utc, float))
    n = len(jd_utc)
    tmp = Observations(jd_utc=jd_utc, ra=np.zeros(n), dec=np.zeros(n), sig_ra=np.ones(n), sig_dec=np.ones(n),
                       stn=np.full(n, stn), trk=np.full(n, ""), mag=np.full(n, np.nan), band=np.full(n, ""),
                       obsid=np.full(n, ""), geo_km=np.asarray(geo_km, float))
    ra, dec, _ = predict.predict(x_true[None], t0, tmp.t, tmp.observer)
    e = rng.normal(0, sigma_arcsec, (n, 2)) * ARCSEC
    tmp.ra = np.mod(ra[0] + e[:, 0] / np.cos(dec[0]), 2 * np.pi)
    tmp.dec = dec[0] + e[:, 1]
    tmp.sig_ra[:] = sigma_arcsec
    tmp.sig_dec[:] = sigma_arcsec
    tmp.trk = np.asarray(trk if trk is not None else np.full(n, "space"))
    return tmp


def survey_history(x_true, t0, years, stn, sigma_arcsec, rng, nights=(-10, 0, 10), per_night=3, spacing_h=0.5):
    """A simulated observing history: for each year, tracklets of `per_night` positions spacing_h hours
    apart on nights at the given offsets (days) from opposition, each taken when the asteroid is
    highest in a dark sky at station `stn`."""
    jd, trk = [], []
    for y in years:
        t_opp = opposition(x_true, t0, y)
        for dn in nights:
            j0, _ = best_time_from(x_true, t0, t_opp + dn, stn)
            for k in range(per_night):
                jd.append(j0 + (k - (per_night - 1) / 2) * spacing_h / 24)
                trk.append(f"{y}_{dn:+d}")
    return astrometry(x_true, t0, np.array(jd), stn, sigma_arcsec, rng, trk=np.array(trk))


def opposition(state, t0, year):
    """TDB time near opposition (largest solar elongation seen from the Earth) in a calendar year."""
    from astropy.time import Time
    days = Time(f"{year}-01-01", scale="tdb").jd - ephem.JD_REF + np.arange(0, 366, 1.0)
    X = ephem.propagate(state[None], t0, days)[:, 0]
    e, s = ephem.body_state("earth", days), ephem.body_state("sun", days)
    a = X[:, :3] - e[:, :3]
    b = s[:, :3] - e[:, :3]
    elong = np.arccos(np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1))
    return days[np.argmax(elong)]


def best_time_from(state, t0, t_day, stn):
    """UTC Julian date, within half a day of t_day (TDB), when the asteroid is highest in a dark sky at
    station `stn`; also returns its altitude (degrees)."""
    jd = frames.tdb_to_utc_jd(t_day) + np.arange(-0.5, 0.5, 1 / 288)
    t = frames.utc_jd_to_tdb(jd)
    site = observer_states([stn] * len(jd), jd)
    zen = site[:, :3] - ephem.body_state("earth", t)[:, :3]
    zen /= np.linalg.norm(zen, axis=1)[:, None]
    X = ephem.propagate(state[None], t0, t)[:, 0]
    u = X[:, :3] - site[:, :3]
    u /= np.linalg.norm(u, axis=1)[:, None]
    sun = ephem.body_state("sun", t)[:, :3] - site[:, :3]
    sun /= np.linalg.norm(sun, axis=1)[:, None]
    alt = np.degrees(np.arcsin(np.sum(u * zen, 1)))
    dark = np.degrees(np.arcsin(np.sum(sun * zen, 1))) < -18
    k = np.argmax(np.where(dark, alt, -90))
    return jd[k], alt[k]


def diameter_km(H, albedo=0.1):
    """Diameter from absolute magnitude H for an assumed geometric albedo (D = 1329 km 10^(-H/5) / sqrt(p))."""
    return 1329.0 / np.sqrt(albedo) * 10 ** (-H / 5)


def occultation(x_true, t0, jd_utc, stn, sigma_along_km, sigma_across_km, star_mas, rng):
    """One simulated occultation measurement of an asteroid on orbit x_true (state at t0).

    The measurement is the asteroid's astrometric position seen from `stn` at jd_utc, with an
    error ellipse sigma_along_km along the asteroid's apparent motion (set by the timing) and
    sigma_across_km across it (set by how many chords there are), plus the star's position
    error star_mas (mas, circular). Returns an orbunc Observations object holding it.
    """
    site = observer_states([stn], [jd_utc])
    t = frames.utc_jd_to_tdb(jd_utc)
    ra, dec, dist = predict.predict(x_true[None], t0, [t], site)
    X = ephem.propagate(x_true[None], t0, [t])[0, 0]
    along, across = _sky_motion_axes(X, t, site[0])
    east = np.array([-np.sin(ra[0, 0]), np.cos(ra[0, 0]), 0])
    north = np.cross(frames.radec_to_unit(ra[0, 0], dec[0, 0]), east)
    R = np.array([[along @ east, across @ east], [along @ north, across @ north]])   # (E, N) <- (along, across)
    km_to_arcsec = 1 / (dist[0, 0] * KM) / frames.ARCSEC
    C = R @ np.diag([sigma_along_km ** 2, sigma_across_km ** 2]) @ R.T * km_to_arcsec ** 2 + np.eye(2) * (star_mas / 1e3) ** 2
    err = rng.multivariate_normal([0, 0], C) * frames.ARCSEC
    sa, sd = np.sqrt(C[0, 0]), np.sqrt(C[1, 1])
    return Observations(
        jd_utc=np.array([jd_utc]), ra=np.array([ra[0, 0] + err[0] / np.cos(dec[0, 0])]), dec=np.array([dec[0, 0] + err[1]]),
        sig_ra=np.array([sa]), sig_dec=np.array([sd]), stn=np.array([stn]), trk=np.array(["simulated"]),
        mag=np.array([np.nan]), band=np.array([""]), obsid=np.array([""]), corr=np.array([C[0, 1] / (sa * sd)]),
        kind=np.array(["occultation"]))


