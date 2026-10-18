"""ASSIST ephemeris files and a vectorised N-body propagator.

Units and frame throughout the toolkit: au, days, barycentric ICRF (equatorial),
time `t` in TDB days since J2000 (JD 2451545.0 TDB), which is ASSIST's convention.
"""
import os
import pathlib
import urllib.request

import assist
import numpy as np
import rebound

JD_REF = 2451545.0
PLANETS = "linux_p1550p2650.440"
ASTEROIDS = "sb441-n16.bsp"
URLS = {
    PLANETS: "https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440/linux_p1550p2650.440",
    ASTEROIDS: "https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n16.bsp",
}

_EPHEM = None


def data_dir():
    """Directory for downloaded data: $ORBUNC_DATA, else ~/.orbunc."""
    d = pathlib.Path(os.environ.get("ORBUNC_DATA", pathlib.Path.home() / ".orbunc"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find(name):
    """Look for an ephemeris file in $ASSIST_DIR, then in data_dir()."""
    for d in [os.environ.get("ASSIST_DIR"), data_dir()]:
        if d and (pathlib.Path(d) / name).exists():
            return pathlib.Path(d) / name
    return None


def get_ephem(download=True):
    """Return a cached assist.Ephem, downloading the two JPL files (about 720 MB) if needed."""
    global _EPHEM
    if _EPHEM is None:
        paths = []
        for name in (PLANETS, ASTEROIDS):
            p = _find(name)
            if p is None:
                if not download:
                    raise FileNotFoundError(f"{name} not found in $ASSIST_DIR or {data_dir()}")
                p = data_dir() / name
                print(f"Downloading {URLS[name]} to {p} ...")
                urllib.request.urlretrieve(URLS[name], p)
            paths.append(str(p))
        _EPHEM = assist.Ephem(*paths)
    return _EPHEM


def body_state(name, t):
    """Barycentric state (len(t), 6) of a solar-system body ('sun', 'earth', ...) at TDB times t."""
    ephem = get_ephem()
    t = np.atleast_1d(t)
    out = np.empty((len(t), 6))
    for k, tk in enumerate(t):
        p = ephem.get_particle(name, float(tk))
        out[k] = p.x, p.y, p.z, p.vx, p.vy, p.vz
    return out


GM_EARTH = 0.01720209895 ** 2 / 332946.0487
R_EARTH_AU = 6378.137 / 149597870.7


def strikes_earth(states, t0, times, margin=1.05):
    """True for states whose geocentric two-body (Kepler) path meets the Earth within `times`.

    Uses the geocentric perigee distance and the time of perigee passage, so it is exact for
    two-body motion about the Earth, and a good guide anywhere the Earth's pull dominates.
    """
    e = body_state("earth", t0)[0]
    r, v = states[:, :3] - e[:3], states[:, 3:] - e[3:]
    rn = np.linalg.norm(r, axis=1)
    rv = np.sum(r * v, axis=1)
    hvec = np.cross(r, v)
    h2 = np.sum(hvec * hvec, axis=1)
    E = 0.5 * np.sum(v * v, axis=1) - GM_EARTH / rn
    ecc = np.sqrt(np.maximum(1 + 2 * E * h2 / GM_EARTH ** 2, 0))
    q = h2 / GM_EARTH / (1 + ecc)
    R = margin * R_EARTH_AU
    hit = (rn < R) | (q < R)
    if not hit.any():
        return hit
    # time from t0 to perigee, from the straight-line or Kepler approach; only orbits heading
    # inwards (rv < 0) can reach perigee in the future, outwards ones only in the past
    tmin, tmax = np.min(times) - t0, np.max(times) - t0
    with np.errstate(all="ignore"):
        a = -GM_EARTH / (2 * E)
        # hyperbolic (E > 0) and elliptic anomalies
        cosh_or_cos = (1 - rn / a) / ecc
        F = np.where(E > 0, np.arccosh(np.maximum(cosh_or_cos, 1)), np.arccos(np.clip(cosh_or_cos, -1, 1)))
        n = np.sqrt(GM_EARTH / np.abs(a) ** 3)
        M = np.where(E > 0, ecc * np.sinh(F) - F, F - ecc * np.sin(F))
        dt_peri = np.sign(rv) * -1 * M / n           # time until perigee (negative: in the past)
    reach = (rn < R) | ((dt_peri >= tmin) & (dt_peri <= tmax)) | ((E < 0) & (tmax - tmin > 2 * np.pi / n))
    return hit & np.nan_to_num(reach, nan=True).astype(bool)


def _run(states, t0, times, check_dt=1.0, r_check=0.1):
    """Integrate in one direction through `times` (sorted away from t0); returns (len(times), N, 6).

    Every check_dt days, particles within r_check au of the Earth are tested with
    strikes_earth over the next interval; those that would hit are removed from the run and
    their later states are NaN.
    """
    sim = rebound.Simulation()
    sim.t = t0
    for s in states:
        sim.add(x=s[0], y=s[1], z=s[2], vx=s[3], vy=s[4], vz=s[5])
    extras = assist.Extras(sim, get_ephem())
    # ASSIST leaves IAS15 in its older "global" step-size mode, which takes ~1e5 times more steps
    # near the Earth; "prs23" (Pham, Rein & Spiegel 2024) gives the same positions far faster.
    sim.ri_ias15.adaptive_mode = "prs23"
    n = len(states)
    alive = list(range(n))
    out = np.full((len(times), n, 6), np.nan)
    xyz = np.empty((n, 3))
    vel = np.empty((n, 3))
    sign = 1.0 if len(times) == 0 or times[-1] >= t0 else -1.0

    def current():
        m = len(alive)
        sim.serialize_particle_data(xyz=xyz[:m], vxvyvz=vel[:m])
        return np.concatenate([xyz[:m], vel[:m]], axis=1)

    def remove_strikers(t_next):
        X = current()
        geo = np.linalg.norm(X[:, :3] - body_state("earth", sim.t)[0, :3], axis=1)
        near = np.where(geo < r_check)[0]
        if len(near):
            hit = strikes_earth(X[near], sim.t, np.array([sim.t, t_next]))
            for idx in sorted(near[hit], reverse=True):
                sim.remove(index=int(idx))
                del alive[idx]

    for k, t in enumerate(times):
        while alive and sign * (t - sim.t) > check_dt:
            remove_strikers(sim.t + sign * 1.5 * check_dt)
            if alive:
                sim.integrate(sim.t + sign * check_dt, exact_finish_time=0)
        if not alive:
            break
        remove_strikers(float(t))
        if not alive:
            break
        sim.integrate(float(t))
        out[k, alive] = current()
    del extras
    return out


def propagate(states, t0, times):
    """Propagate test particles under ASSIST forces (Sun, planets, Moon, 16 large asteroids, GR).

    states : (N, 6) barycentric ICRF states at TDB time t0.
    times  : (M,) TDB times, before or after t0, in any order.
    Returns (M, N, 6).

    IAS15 picks one step size for all particles in a run, so particles close to the Earth
    (which need short steps) are propagated in separate runs from the rest. Orbits whose
    geocentric two-body path reaches the Earth's surface within the requested times are not
    integrated (they would end on the ground) and are returned as NaN.
    """
    states = np.atleast_2d(np.asarray(states, float))
    times = np.atleast_1d(np.asarray(times, float))
    hit = strikes_earth(states, t0, np.append(times, t0))
    if hit.any():
        out = np.full((len(times), len(states), 6), np.nan)
        out[:, ~hit] = propagate(states[~hit], t0, times)
        return out
    geo = np.linalg.norm(states[:, :3] - body_state("earth", t0)[0, :3], axis=1)
    band = np.digitize(geo, [0.003, 0.03])
    if len(np.unique(band)) > 1:
        out = np.empty((len(times), len(states), 6))
        for b in np.unique(band):
            k = band == b
            out[:, k] = _propagate(states[k], t0, times)
        return out
    return _propagate(states, t0, times)


def _propagate(states, t0, times):
    out = np.empty((len(times), len(states), 6))
    if len(states) == 0:
        return out
    fwd = np.where(times >= t0)[0]
    bwd = np.where(times < t0)[0]
    if len(fwd):
        order = fwd[np.argsort(times[fwd])]
        out[order] = _run(states, t0, times[order])
    if len(bwd):
        order = bwd[np.argsort(-times[bwd])]
        out[order] = _run(states, t0, times[order])
    return out
