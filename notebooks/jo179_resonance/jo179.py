"""Helper module for the notebook "2010 JO179: A Dwarf Planet in the 21:5 Resonance with Neptune".

It follows Holman, Payne, Fraser et al. (2018, ApJL 855, L6; H18): take the orbit of 2010 JO179,
make clones consistent with its uncertainties, integrate them with the Sun and the four giant
planets, and follow the 21:5 resonant angle with Neptune,

    theta = 21 lambda - 5 lambda_N - 16 varpi      (H18 Eq. 1),

where lambda and varpi are the TNO's mean longitude and longitude of pericentre and lambda_N is
Neptune's mean longitude, all barycentric.

Orbits come from the MPC Orbits API in the mpc_orb JSON format; planet positions from JPL
Horizons. Both are cached in data/ so the notebook can be rerun without a network connection.

Units: au, years and solar masses. Functions that run in worker processes live here, not in the
notebook, so that multiprocessing works on macOS and Windows.
"""
import json
import os

import numpy as np
import rebound

DAYS_PER_YEAR = 365.25
GIANTS = ["Jupiter", "Saturn", "Uranus", "Neptune"]
# Masses of Mercury, Venus, Earth+Moon and Mars (in solar masses), added to the Sun's
M_INNER = 1.660114e-7 + 2.447838e-6 + 3.040432e-6 + 3.227156e-7
P, Q = 21, 5                       # the 21:5 resonance: 21 Neptune orbits for 5 of the TNO
ORDER = P - Q                      # 16


# ------------------------------------------------------------------ data
def fetch_orbit(desig, path):
    """The object's mpc_orb JSON from the MPC Orbits API, saved to path (and read from it if
    it exists)."""
    if os.path.exists(path):
        return json.load(open(path))
    import requests
    r = requests.get("https://data.minorplanetcenter.net/api/get-orb", json={"desig": desig}, timeout=60)
    r.raise_for_status()
    mo = r.json()[0]["mpc_orb"][0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(mo, open(path, "w"), indent=1)
    return mo


def planet_states(jd_tdb, path):
    """Barycentric ecliptic (J2000) states of the Sun and the four giant-planet barycentres at
    the given Julian date, from JPL Horizons via REBOUND; saved to path (and read from it).
    Units au and au/yr."""
    if os.path.exists(path):
        return json.load(open(path))
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    for name in ["Sun"] + GIANTS:
        sim.add(name, date=f"JD{jd_tdb}")
    out = dict(jd_tdb=jd_tdb, source="JPL Horizons via rebound.Simulation.add",
               bodies={name: dict(m=p.m, xyz=list(p.xyz), vxyz=list(p.vxyz))
                       for name, p in zip(["Sun"] + GIANTS, sim.particles)})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def clone_states(car_values, car_cov, n, seed):
    """n heliocentric Cartesian states (au, au/day) drawn from the multivariate normal
    distribution given by the orbit's best-fit state and covariance; row 0 is the best fit."""
    rng = np.random.default_rng(seed)
    x0 = np.asarray(car_values, float)
    draws = rng.multivariate_normal(x0, np.asarray(car_cov, float), size=n)
    return np.vstack([x0, draws])


# ------------------------------------------------------------------ simulation
def make_sim(planets, helio_states, integrator="whfast", dt=0.5):
    """Sun (carrying the inner planets' mass) and the four giant planets, plus test particles at
    the given heliocentric ecliptic states (au, au/day)."""
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    for name in ["Sun"] + GIANTS:
        b = planets["bodies"][name]
        m = b["m"] + (M_INNER if name == "Sun" else 0.0)
        sim.add(m=m, x=b["xyz"][0], y=b["xyz"][1], z=b["xyz"][2], vx=b["vxyz"][0], vy=b["vxyz"][1], vz=b["vxyz"][2], hash=name)
    sun = sim.particles["Sun"]
    for s in helio_states:
        v = np.asarray(s[3:]) * DAYS_PER_YEAR
        sim.add(m=0.0, x=sun.x + s[0], y=sun.y + s[1], z=sun.z + s[2], vx=sun.vx + v[0], vy=sun.vy + v[1], vz=sun.vz + v[2])
    sim.N_active = 5
    sim.move_to_com()
    sim.integrator = integrator
    if integrator == "whfast":
        sim.dt = dt
        sim.ri_whfast.corrector = 11
        sim.ri_whfast.safe_mode = 0
    return sim


def bary_elements(sim):
    """Barycentric osculating elements of Neptune and of every test particle, about the total
    mass of the Sun and planets at the system's barycentre (as H18's barycentric elements).
    Returns (Neptune orbit, list of test-particle orbits)."""
    bary = sim.com(last=5)
    ps = sim.particles
    return ps["Neptune"].orbit(primary=bary), [ps[k].orbit(primary=bary) for k in range(5, sim.N)]


def theta(o, oN, n=0):
    """The 21:5 resonant angle (radians, 0-2pi). n = 0 is H18 Eq. 1; n = 1..15 are the other
    angles of the same resonance, with -(n varpi_N + (16 - n) varpi) in place of -16 varpi."""
    return (P * o.l - Q * oN.l - (n * oN.pomega + (ORDER - n) * o.pomega)) % (2 * np.pi)


# ------------------------------------------------------------------ the long integration
def long_run(args):
    """Integrate a batch of clones for t_end years with WHFast, sampling every `cadence` years.

    In each window of `window` years a clone counts as resonant if its resonant angle stays
    away from 0 deg (the point opposite the libration centre near 180 deg) by more than `gap`
    degrees throughout. Returns, per clone and per window: the resonant flag, the period ratio
    with Neptune (min, max), pericentre (min, max), and mean eccentricity and inclination; and,
    for the clones listed in `keep`, the resonant angle, period ratio, e and i every
    `keep_every` years."""
    planets, states, ids, t_end, cadence, window, gap, keep, keep_every = args
    sim = make_sim(planets, states)
    n = len(states)
    nwin = int(round(t_end / window))
    per_win = int(round(window / cadence))
    flags = np.ones((n, nwin), bool)
    pr_min = np.full((n, nwin), np.inf); pr_max = np.full((n, nwin), -np.inf)
    q_min = np.full((n, nwin), np.inf); q_max = np.full((n, nwin), -np.inf)
    e_mean = np.zeros((n, nwin)); i_mean = np.zeros((n, nwin))
    kept = [j for j, cid in enumerate(ids) if cid in keep]
    stride = int(round(keep_every / cadence))
    series = {ids[j]: dict(theta=[], pr=[], e=[], i=[]) for j in kept}
    g = np.radians(gap)
    step = 0
    for w in range(nwin):
        for k in range(per_win):
            step += 1
            sim.integrate(step * cadence, exact_finish_time=0)
            oN, os_ = bary_elements(sim)
            for j, o in enumerate(os_):
                th = theta(o, oN)
                if th < g or th > 2 * np.pi - g:
                    flags[j, w] = False
                pr = o.P / oN.P
                q = o.a * (1 - o.e)
                pr_min[j, w] = min(pr_min[j, w], pr); pr_max[j, w] = max(pr_max[j, w], pr)
                q_min[j, w] = min(q_min[j, w], q); q_max[j, w] = max(q_max[j, w], q)
                e_mean[j, w] += o.e / per_win; i_mean[j, w] += o.inc / per_win
                if j in kept and step % stride == 0:
                    s = series[ids[j]]
                    s["theta"].append(th); s["pr"].append(pr); s["e"].append(o.e); s["i"].append(o.inc)
    series = {cid: {k: np.array(v, np.float32) for k, v in s.items()} for cid, s in series.items()}
    return dict(ids=list(ids), flags=flags, pr_min=pr_min, pr_max=pr_max, q_min=q_min, q_max=q_max,
                e_mean=e_mean.astype(np.float32), i_mean=i_mean.astype(np.float32), series=series)
