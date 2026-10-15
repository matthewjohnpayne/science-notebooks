"""Helper module for the transit-timing-variation notebook.

Follows M. J. Payne, E. B. Ford & D. Veras (2010), "Transit Timing Variations
for Inclined and Retrograde Exoplanetary Systems", ApJL 712, L86 ("PFV10"):
a transiting hot Jupiter perturbed by an Earth-mass companion on an inclined,
possibly retrograde, orbit.

Units: days, au and solar masses (REBOUND's units). The transiting planet orbits
in the x-y plane and the observer lies far away along +x.
"""
import numpy as np
import rebound

C_AU_PER_DAY = 173.1446326846693
M_JUP = 9.547919e-4        # solar masses (IAU 2015 nominal GM_J / GM_sun)
M_EARTH = 3.003489e-6      # solar masses (IAU 2015 nominal GM_E / GM_sun)
R_SUN_AU = 0.00465047      # nominal solar radius in au
DAY_S = 86400.0
YEAR_D = 365.25

# PFV10's fiducial system. The paper does not state the stellar mass or radius;
# we assume a Sun-like star.
A1 = 0.05                  # au, hot Jupiter on a circular orbit
T_END = 10 * YEAR_D        # "~10 years, equating to ~1,000 transits"


def make_sim(a_ratio, e2, inc2, omega2=0.0, Omega2=0.0, M2=0.0, *, a1=A1, m1=M_JUP, m2=M_EARTH):
    """Star, transiting planet (a1, e = 0, in the x-y plane) and perturber
    (a2 = a_ratio * a1, e2, relative inclination inc2 and angles in radians)."""
    sim = rebound.Simulation()
    sim.units = ("day", "AU", "Msun")
    sim.add(m=1.0)
    sim.add(m=m1, a=a1, e=0.0, inc=0.0, M=0.0)
    sim.add(m=m2, a=a_ratio * a1, e=e2, inc=inc2, omega=omega2, Omega=Omega2, M=M2)
    sim.move_to_com()
    sim.integrator = "ias15"
    return sim


def transit_times(sim, t_end=T_END, r_star=R_SUN_AU, steps_per_orbit=12, light_time=True):
    """Mid-transit times of planet 1, as in PFV10 Section 2.1.

    After each step, test whether the projected star-planet separation Delta
    (in the y-z sky plane) has passed through a minimum with the planet closer
    to the observer than the star; find the time of that minimum by Newton
    iteration on d(Delta^2)/dt = 0; and keep it if Delta < r_star. The observed
    time includes the light travel time, -(r_p . x_hat)/c with r_p barycentric.
    """
    p, s = sim.particles[1], sim.particles[0]
    dt = p.P / steps_per_orbit
    f_of = lambda: (p.y - s.y) * (p.vy - s.vy) + (p.z - s.z) * (p.vz - s.vz)
    times = []
    t = sim.t
    f_old = f_of()
    while t < t_end:
        t_prev, t = t, t + dt
        sim.integrate(t)
        f_new = f_of()
        if f_old < 0 <= f_new and p.x > s.x:
            tt = t_prev + dt * (-f_old) / (f_new - f_old)
            for _ in range(30):
                sim.integrate(tt)
                step = -f_of() / ((p.vy - s.vy) ** 2 + (p.vz - s.vz) ** 2)
                tt += step
                if abs(step) < 1e-10:
                    break
            sim.integrate(tt)
            if np.hypot(p.y - s.y, p.z - s.z) < r_star and tt <= t_end:
                times.append(tt - (p.x / C_AU_PER_DAY if light_time else 0.0))
            sim.integrate(t)
        f_old = f_of()
    return np.array(times)


def ttv(times):
    """Residuals of the transit times from a least-squares linear ephemeris
    t_i = t0 + i P (PFV10's delta t(i)), and the fitted (P, t0)."""
    n = np.arange(len(times), dtype=float)
    A = np.vstack([n, np.ones_like(n)]).T
    coef, *_ = np.linalg.lstsq(A, times, rcond=None)
    return times - A @ coef, coef


def rms_ttv(job):
    """RMS TTV in seconds for one perturber orbit, job = (a_ratio, e2, inc2,
    omega2, Omega2, M2) with angles in radians. NaN if fewer than 10 transits."""
    sim = make_sim(*job)
    tt = transit_times(sim)
    if len(tt) < 10:
        return np.nan
    r, _ = ttv(tt)
    return float(np.sqrt(np.mean(r ** 2)) * DAY_S)


def ttv_series(job):
    """Transit times (days) and TTVs (seconds) for one perturber orbit, job as
    in rms_ttv."""
    tt = transit_times(make_sim(*job))
    r, _ = ttv(tt)
    return tt, r * DAY_S


def mean_periods(job, n_samples=20001):
    """Mean orbital periods (days) of the two planets over the run, from linear
    fits to their unwrapped Jacobi mean longitudes."""
    sim = make_sim(*job)
    ts = np.linspace(0, T_END, n_samples)
    lam = np.empty((n_samples, 2))
    for k, t in enumerate(ts):
        sim.integrate(t)
        o = sim.orbits()
        lam[k] = o[0].l, o[1].l
    lam = np.unwrap(lam, axis=0)
    return tuple(2 * np.pi / np.polyfit(ts, lam[:, j], 1)[0] for j in range(2))


def dominant_period(times, resid):
    """Period (days) of the strongest sinusoid in a TTV series, from a
    least-squares periodogram over periods from 4 transit intervals to the
    full span."""
    span = times[-1] - times[0]
    P1 = span / (len(times) - 1)
    periods = np.geomspace(4 * P1, span, 4000)
    power = []
    for P in periods:
        w = 2 * np.pi / P
        A = np.vstack([np.sin(w * times), np.cos(w * times), np.ones_like(times)]).T
        coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
        power.append(np.hypot(coef[0], coef[1]))
    return periods[int(np.argmax(power))]


# --- Hill stability (Gladman 1993) ----------------------------------------------

def gladman_hill_stable(a_ratio, e2, e1=0.0, m0=1.0, m1=M_JUP, m2=M_EARTH):
    """True if two coplanar prograde orbits satisfy Gladman's (1993) Hill-stability
    criterion in its orbital-element form,

        mu^-3 (mu1 + mu2 / lam^2) (mu1 g1 + mu2 g2 lam)^2 > 1 + 3^(4/3) mu1 mu2 / mu^(4/3),

    with mu_i = m_i / m0, mu = mu1 + mu2, lam = sqrt(a2 / a1) and
    g_i = sqrt(1 - e_i^2). Works on arrays."""
    mu1, mu2 = m1 / m0, m2 / m0
    mu = mu1 + mu2
    lam = np.sqrt(np.asarray(a_ratio, dtype=float))
    g1, g2 = np.sqrt(1 - e1 ** 2), np.sqrt(1 - np.asarray(e2, dtype=float) ** 2)
    lhs = mu ** -3 * (mu1 + mu2 / lam ** 2) * (mu1 * g1 + mu2 * g2 * lam) ** 2
    return lhs > 1 + 3 ** (4 / 3) * mu1 * mu2 / mu ** (4 / 3)


# --- Grids --------------------------------------------------------------------------

def random_angles(n, seed):
    """n sets of (omega, Omega, M), uniform in [0, 2 pi)."""
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, (n, 3))


def cached_grid(path, jobs, func, nproc, recompute=False, chunksize=8):
    """Evaluate func over jobs in parallel, caching the results (float32) in an
    .npz file so the notebook can be re-run without repeating long computations.
    The cache is reused only if it holds the same number of results."""
    import os
    from multiprocessing import Pool
    if os.path.exists(path) and not recompute:
        values = np.load(path)["values"]
        if len(values) == len(jobs):
            return values.astype(float)
    with Pool(nproc) as pool:
        values = np.array(pool.map(func, jobs, chunksize=chunksize), dtype=np.float32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, values=values)
    return values.astype(float)


def median_map(values, shape, n_real):
    """Median over the n_real random orientations of each grid point."""
    return np.nanmedian(np.asarray(values).reshape(*shape, n_real), axis=-1)
