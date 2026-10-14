"""Helpers for the resonance-overlap notebook.

Kept in a module (not the notebook) so that multiprocessing workers can import
them on every platform, including macOS and Windows, which spawn rather than fork.

Units: G = 1 and stellar mass 1. The inner planet starts at a = 1, so its
period is 2*pi. Equation references are to Deck, Payne & Holman (2013,
ApJ 774, 129; "DPH13"), arXiv version.
"""
import numpy as np
import rebound

TWO_PI = 2 * np.pi
DAYS_PER_YEAR = 365.25


# --- Simple analytic criteria (DPH13 Section 3) --------------------------------

def overlap_delta_circular(eps):
    """(a2 - a1)/a1 below which first-order resonances overlap, for initially
    circular orbits: 1.46 eps^(2/7)."""
    return 1.46 * np.asarray(eps) ** (2 / 7)


def overlap_period_ratio_circular(eps):
    """The same criterion in averaged period ratio: P2/P1 < 1 + 2.2 eps^(2/7)."""
    return 1 + 2.2 * np.asarray(eps) ** (2 / 7)


def overlap_period_ratio_eccentric(eps, sigma):
    """Eccentric-orbit estimate P2/P1 < 1 + 2.08 (eps sigma)^(1/5), which DPH13
    expect to hold only for sigma >~ 1.33 eps^(3/7)."""
    return 1 + 2.08 * (eps * np.asarray(sigma)) ** 0.2


def hill_delta_circular(eps):
    """(a2 - a1)/a1 above which initially circular orbits are Hill stable: 2.4 eps^(1/3)."""
    return 2.4 * np.asarray(eps) ** (1 / 3)


def resonance_full_width(eps, m):
    """Full width in alpha = a1/a2 of the m+1:m resonance, as a fraction of
    alpha_res, for circular orbits: 4.18 eps^(2/3) m^(1/3)."""
    return 4.18 * eps ** (2 / 3) * np.asarray(m, dtype=float) ** (1 / 3)


def resonance_spacing(m):
    """Fractional spacing in alpha between the m+1:m and m+2:m+1 resonances: about 2/(3 m^2)."""
    return 2 / (3 * np.asarray(m, dtype=float) ** 2)


# --- Separatrices of the reduced Hamiltonian (DPH13 Section 2) -----------------

def resonance_separatrix(eps, m, zeta=1.0, n=600, gamma_max=60.0):
    """Separatrix of the m+1:m resonance in the plane of period ratio and
    weighted eccentricity sigma.

    DPH13 reduce the problem to H' = -1/2 [ (X^2+Y^2)/2 - Gamma' ]^2 - X.
    For Gamma' > 3/2 the saddle X3 is the largest root of X^3 - 2 Gamma' X + 2 = 0,
    and the separatrix crosses Y = 0 at X = -X3 +/- 2/sqrt(X3). Each crossing is
    a point (Phi', Gamma'), which we map back to sigma and period ratio through
    Psi = Q Phi' = g_m R^2 sigma^2 and delta Theta = Q (Phi' - Gamma').

    Returns two (period_ratio, sigma) curves: the inner and outer crossing loci.
    Period ratios are in averaged coordinates. zeta = m1/m2.
    """
    f27 = -0.46 - 0.802 * m          # DPH13 linear fits to the Laplace coefficients
    f31 = 0.87 + 0.802 * m
    R = abs(f27 / f31)
    a_res = (m / (m + 1)) ** (2 / 3)
    Q = (eps ** (2 / 3) * zeta * (f31 ** 2 / (9 * (m + 1) * (1 + zeta) ** 2)) ** (1 / 3)
         * a_res ** (5 / 6) / m * ((R ** 2 + zeta * np.sqrt(a_res)) / (a_res + zeta) ** 5) ** (1 / 3))
    g_m = zeta / (a_res + zeta) / (2 * (R ** 2 + zeta * np.sqrt(a_res)))

    gammas = 1.5 + np.geomspace(1e-4, gamma_max, n)
    curves = []
    for sign in (+1, -1):
        prs, sigmas = [], []
        for G in gammas:
            X3 = np.max(np.real(np.roots([1, 0, -2 * G, 2])))
            Xs = -X3 + sign * 2 / np.sqrt(X3)
            phi_p = Xs ** 2 / 2
            sigma = np.sqrt(Q * phi_p / (g_m * R ** 2))
            dtheta = Q * (phi_p - G)
            s = (zeta / m - dtheta * (a_res + zeta)) / (zeta / m + dtheta * zeta * (1 + zeta / a_res))
            alpha = s ** 2 * a_res
            prs.append(alpha ** -1.5)
            sigmas.append(sigma)
        curves.append((np.array(prs), np.array(sigmas)))
    return curves


# --- Hill stability for arbitrary orbits (Marchal & Bozis 1982; Gladman 1993) ---

def hill_margin(sim):
    """(-2 M / M_*^3) L^2 E divided by its critical value, minus one. Positive
    means the configuration is Hill stable (Gladman 1993, his eq. 21, to second
    order in the planet masses)."""
    m0, m1, m2 = (p.m for p in sim.particles[:3])
    M = m0 + m1 + m2
    Ms = m0 * m1 + m0 * m2 + m1 * m2
    u1, u2 = m1 / m0, m2 / m0
    L = np.linalg.norm(sim.angular_momentum())
    lhs = -2 * M / Ms ** 3 * L ** 2 * sim.energy()
    crit = 1 + 3 ** (4 / 3) * u1 * u2 / (u1 + u2) ** (4 / 3) - u1 * u2 * (11 * u1 + 7 * u2) / (3 * (u1 + u2) ** 2)
    return lhs / crit - 1


def hill_period_ratio(eps, e1=0.0, M1=np.pi / 2, lo=1.0005, hi=2.5):
    """Smallest initial period ratio at which the pair is Hill stable, by bisection."""
    f = lambda pr: hill_margin(two_planet_sim(eps, pr, e1=e1, M1=M1))
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


# --- Numerical experiments ------------------------------------------------------

def two_planet_sim(eps, period_ratio, e1=0.0, e2=0.0, M1=np.pi / 2, mass_ratio=1.0, steps_per_orbit=50):
    """Star plus two planets with total mass eps (relative to the star) and
    m1/m2 = mass_ratio. All angles are zero except the inner mean anomaly M1.
    WHFast (Wisdom-Holman) without a symplectic corrector: DPH13 used one, but in
    REBOUND it cannot be combined with the variational equations efficiently, and it
    improves energy errors rather than which orbits are chaotic."""
    m2 = eps / (1 + mass_ratio)
    m1 = eps - m2
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=1.0)
    sim.add(m=m1, a=1.0, e=e1, M=M1)
    sim.add(m=m2, P=TWO_PI * period_ratio, e=e2)
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.dt = TWO_PI / steps_per_orbit
    sim.exit_max_distance = 20.0
    return sim


def tangent_log_length(sim):
    """ln of the length of the variational (tangent) vector, including the growth
    REBOUND has already removed by rescaling it to avoid overflow."""
    d2 = sum(p.x ** 2 + p.y ** 2 + p.z ** 2 + p.vx ** 2 + p.vy ** 2 + p.vz ** 2 for p in sim.particles_var)
    return 0.5 * np.log(d2) + sim.var_config[0]._lrescale


def lyapunov_time(args):
    """log10 of the estimated Lyapunov time in units of the initial inner period.

    args = (eps, period_ratio, e1, M1, t_days)

    Follows DPH13 Section 4. The outer planet's period is one year and the run
    lasts t_days days. A random unit tangent vector is integrated with the orbit
    (REBOUND's variational equations), and T_Ly = t_final / ln(d), where d is its
    final length. Chaotic orbits give T_Ly well below the run length; regular
    orbits, whose tangent vector grows only polynomially, give T_Ly of order a
    tenth to a hundredth of it.

    Returns +inf if the tangent vector did not grow (regular), and -inf if a
    planet escaped or the integration broke down after a close encounter
    (certainly chaotic), so that these plot at the ends of the colour scale.
    """
    eps, pr, e1, M1, t_days = args
    sim = two_planet_sim(eps, pr, e1=e1, M1=M1)
    sim.init_megno()
    ln_d0 = tangent_log_length(sim)
    t_final = TWO_PI * pr * t_days / DAYS_PER_YEAR
    try:
        sim.integrate(t_final, exact_finish_time=0)
    except rebound.Escape:
        return -np.inf
    ln_d = tangent_log_length(sim) - ln_d0
    if not np.isfinite(ln_d):
        return -np.inf
    if ln_d <= 0:
        return np.inf
    return np.log10(t_final / ln_d / TWO_PI)


def period_ratio_history(eps, delta, n_orbits, n_samples=400):
    """Period ratio of a pair started on circular orbits with (a2-a1)/a1 = delta,
    sampled n_samples times over n_orbits inner orbits. Returns (t_orbits, ratio,
    relative energy error)."""
    sim = two_planet_sim(eps, (1 + delta) ** 1.5)
    e0 = sim.energy()
    t = np.linspace(0, n_orbits, n_samples + 1)
    ratio = np.empty_like(t)
    for k, tk in enumerate(t):
        sim.integrate(TWO_PI * tk, exact_finish_time=0)
        o = sim.orbits()
        ratio[k] = o[1].P / o[0].P
    return t, ratio, abs((sim.energy() - e0) / e0)


def cached_grid(path, jobs, func, nproc, recompute=False, chunksize=16):
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


def separatrix_crossings(eps, m_values, zeta=1.0, gamma_max=3000.0):
    """Where the inner arm of the m+1:m separatrix meets the outer arm of the
    m+2:m+1 separatrix: the (period ratio, sigma) at which neighbouring
    first-order resonances begin to overlap (the red dots of DPH13 Figure 4)."""
    out = []
    for m in m_values:
        inner_m, _ = resonance_separatrix(eps, m, zeta, n=3000, gamma_max=gamma_max)   # lower-P arm of m+1:m
        _, outer_n = resonance_separatrix(eps, m + 1, zeta, n=3000, gamma_max=gamma_max)   # upper-P arm of m+2:m+1
        # On the rising parts of each arm, compare period ratios at equal sigma
        k = np.argmin(inner_m[1])
        s_in, p_in = inner_m[1][k:], inner_m[0][k:]
        s_grid = np.linspace(max(s_in.min(), outer_n[1].min()), min(s_in.max(), outer_n[1].max()), 2000)
        if len(s_grid) < 2:
            continue
        diff = np.interp(s_grid, s_in, p_in) - np.interp(s_grid, outer_n[1], outer_n[0])
        j = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(j):
            out.append((np.interp(s_grid[j[0]], s_in, p_in), s_grid[j[0]], m))
    return out
