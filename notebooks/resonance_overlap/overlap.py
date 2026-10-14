"""Helpers for the resonance-overlap notebook.

Kept in a module (not the notebook) so that multiprocessing workers can import
them on every platform, including macOS and Windows, which spawn rather than fork.

Units: G = 1, stellar mass 1, inner semimajor axis 1, so the inner period is 2*pi.
"""
import numpy as np
import rebound

TWO_PI = 2 * np.pi


# --- Analytic criteria from Deck, Payne & Holman (2013), ApJ 774, 129 ---------

def overlap_delta_circular(eps):
    """(a2 - a1)/a1 below which first-order resonances overlap, for initially
    circular orbits: 1.46 eps^(2/7)  (DPH13 eq. for the circular case)."""
    return 1.46 * np.asarray(eps) ** (2 / 7)


def overlap_period_ratio_circular(eps):
    """Same criterion in averaged period ratio: P2/P1 < 1 + 2.2 eps^(2/7)."""
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


# --- Numerical experiments ----------------------------------------------------

def two_planet_sim(eps, period_ratio, e1=0.0, e2=0.0, M1=np.pi / 2, mass_ratio=1.0):
    """Star plus two planets with total mass eps (relative to the star) and
    m1/m2 = mass_ratio. All angles are zero except the inner mean anomaly M1."""
    m2 = eps / (1 + mass_ratio)
    m1 = eps - m2
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=1.0)
    sim.add(m=m1, a=1.0, e=e1, M=M1)
    sim.add(m=m2, P=TWO_PI * period_ratio, e=e2)
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.dt = TWO_PI / 40          # 40 steps per inner orbit
    sim.exit_max_distance = 10.0
    return sim


# Classes returned by classify()
REGULAR, CHAOTIC, UNSTABLE = 0, 1, 2


def classify(args):
    """Integrate one configuration and classify it.

    args = (eps, period_ratio, e1, n_orbits)
    Returns (class, MEGNO). MEGNO tends to 2 for regular (quasi-periodic) orbits
    and grows without bound for chaotic ones; we call MEGNO > 4 chaotic.
    A run is UNSTABLE if the period ratio has drifted by more than 5% or an
    eccentricity has exceeded 0.3 (the planets have scattered), or if a planet
    escaped. A fixed-step integrator is meaningless after a close encounter,
    so MEGNO is not reported for unstable runs.
    """
    eps, pr, e1, n_orbits = args
    sim = two_planet_sim(eps, pr, e1=e1)
    sim.init_megno()
    try:
        sim.integrate(TWO_PI * n_orbits, exact_finish_time=0)
    except rebound.Escape:
        return UNSTABLE, np.nan
    o = sim.orbits()
    ratio = o[1].P / o[0].P
    y = sim.megno()
    if abs(ratio / pr - 1) > 0.05 or max(o[0].e, o[1].e) > 0.3 or not np.isfinite(y) or y < 0:
        return UNSTABLE, np.nan
    return (CHAOTIC if y > 4 else REGULAR), y


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
