"""Helper module for the notebook "Can Moons Survive in Tightly Packed Planetary Systems?".

It follows Payne, Deck, Holman & Perets (2013, ApJL 775, L44; PDHP13): add a lunar-mass
satellite to the outer of two planets packed as closely as stability allows, and ask which
satellite orbits survive. Integrations use REBOUND's IAS15.

Units: days, au and solar masses.
"""
import numpy as np
import rebound

G = 2.959122082855911e-4          # au^3 / (Msun day^2)
M_MOON = 3.69e-8                  # lunar mass in solar masses
YEAR = 365.25

# PDHP13 Sec. 2.1: two-planet sets (equal masses m_i in units of the star's mass), with the
# ranges of period ratio P2/P1 and inner-planet eccentricity e1 from which systems are drawn;
# the outer planet is circular at 1 au and M1 = 0.
SETS = {
    "single": dict(m=1e-3, pr=None, e1=None),
    "C": dict(m=1e-4, pr=(1.130, 1.136), e1=(0.012, 0.015)),
    "D": dict(m=1e-4, pr=(1.187, 1.192), e1=(0.095, 0.100)),
    "E": dict(m=1e-4, pr=(1.31, 1.35), e1=(0.00, 0.01)),
    "F": dict(m=1e-3, pr=(1.28, 1.30), e1=(0.095, 0.100)),
    "G": dict(m=1e-3, pr=(1.49, 1.51), e1=(0.095, 0.100)),
}


def hill_parent(m, a=1.0):
    """PDHP13 Sec. 2: R_H,parent = a (m / 3)^(1/3), with m the planet-to-star mass ratio."""
    return a * (m / 3) ** (1 / 3)


def hill_mutual(m1, m2, a1, a2):
    """PDHP13 Sec. 2: R_H,mutual = (a1 + a2)/2 ((m1 + m2)/3)^(1/3)."""
    return 0.5 * (a1 + a2) * ((m1 + m2) / 3) ** (1 / 3)


def boundary_d06(e_sat, retro=False):
    """Domingos et al. (2006) outermost stable satellite orbit for a circular planet, as quoted
    in PDHP13 Sec. 3.1 and 3.3 (in units of R_H,parent)."""
    return 0.9309 * (1 - 0.9812 * e_sat) if retro else 0.49 * (1 - 0.27 * e_sat)


def boundary_pdhp13(e_sat):
    """PDHP13 Eq. 1: the 'extremely approximate' prograde boundary with a close neighbour."""
    return 0.4 * (1 - e_sat / 4)


def system(set_name, rng):
    """Star, (inner planet), parent planet: positions set as in PDHP13, with the inner planet's
    period ratio and eccentricity drawn from the set's ranges. Returns the simulation and the
    parent's index."""
    s = SETS[set_name]
    sim = rebound.Simulation()
    sim.G = G
    sim.add(m=1.0)
    if s["pr"] is not None:
        pr = rng.uniform(*s["pr"])
        e1 = rng.uniform(*s["e1"])
        sim.add(m=s["m"], a=pr ** (-2 / 3), e=e1, M=0.0)
    sim.add(m=s["m"], a=1.0, e=0.0, M=0.0)
    return sim, sim.N - 1


def moon_run(args):
    """One satellite (lunar mass) around the parent planet: a in units of R_H,parent, e, prograde
    or retrograde, other elements zero (PDHP13 Sec. 3.1). Integrated for n_orbits orbits of the
    parent. Returns (survived, closest planet-planet approach in mutual Hill radii)."""
    set_name, a_frac, e_sat, retro, n_orbits, seed = args
    rng = np.random.default_rng(seed)
    sim, ip = system(set_name, rng)
    m = SETS[set_name]["m"]
    rh = hill_parent(m)
    sim.add(primary=sim.particles[ip], m=M_MOON, a=a_frac * rh, e=e_sat, inc=np.pi if retro else 0.0)
    sim.move_to_com()
    sim.integrator = "ias15"
    t_end = n_orbits * YEAR
    n_chk = 200
    gm = G * (m + M_MOON)
    dmin = np.inf
    for t in np.linspace(t_end / n_chk, t_end, n_chk):
        sim.integrate(t)
        p, s = sim.particles[ip], sim.particles[sim.N - 1]
        r = np.array(s.xyz) - np.array(p.xyz)
        v = np.array(s.vxyz) - np.array(p.vxyz)
        rn = np.linalg.norm(r)
        if 0.5 * v @ v - gm / rn >= 0 or rn > rh:
            return False, dmin
        if ip == 2:
            q1 = sim.particles[1]
            d = np.linalg.norm(np.array(q1.xyz) - np.array(p.xyz))
            dmin = min(dmin, d / hill_mutual(m, m, q1.a if q1.a > 0 else 1, 1.0))
    return True, dmin


def planets_only_closest(set_name, n_orbits, seed):
    """Closest approach (in mutual Hill radii) between the two planets without a satellite."""
    rng = np.random.default_rng(seed)
    sim, ip = system(set_name, rng)
    sim.integrator = "ias15"
    m = SETS[set_name]["m"]
    dmin = np.inf
    for t in np.linspace(1, n_orbits * YEAR, 20 * n_orbits):
        sim.integrate(t)
        p, q = sim.particles[2], sim.particles[1]
        dmin = min(dmin, np.linalg.norm(np.array(p.xyz) - np.array(q.xyz)) / hill_mutual(m, m, q.a, p.a))
    return dmin
