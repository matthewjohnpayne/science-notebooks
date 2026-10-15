"""Helper module for the notebook "The Laplace Resonance of GJ 876".

It integrates the coplanar four-planet solution of Nelson, Robertson, Payne, Pritchard,
Deck & Ford (2016, MNRAS 455, 2484; N16) with REBOUND, and measures the libration of the
resonant angles the way N16 do: sqrt(2) times the root-mean-square variation of each angle.

Units: days, au and solar masses.
"""
import numpy as np
import rebound

G = 2.959122082855911e-4        # au^3 / (Msun day^2)
M_EARTH = 3.003489e-6           # Msun
M_STAR = 0.37                   # N16 Sec. 3.1 (von Braun et al. 2014)
INC = 53.19                     # deg: N16 coplanar solution, all planets

# N16 Table 3 (coplanar model): posterior medians, with the 15.9 and 84.1 percentiles as
# (minus, plus) uncertainties. Masses in Earth masses for M_star = 0.37 Msun; semi-major axes
# in au (Jacobi); angles in degrees at N16's epoch, the first ELODIE observation.
COPLANAR = {
    "d": dict(m=(7.49, 0.41, 0.43), a=(0.02183911, 1.8e-7, 1.9e-7), e=(0.119, 0.052, 0.050),
              omega=(-140.90, 23.85, 286.32), M=(301.87, 286.24, 24.82)),
    "c": dict(m=(266.2, 4.3, 4.8), a=(0.135989, 2.1e-5, 2.3e-5), e=(0.2531, 0.0031, 0.0031),
              omega=(115.96, 1.35, 1.21), M=(-220.68, 1.35, 1.38)),
    "b": dict(m=(845.3, 15.0, 16.6), a=(0.218585, 2.6e-5, 2.6e-5), e=(0.0368, 0.0025, 0.0019),
              omega=(110.80, 3.91, 3.86), M=(-285.38, 4.11, 4.15)),
    "e": dict(m=(16.58, 2.11, 1.96), a=(0.3514, 0.0025, 0.0025), e=(0.031, 0.025, 0.030),
              omega=(129.56, 185.78, 36.31), M=(-175.89, 41.44, 190.80)),
}
ORDER = ["d", "c", "b", "e"]

# N16 Table 8: libration amplitudes (deg) for the coplanar model, (median, minus, plus)
N16_AMPLITUDES = {"phi_cb_c": (4.0, 1.4, 1.6), "phi_cb_b": (13.4, 1.4, 3.0), "dvarpi_cb": (14.7, 3.3, 4.0),
                  "phi_be_b": (31.3, 8.7, 11.9), "phi_ce_0": (67.5, 19.4, 25.4), "laplace": (33.0, 9.3, 12.4)}


def medians():
    return {k: {p: v[0] for p, v in d.items()} for k, d in COPLANAR.items()}


def draw(rng, planets=("c", "b", "e")):
    """One set of parameters drawn independently from N16's marginal uncertainties (a split
    Gaussian for each parameter), for the listed planets; the others at their medians.
    Correlations between parameters are ignored, so some draws are poor fits or unstable."""
    out = medians()
    for k in planets:
        for p, (med, lo, hi) in COPLANAR[k].items():
            z = rng.normal()
            out[k][p] = med + z * (hi if z > 0 else lo)
        out[k]["e"] = min(max(out[k]["e"], 0.0), 0.95)
        out[k]["m"] = max(out[k]["m"], 0.1)
    return out


def make_sim(params, inc=INC, mass_scale=None):
    """REBOUND simulation of the star and four planets, coplanar at inclination inc (deg) to the
    sky, from Jacobi elements. mass_scale: optional dict of factors on planet masses."""
    sim = rebound.Simulation()
    sim.G = G
    sim.add(m=M_STAR)
    for k in ORDER:
        p = params[k]
        f = (mass_scale or {}).get(k, 1.0)
        sim.add(m=p["m"] * M_EARTH * f, a=p["a"], e=p["e"], inc=np.radians(inc), Omega=0.0,
                omega=np.radians(p["omega"]), M=np.radians(p["M"]))
    sim.move_to_com()
    sim.integrator = "ias15"
    return sim


def evolve(params, years=60.0, n_out=3000, mass_scale=None):
    """Integrate and return times (yr) and each planet's mean longitude and longitude of
    pericentre (radians, measured in the orbital plane, Jacobi elements)."""
    sim = make_sim(params, mass_scale=mass_scale)
    times = np.linspace(0, years * 365.25, n_out)
    lam = np.zeros((n_out, 4))
    pomega = np.zeros((n_out, 4))
    a = np.zeros((n_out, 4))
    ok = True
    for i, t in enumerate(times):
        try:
            sim.integrate(t)
        except rebound.Escape:
            ok = False
            break
        orbs = sim.orbits()
        for j, o in enumerate(orbs):
            if o.e >= 1:
                ok = False
            lam[i, j], pomega[i, j], a[i, j] = o.l, o.pomega, o.a
    return dict(t=times / 365.25, lam=lam, pomega=pomega, a=a, stable=ok and np.all(np.isfinite(a)) and
                np.max(np.abs(a / a[0] - 1)) < 0.2)


def angles(run):
    """N16 Sec. 6.1's resonant angles, in radians wrapped to (-pi, pi]."""
    lam, pom = run["lam"], run["pomega"]
    d, c, b, e = 0, 1, 2, 3
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    return {
        "phi_cb_c": wrap(lam[:, c] - 2 * lam[:, b] + pom[:, c]),
        "phi_cb_b": wrap(lam[:, c] - 2 * lam[:, b] + pom[:, b]),
        "dvarpi_cb": wrap(pom[:, c] - pom[:, b]),
        "phi_be_b": wrap(lam[:, b] - 2 * lam[:, e] + pom[:, b]),
        "phi_be_e": wrap(lam[:, b] - 2 * lam[:, e] + pom[:, e]),
        "phi_ce_0": wrap(lam[:, c] - 4 * lam[:, e] + 3 * pom[:, c]),
        "laplace": wrap(lam[:, c] - 3 * lam[:, b] + 2 * lam[:, e]),
    }


def amplitude(x):
    """N16's libration amplitude: sqrt(2) x the RMS variation of the angle (degrees), or None if
    it circulates (takes values all round the circle)."""
    deg = np.degrees(x)
    hist, _ = np.histogram(deg, bins=36, range=(-180, 180))
    if np.all(hist > 0):
        return None
    centre = np.degrees(np.angle(np.mean(np.exp(1j * x))))
    dev = (deg - centre + 180) % 360 - 180
    return float(np.sqrt(2) * np.std(dev))


def draw_job(args):
    """Worker: integrate one draw from N16's uncertainties and return its libration amplitudes."""
    seed, years = args
    run = evolve(draw(np.random.default_rng(seed)), years=years, n_out=2000)
    if not run["stable"]:
        return None
    return {k: amplitude(v) for k, v in angles(run).items()}
