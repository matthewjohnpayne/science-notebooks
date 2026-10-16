"""Helper module for the notebook "How Planets Lose Their Moons around White Dwarfs".

It reproduces the numerical experiments of Payne, Veras, Holman & Gänsicke
(2016, MNRAS 457, 217; PVHG16) with REBOUND: how stellar mass loss changes a
planet's Hill sphere, how single close encounters between planets strip moons,
and how a close pericentre passage about the white dwarf contracts the Hill
sphere; and, at reduced scale, the follow-up of Payne, Veras, Gänsicke &
Holman (2017, MNRAS 464, 2557; PVGH17), in which liberated moons are tracked
through planet-planet scattering.

Units: au, days and solar masses (G = k^2), unless a name says otherwise.
Functions that run in worker processes live here, not in the notebook, so that
multiprocessing works on macOS and Windows.
"""
import numpy as np
import rebound

G = 2.959122082855911e-4          # au^3 / (Msun day^2)
YEAR = 365.25
AU_KM = 1.495978707e8
M_JUP = 9.547919e-4               # Msun (IAU 2015 nominal GM ratio)
M_EARTH = 3.003489e-6
R_JUP = 71492.0 / AU_KM           # equatorial radius, au
R_EARTH = 6371.0 / AU_KM          # mean radius, au


def hill_radius(a, e, m_p, m_star):
    """PVHG16 Eq. 1: r_H = a (1 - e) (M_p / 3 M_star)^(1/3)."""
    return a * (1 - e) * (m_p / (3 * m_star)) ** (1 / 3)


def new_sim():
    sim = rebound.Simulation()
    sim.G = G
    sim.integrator = "ias15"
    return sim


# ---------------------------------------------------------------- 1. mass loss
def mass_loss_run(m0=1.5, m1=0.6, t_loss_yr=2.0e4, a_p=10.0, m_p=M_JUP, moon_frac=(0.05, 0.2, 0.4),
                  n_out=400, seed=3):
    """A planet with three moons around a star that loses mass at a constant rate from m0
    to m1 (Msun) over t_loss_yr, followed by as long again at constant mass.

    Mass loss is isotropic and slow compared with the planet's orbit, the "adiabatic"
    regime of PVHG16 Sec. 2. The star's mass is lowered in small steps between calls to
    integrate(), which leaves positions and velocities unchanged, as isotropic mass loss
    does. Moons start on circular orbits at the given fractions of the initial Hill radius.
    Returns times (yr), the star's mass, the planet's a and e, the Hill radius, and each
    moon's planetocentric semi-major axis and eccentricity.
    """
    rng = np.random.default_rng(seed)
    sim = new_sim()
    sim.add(m=m0)
    sim.add(m=m_p, a=a_p, e=0.0)
    rH0 = hill_radius(a_p, 0.0, m_p, m0)
    for f in moon_frac:
        sim.add(primary=sim.particles[1], a=f * rH0, e=0.0, inc=rng.uniform(0, 0.3),
                Omega=rng.uniform(0, 2 * np.pi), M=rng.uniform(0, 2 * np.pi))
    sim.N_active = 2
    sim.move_to_com()
    t_end = 2 * t_loss_yr * YEAR
    n_sub = 2000                                    # mass updates per output interval
    times = np.linspace(0, t_end, n_out)
    out = dict(t=times / YEAR, m_star=np.zeros(n_out), a_p=np.zeros(n_out), e_p=np.zeros(n_out),
               r_H=np.zeros(n_out), a_m=np.zeros((n_out, len(moon_frac))), e_m=np.zeros((n_out, len(moon_frac))))
    star_mass = lambda t: m0 + (m1 - m0) * min(t / (t_loss_yr * YEAR), 1.0)
    for k, t in enumerate(times):
        if k:
            for tt in np.linspace(times[k - 1], t, n_sub // (n_out // 20) + 2)[1:]:
                sim.integrate(tt)
                sim.particles[0].m = star_mass(sim.t)
        o = sim.particles[1].orbit(primary=sim.particles[0])
        out["m_star"][k] = sim.particles[0].m
        out["a_p"][k], out["e_p"][k] = o.a, o.e
        out["r_H"][k] = hill_radius(o.a, 0.0, m_p, sim.particles[0].m)
        for j in range(len(moon_frac)):
            om = sim.particles[2 + j].orbit(primary=sim.particles[1])
            out["a_m"][k, j], out["e_m"][k, j] = om.a, om.e
    return out


# ---------------------------------------------------------------- 2. single flybys
# PVHG16 Table 1: close-encounter pericentre q (au), speed at pericentre V_q (au/day),
# planet mass, and the total simulated time t_sim (days). M_star = 1, a_p = 30 au, e_p = 0.
FLYBYS = {
    "A": dict(q=10.0, vq=1.0e-3, m_p=M_JUP, r_p=R_JUP, t_sim=2.5e4, note="Common, little effect"),
    "B": dict(q=1.0, vq=1.0e-3, m_p=M_JUP, r_p=R_JUP, t_sim=2.5e4, note="Relatively rare, destructive"),
    "C": dict(q=1.0, vq=3.0e-4, m_p=M_JUP, r_p=R_JUP, t_sim=8.3e3, note="Very rare, highly destructive"),
    "X": dict(q=1.0, vq=1.0e-3, m_p=M_EARTH, r_p=R_EARTH, t_sim=3.6e3, note="Common, little effect"),
    "Y": dict(q=3.0e-2, vq=3.0e-4, m_p=M_EARTH, r_p=R_EARTH, t_sim=1.2e4, note="Relatively rare, destructive"),
    "Z": dict(q=1.0e-2, vq=3.0e-4, m_p=M_EARTH, r_p=R_EARTH, t_sim=1.2e4, note="Very rare, highly destructive"),
}
A_P, M_STAR = 30.0, 1.0


def flyby_massive(q, vq, m_p, t_sim, a_p=A_P, m_star=M_STAR, geometry="crossing"):
    """The star, the moon's parent planet (circular orbit at a_p) and an equal-mass flyby
    planet, set up at the moment of closest approach (separation q, relative speed vq) and
    integrated back by t_sim/2, so that the encounter's closest approach falls in the middle
    of a forward run of length t_sim.

    PVHG16 do not give the geometry. By default ("crossing") we offset the flyby along the
    parent's direction of motion and give it a relative velocity along the radius vector,
    so that its orbit crosses the parent's, as in planet-planet scattering. The
    alternative ("overtaking") offsets it radially, with a relative velocity along the
    direction of motion.
    """
    sim = new_sim()
    sim.add(m=m_star)
    sim.add(m=m_p, a=a_p, e=0.0)
    p = sim.particles[1]
    P, V = np.array(p.xyz), np.array(p.vxyz)
    rhat, that = P / np.linalg.norm(P), V / np.linalg.norm(V)
    off, vel = (that, rhat) if geometry == "crossing" else (rhat, that)
    x, v = P + q * off, V + vq * vel
    sim.add(m=m_p, x=x[0], y=x[1], z=x[2], vx=v[0], vy=v[1], vz=v[2])
    sim.move_to_com()
    sim.dt = -1.0
    sim.integrate(-0.5 * t_sim)
    sim.t = 0.0
    sim.dt = 1.0
    return sim


def sample_moons(n, r_H, r_p, rng):
    """PVHG16 Sec. 4.1: a log-uniform between R_p and 0.5 r_H, e uniform in (0, 1), keeping
    pericentre > R_p and apocentre < 0.5 r_H; inclination uniform in (0, 180 deg); the other
    angles uniform. Returns arrays a (au), e, inc, Omega, omega, M (radians)."""
    out = []
    while sum(len(o) for o in out) < n:
        m = 4 * n
        a = np.exp(rng.uniform(np.log(r_p), np.log(0.5 * r_H), m))
        e = rng.uniform(0, 1, m)
        ok = (a * (1 - e) > r_p) & (a * (1 + e) < 0.5 * r_H)
        out.append(np.stack([a[ok], e[ok]], 1))
    ae = np.concatenate(out)[:n]
    return dict(a=ae[:, 0], e=ae[:, 1], inc=rng.uniform(0, np.pi, n), Omega=rng.uniform(0, 2 * np.pi, n),
                omega=rng.uniform(0, 2 * np.pi, n), M=rng.uniform(0, 2 * np.pi, n))


def moon_state(sim, j, parent=1):
    """Position and velocity of particle j relative to the parent planet."""
    p, m = sim.particles[parent], sim.particles[j]
    return np.array(m.xyz) - np.array(p.xyz), np.array(m.vxyz) - np.array(p.vxyz)


def flyby_batch(args):
    """Integrate one batch of test moons through one encounter. args = (case name or dict,
    moons dict for this batch). Returns, per moon: bound to the parent at the end, bound to
    the flyby planet instead, and the final planetocentric a and e (nan if unbound)."""
    case, moons = args
    c = FLYBYS[case] if isinstance(case, str) else case
    sim = flyby_massive(c["q"], c["vq"], c["m_p"], c["t_sim"], geometry=c.get("geometry", "crossing"))
    r_H = hill_radius(A_P, 0.0, c["m_p"], M_STAR)
    for k in range(len(moons["a"])):
        sim.add(primary=sim.particles[1], m=0.0, a=moons["a"][k], e=moons["e"][k], inc=moons["inc"][k],
                Omega=moons["Omega"][k], omega=moons["omega"][k], M=moons["M"][k])
    sim.N_active = 3
    sim.integrate(c["t_sim"])
    n = len(moons["a"])
    bound, captured = np.zeros(n, bool), np.zeros(n, bool)
    a_f, e_f = np.full(n, np.nan), np.full(n, np.nan)
    gm = G * c["m_p"]
    for k in range(n):
        for parent, flag in ((1, bound), (2, captured)):
            r, v = moon_state(sim, 3 + k, parent)
            rn = np.linalg.norm(r)
            energy = 0.5 * v @ v - gm / rn
            if energy < 0 and rn < r_H:
                flag[k] = True
                if parent == 1:
                    o = sim.particles[3 + k].orbit(primary=sim.particles[1])
                    a_f[k], e_f[k] = o.a, o.e
                break
    return bound, captured, a_f, e_f


def flyby_experiment(case, n_moons, pool=None, seed=2016, batch=25):
    """PVHG16's experiment for one encounter: n_moons test moons, sorted by semi-major axis
    into batches (so that each batch's shared time step suits all its moons)."""
    c = FLYBYS[case] if isinstance(case, str) else case
    r_H = hill_radius(A_P, 0.0, c["m_p"], M_STAR)
    moons = sample_moons(n_moons, r_H, c["r_p"], np.random.default_rng(seed))
    order = np.argsort(moons["a"])
    jobs = [(case, {k: v[order[i:i + batch]] for k, v in moons.items()}) for i in range(0, n_moons, batch)]
    # Most expensive batches (smallest a) first, so the pool stays busy
    parts = pool.map(flyby_batch, jobs, chunksize=1) if pool is not None else list(map(flyby_batch, jobs))
    res = [np.concatenate([p[i] for p in parts]) for i in range(4)]
    inv = np.empty_like(order)
    inv[order] = np.arange(n_moons)
    bound, captured, a_f, e_f = (r[inv] for r in res)
    return dict(a=moons["a"], e=moons["e"], inc=moons["inc"], a_rH=moons["a"] / r_H, bound=bound,
                captured=captured, a_final=a_f, e_final=e_f, r_H=r_H)


def flyby_track(case, n_out=600):
    """The flyby planet's path relative to the parent planet through one encounter (au),
    and the times (days)."""
    c = FLYBYS[case] if isinstance(case, str) else case
    sim = flyby_massive(c["q"], c["vq"], c["m_p"], c["t_sim"], geometry=c.get("geometry", "crossing"))
    t = np.linspace(0, c["t_sim"], n_out)
    rel = np.zeros((n_out, 3))
    for k, tt in enumerate(t):
        sim.integrate(tt)
        rel[k] = np.array(sim.particles[2].xyz) - np.array(sim.particles[1].xyz)
    return t, rel


def flyby_cached(case, n_moons, pool, cache_dir, recompute=False, seed=2016, **overrides):
    """flyby_experiment, saved to (and read from) cache_dir/flyby_<case>[_<overrides>]_<n>.npz."""
    import os
    tag = case + "".join(f"_{k}-{v}" for k, v in sorted(overrides.items()))
    path = os.path.join(cache_dir, f"flyby_{tag}_{n_moons}.npz")
    if os.path.exists(path) and not recompute:
        return dict(np.load(path))
    c = dict(FLYBYS[case], **overrides)
    res = flyby_experiment(c, n_moons, pool, seed=seed)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32 if np.asarray(v).dtype.kind == "f" else None) for k, v in res.items()})
    return dict(np.load(path))


# ---------------------------------------------------------------- 3. pericentre passages
def pericentre_batch(args):
    """Moons on circular orbits at a fraction f of the Hill radius *at pericentre*, around a
    planet on an eccentric orbit about a white dwarf, followed through one pericentre passage
    (from apocentre to apocentre). args = (q, a_p, m_p, m_wd, fractions, seed).
    Returns the fraction of moons still bound at each f."""
    q, a_p, m_p, m_wd, fracs, seed = args
    rng = np.random.default_rng(seed)
    e_p = 1 - q / a_p
    rH_q = hill_radius(a_p, e_p, m_p, m_wd)
    sim = new_sim()
    sim.add(m=m_wd)
    sim.add(m=m_p, a=a_p, e=e_p, f=np.pi)                 # start at apocentre
    for f in fracs:
        sim.add(primary=sim.particles[1], a=f * rH_q, e=0.0, inc=np.arccos(rng.uniform(-1, 1)),
                Omega=rng.uniform(0, 2 * np.pi), omega=rng.uniform(0, 2 * np.pi), M=rng.uniform(0, 2 * np.pi))
    sim.N_active = 2
    sim.move_to_com()
    sim.integrate(sim.particles[1].P)                      # one orbit: back to apocentre
    gm = G * m_p
    rH_apo = hill_radius(a_p, -e_p, m_p, m_wd)             # a(1+e) at apocentre
    bound = np.zeros(len(fracs), bool)
    for k in range(len(fracs)):
        r, v = moon_state(sim, 2 + k)
        rn = np.linalg.norm(r)
        bound[k] = (0.5 * v @ v - gm / rn < 0) and rn < rH_apo
    return bound


# ---------------------------------------------------------------- 4. scattering with moons
def scattering_run(args):
    """A tightly packed system of planets with one moon each, around a white dwarf,
    integrated until it has scattered (a small-scale analogue of PVGH17).

    args = (seed, n_planets, m_p, a_inner, spacing in mutual Hill radii, moon fraction of
    r_H, t_end in years, n_out). Planets start on near-circular, nearly coplanar orbits;
    each moon on a circular orbit at the given fraction of its planet's Hill radius.
    Returns sampled times and, for every planet and moon, the heliocentric pericentre and
    inclination, and whether each moon is still bound to its parent planet.
    """
    seed, n_pl, m_p, a_in, spacing, f_moon, t_end_yr, n_out = args
    rng = np.random.default_rng(seed)
    m_wd = 0.6
    sim = new_sim()
    sim.add(m=m_wd)
    a = a_in
    for k in range(n_pl):
        sim.add(m=m_p, a=a, e=rng.uniform(0, 0.01), inc=np.radians(rng.uniform(0, 0.5)),
                Omega=rng.uniform(0, 2 * np.pi), omega=rng.uniform(0, 2 * np.pi), M=rng.uniform(0, 2 * np.pi))
        # next planet a given number of mutual Hill radii further out
        rmh = 0.5 * (2 * m_p / (3 * m_wd)) ** (1 / 3)
        a = a * (1 + spacing * rmh) / (1 - spacing * rmh)
    for k in range(n_pl):
        o = sim.particles[1 + k].orbit(primary=sim.particles[0])
        rH = hill_radius(o.a, 0.0, m_p, m_wd)
        sim.add(primary=sim.particles[1 + k], m=0.0, a=f_moon * rH, e=0.0, inc=np.arccos(rng.uniform(-1, 1)),
                Omega=rng.uniform(0, 2 * np.pi), M=rng.uniform(0, 2 * np.pi))
    sim.N_active = 1 + n_pl          # planets are point masses: no collisions are modelled
    sim.move_to_com()
    times = np.linspace(0, t_end_yr * YEAR, n_out)
    q = np.full((n_out, 2 * n_pl), np.nan)
    inc = np.full((n_out, 2 * n_pl), np.nan)
    moon_bound = np.zeros((n_out, n_pl), bool)
    star = sim.particles[0]
    for i, t in enumerate(times):
        sim.integrate(t)
        for j in range(2 * n_pl):
            o = sim.particles[1 + j].orbit(primary=star)
            if o.e < 1:
                q[i, j], inc[i, j] = o.a * (1 - o.e), np.degrees(o.inc)
        for k in range(n_pl):
            r, v = moon_state(sim, 1 + n_pl + k, parent=1 + k)
            o = sim.particles[1 + k].orbit(primary=star)
            rH = hill_radius(o.a, max(o.e, 0.0), m_p, m_wd) if o.e < 1 else 0.0
            rn = np.linalg.norm(r)
            moon_bound[i, k] = (0.5 * v @ v - G * m_p / rn < 0) and rn < rH
    return dict(t=times / YEAR, q=q, inc=inc, moon_bound=moon_bound, n_pl=n_pl)
