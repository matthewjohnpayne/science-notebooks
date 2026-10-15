"""Helper module for the notebook "Fitting Interacting Planets: N-body Radial Velocities and
Differential-Evolution MCMC".

It follows Nelson, Ford & Payne (2014, ApJS 210, 11; NFP14), the RUN DMC method: model a
star's radial velocity (RV) with a self-consistent N-body integration when its planets
interact, and sample the posterior with differential-evolution Markov chain Monte Carlo
(DEMCMC; ter Braak 2006) rather than a random walk.

Units: days, m/s, solar masses and au inside the integrator.
"""
import numpy as np
import rebound

G_AU_D = 2.959122082855911e-4          # au^3 / (Msun day^2)
AU_D_TO_MS = 1.495978707e11 / 86400.0  # au/day -> m/s
M_JUP = 9.547919e-4                    # Msun
P0, K0, J0 = 1.0, 1.0, 1.0             # NFP14's modified-Jeffreys scales: 1 day, 1 m/s, 1 m/s


# ---------------------------------------------------------------- RV models
def kepler_E(M, e):
    E = M + e * np.sin(M)
    for _ in range(50):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def keplerian_rv(planets, t, t0=0.0):
    """NFP14 Eq. 1 without the offset: sum over planets of K [cos(omega + f) + e cos omega].
    planets: list of dicts with P (days), K (m/s), e, omega, M (radians, mean anomaly at t0)."""
    v = np.zeros_like(t, dtype=float)
    for p in planets:
        M = p["M"] + 2 * np.pi * (t - t0) / p["P"]
        E = kepler_E(np.mod(M, 2 * np.pi), p["e"])
        f = 2 * np.arctan2(np.sqrt(1 + p["e"]) * np.sin(E / 2), np.sqrt(1 - p["e"]) * np.cos(E / 2))
        v += p["K"] * (np.cos(p["omega"] + f) + p["e"] * np.cos(p["omega"]))
    return v


def mass_from_K(K, P, e, m_star, sin_i=1.0):
    """Planet mass (Msun) giving semi-amplitude K (m/s) for period P (days), from
    K = (2 pi G / P)^(1/3) m sin i / (M + m)^(2/3) / sqrt(1 - e^2), solved by iteration."""
    k = (2 * np.pi * G_AU_D / P) ** (1 / 3) * AU_D_TO_MS
    m = K / k * m_star ** (2 / 3) * np.sqrt(1 - e * e) / sin_i
    for _ in range(30):
        m = K / k * (m_star + m) ** (2 / 3) * np.sqrt(1 - e * e) / sin_i
    return m


def nbody_rv(planets, t, m_star, t0=0.0, steps_per_orbit=200):
    """The star's RV from an N-body integration of coplanar, edge-on planets (sin i = 1), set up
    from the same osculating parameters at t0 (Jacobi elements, as is usual for RV fits), with
    WHFast. The line of sight is +z, and RV is -v_z of the star relative to the barycentre, so
    that a single planet reproduces the Keplerian formula above."""
    sim = rebound.Simulation()
    sim.G = G_AU_D
    sim.add(m=m_star)
    interior = m_star
    for p in sorted(planets, key=lambda q: q["P"]):
        m = mass_from_K(p["K"], p["P"], p["e"], interior)    # Jacobi: the mass interior to this orbit
        interior += m
        sim.add(m=m, P=p["P"], e=p["e"], inc=np.pi / 2, Omega=0.0, omega=p["omega"], M=p["M"])
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.dt = min(p["P"] for p in planets) / steps_per_orbit
    sim.t = t0
    out = np.empty(len(t))
    for k, tk in enumerate(t):
        sim.integrate(tk, exact_finish_time=1)
        out[k] = -sim.particles[0].vz * AU_D_TO_MS
    return out


# ---------------------------------------------------------------- parameters and posterior
# Each planet: x = [log(1 + P/P0), K cos(omega + M), K sin(omega + M), e cos omega, e sin omega]
# (NFP14 Sec. 4); then the offset C and log(1 + jitter/J0).
def unpack(x, n_pl):
    planets = []
    for i in range(n_pl):
        lp, kc, ks, ec, es = x[5 * i:5 * i + 5]
        e = np.hypot(ec, es)
        omega = np.arctan2(es, ec)
        planets.append(dict(P=P0 * np.expm1(lp), K=np.hypot(kc, ks), e=e, omega=omega,
                            M=np.arctan2(ks, kc) - omega))
    C = x[5 * n_pl]
    jit = J0 * np.expm1(x[5 * n_pl + 1])
    return planets, C, jit


def pack(planets, C, jit):
    x = []
    for p in planets:
        x += [np.log1p(p["P"] / P0), p["K"] * np.cos(p["omega"] + p["M"]), p["K"] * np.sin(p["omega"] + p["M"]),
              p["e"] * np.cos(p["omega"]), p["e"] * np.sin(p["omega"])]
    return np.array(x + [C, np.log1p(jit / J0)])


def log_prior(planets, C, jit):
    """NFP14 Table 1, coplanar edge-on: modified Jeffreys in P, K and jitter, uniform in e, omega,
    M and C. Written as a density in the sampling variables above: the Jacobian of
    (K, omega+M) -> (K cos, K sin) is 1/K, and of (e, omega) -> (e cos, e sin) is 1/e; the
    log(1 + x) transforms exactly cancel their modified-Jeffreys priors."""
    if jit < 0:
        return -np.inf
    lp = 0.0
    for p in planets:
        if not (0 <= p["e"] < 0.9) or p["P"] <= 0 or p["K"] <= 0:
            return -np.inf
        lp += -np.log1p(p["K"] / K0) - np.log(p["K"]) - np.log(max(p["e"], 1e-12))
    return lp


def log_like(model_rv, data, jit):
    var = data["err"] ** 2 + jit ** 2
    return -0.5 * np.sum((data["rv"] - model_rv) ** 2 / var + np.log(2 * np.pi * var))


class Posterior:
    """log posterior for n_pl planets with either the Keplerian or the N-body model."""

    def __init__(self, data, n_pl, model, m_star):
        self.data, self.n_pl, self.model, self.m_star = data, n_pl, model, m_star

    def __call__(self, x):
        planets, C, jit = unpack(x, self.n_pl)
        lp = log_prior(planets, C, jit)
        if not np.isfinite(lp):
            return -np.inf
        if self.model == "kepler":
            rv = keplerian_rv(planets, self.data["t"])
        else:
            try:
                rv = nbody_rv(planets, self.data["t"], self.m_star)
            except Exception:
                return -np.inf
        return lp + log_like(rv + C, self.data, jit)


# ---------------------------------------------------------------- samplers
def demcmc(logp, x0, n_gen, pool=None, sigma_gamma=0.1, seed=0, target=0.25):
    """NFP14 Sec. 4: differential-evolution MCMC. x0: initial ensemble (n_chains, n_dim). Each
    generation proposes x_i' = x_i + gamma (x_j - x_k) for distinct j, k != i, with
    gamma = gamma0 (1 + z), z ~ N(0, sigma_gamma), and gamma = 1 every 100th generation; gamma0
    starts at 2.38 / sqrt(2 n_dim) and is adapted towards an acceptance fraction of 0.25 by
    NFP14's rule. Returns the chain (n_gen, n_chains, n_dim), log posteriors and acceptance."""
    rng = np.random.default_rng(seed)
    n_ch, n_dim = x0.shape
    gamma0 = 2.38 / np.sqrt(2 * n_dim)
    mapper = pool.map if pool is not None else map
    x = x0.copy()
    lp = np.array(list(mapper(logp, list(x))))
    chain = np.empty((n_gen, n_ch, n_dim))
    lps = np.empty((n_gen, n_ch))
    acc_hist = np.empty(n_gen)
    for g in range(n_gen):
        prop = np.empty_like(x)
        for i in range(n_ch):
            j, k = rng.choice([c for c in range(n_ch) if c != i], 2, replace=False)
            gam = 1.0 if g % 100 == 99 else gamma0 * (1 + rng.normal(0, sigma_gamma))
            prop[i] = x[i] + gam * (x[j] - x[k])
        lp_new = np.array(list(mapper(logp, list(prop))))
        accept = np.log(rng.random(n_ch)) < lp_new - lp
        x[accept], lp[accept] = prop[accept], lp_new[accept]
        chain[g], lps[g] = x, lp
        f = accept.mean()
        acc_hist[g] = f
        if g % 100 != 99:                       # NFP14's adaptation of gamma0
            gamma0 *= 0.9 if f < 0.2 else 1.1 if f > 0.31 else np.sqrt(max(f, 1e-3) / target)
    return chain, lps, acc_hist


def random_walk(logp, x0, n_steps, scales, seed=0):
    """Random-walk Metropolis-Hastings with a Gaussian proposal of the given per-parameter scales,
    updating all parameters at once (NFP14 Sec. 3.2), for comparison."""
    rng = np.random.default_rng(seed)
    x, lp = x0.copy(), logp(x0)
    chain = np.empty((n_steps, x0.size))
    n_acc = 0
    for s in range(n_steps):
        prop = x + scales * rng.normal(size=x.size)
        lpp = logp(prop)
        if np.log(rng.random()) < lpp - lp:
            x, lp = prop, lpp
            n_acc += 1
        chain[s] = x
    return chain, n_acc / n_steps


def autocorr_time(y, c=5.0):
    """Integrated autocorrelation time of a 1-D series (Sokal's automatic window)."""
    y = np.asarray(y) - np.mean(y)
    n = len(y)
    f = np.fft.rfft(y, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    taus = 2 * np.cumsum(acf) - 1
    for m in range(1, n):
        if m >= c * taus[m]:
            return taus[m]
    return taus[-1]
