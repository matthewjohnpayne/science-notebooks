"""Short-arc orbits: attributables, the admissible region and systematic ranging.

A tracklet measures an *attributable*, the position (alpha, delta) and rate (alpha-dot, delta-dot)
on the sky at one time; the distance rho and radial velocity rho-dot are not measured. Every
(rho, rho-dot) completes the attributable to a full orbit. The *admissible region* (Milani et al.
2004) is the set of (rho, rho-dot) giving a heliocentric orbit that is bound to the Sun and not
bound to the Earth.

*Systematic ranging* (Chesley 2005; Farnocchia, Chesley & Micheli 2015) covers the admissible
region with a grid; at each node rho and rho-dot are held fixed and the other four parameters are
fitted by least squares to all the observations. The chi-squared surface over the grid then
says how well each part of the region fits, and gives every node a probability. Here the grid
in rho-dot follows the floor of the chi-squared valley, so that the same code works for one
tracklet (a valley as wide as the admissible region) and for several (a very narrow one).
"""
from dataclasses import dataclass

import numpy as np

from .ephem import body_state
from .frames import C_AU_DAY, GM_SUN, radec_to_unit
from .mpc import observer_states
from .predict import light_time_shift, residuals, topocentric

GM_EARTH = GM_SUN / 332946.0487
R_HILL_EARTH = 0.0098   # au


@dataclass
class Attributable:
    t: float               # TDB reference time (light-reception time)
    jd_utc: float
    x: np.ndarray          # (alpha, delta, alpha-dot, delta-dot) in rad and rad/day
    cov: np.ndarray        # 4x4
    observer: np.ndarray   # barycentric station state (6,) at t
    stn: str


def fit_attributable(obs, idx=None):
    """Weighted straight-line fit of RA and Dec against time over one tracklet."""
    o = obs if idx is None else obs.select(idx)
    stns = np.unique(o.stn)
    if len(stns) != 1:
        raise ValueError("an attributable needs a single-station tracklet")
    t_ref = o.t.mean()
    jd_ref = o.jd_utc.mean()
    dt = o.t - t_ref
    ra = np.unwrap(o.ra)
    x = np.empty(4)
    cov = np.zeros((4, 4))
    for k, (y, sig) in enumerate([(ra, o.sig_ra / np.cos(o.dec)), (o.dec, o.sig_dec)]):
        w = 1 / (sig * np.pi / 180 / 3600) ** 2
        A = np.stack([np.ones_like(dt), dt], axis=1)
        N = A.T @ (A * w[:, None])
        C = np.linalg.inv(N)
        b = C @ (A.T @ (w * y))
        x[[k, k + 2]] = b
        cov[np.ix_([k, k + 2], [k, k + 2])] = C
    x[0] %= 2 * np.pi
    observer = observer_states([stns[0]], [jd_ref])[0]
    return Attributable(t_ref, jd_ref, x, cov, observer, str(stns[0]))


def states_from_attributable(p, rho, rhodot, attr):
    """Barycentric states (N, 6) at attr.t from attributables p (N, 4) and (rho, rho-dot) (N,).

    rho is the distance at the time the light left the object, so the state is first built at
    t - rho/c and then shifted to the common reference time.
    """
    p = np.atleast_2d(p)
    rho, rhodot = np.atleast_1d(rho), np.atleast_1d(rhodot)
    a, d, ad, dd = p.T
    u = radec_to_unit(a, d)
    ua = np.stack([-np.cos(d) * np.sin(a), np.cos(d) * np.cos(a), np.zeros_like(a)], axis=-1)
    ud = np.stack([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)], axis=-1)
    udot = ad[:, None] * ua + dd[:, None] * ud
    R, V = attr.observer[:3], attr.observer[3:]
    r = R + rho[:, None] * u
    v = (V + rhodot[:, None] * u + rho[:, None] * udot) / (1 - rhodot / C_AU_DAY)[:, None]
    sun = body_state("sun", attr.t)[0, :3]
    return light_time_shift(np.concatenate([r, v], axis=1), sun, rho / C_AU_DAY)


def rhodot_interval(attr, rho):
    """Range of rho-dot for which the heliocentric orbit is bound, at each rho (NaN if none)."""
    rho = np.atleast_1d(rho)
    a, d, ad, dd = attr.x
    u = radec_to_unit(a, d)
    ua = np.array([-np.cos(d) * np.sin(a), np.cos(d) * np.cos(a), 0])
    ud = np.array([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)])
    sun = body_state("sun", attr.t)[0]
    r = attr.observer[:3] - sun[:3] + rho[:, None] * u
    w = attr.observer[3:] - sun[3:] + rho[:, None] * (ad * ua + dd * ud)
    b = w @ u
    c = np.sum(w * w, axis=1) - 2 * GM_SUN / np.linalg.norm(r, axis=1)
    disc = b * b - c
    root = np.sqrt(np.where(disc > 0, disc, np.nan))
    return -b - root, -b + root


def earth_bound(states, t):
    """True where an orbit is inside the Earth's Hill sphere with negative geocentric energy."""
    e = body_state("earth", t)[0]
    dr = states[:, :3] - e[:3]
    dv = states[:, 3:] - e[3:]
    r = np.linalg.norm(dr, axis=1)
    return (r < R_HILL_EARTH) & (0.5 * np.sum(dv * dv, 1) - GM_EARTH / r < 0)


def helio_energy(states, t):
    sun = body_state("sun", t)[0]
    dr, dv = states[:, :3] - sun[:3], states[:, 3:] - sun[3:]
    return 0.5 * np.sum(dv * dv, 1) - GM_SUN / np.linalg.norm(dr, axis=1)


@dataclass
class Nodes:
    """Ranging nodes: fixed (rho, rho-dot), each the centre of a cell of size ds (in ln rho) by drd."""
    rho: np.ndarray
    rhodot: np.ndarray
    ds: np.ndarray
    drd: np.ndarray
    p: np.ndarray              # fitted attributables (N, 4)
    chi2: np.ndarray = None
    states: np.ndarray = None  # barycentric states at attr.t
    A: np.ndarray = None       # (N, 4, 4) normal matrices of the attributable fits

    def __len__(self):
        return len(self.rho)

    def take(self, k):
        return Nodes(*(None if v is None else v[k] for v in
                       (self.rho, self.rhodot, self.ds, self.drd, self.p, self.chi2, self.states, self.A)))

    @property
    def area(self):
        """Cell area in (rho, rho-dot): the prior weight of a node, for a prior uniform in both."""
        return self.rho * self.ds * self.drd


def _lm(q, build, obs, t, h, n_iter, max_step=None, lower=None, upper=None):
    """Levenberg-Marquardt on many independent problems at once.

    q (n, m) starting parameters; build(Q) -> states (len(Q), 6) at time t; h (m,) difference steps.
    Each iteration propagates every problem and m finite-difference neighbours in one ASSIST run.
    max_step (m,) caps each parameter's change per iteration (a crude trust region, needed where
    the chi-squared surface is nearly flat). Orbits that strike the Earth give NaN residuals and
    so are never accepted. lower, upper (n, m) bound the parameters (steps are clipped to them).
    Returns (q_best, chi2_best, normal matrix J^T W J at q_best (n, m, m)).
    """
    n, m = q.shape
    W = obs.weights.reshape(1, -1)
    q_best, chi_best = q.copy(), np.full(n, np.inf)
    A, g = np.zeros((n, m, m)), np.zeros((n, m))
    lam = np.full(n, 1e-3)
    for _ in range(n_iter):
        Q = np.concatenate([q] + [q + h[k] * np.eye(m)[k] for k in range(m)])
        res = residuals(build(Q), t, obs).reshape(m + 1, n, -1)
        chi = np.sum(res[0] ** 2 * W, axis=1)
        chi = np.where(np.isfinite(chi), chi, np.inf)
        better = chi < chi_best
        q_best[better], chi_best[better] = q[better], chi[better]
        lam = np.where(better, lam / 10, lam * 10)
        J = np.moveaxis(-(res[1:] - res[0][None]) / h[:, None, None], 0, -1)   # (n, M, m)
        A[better] = np.einsum("nki,k,nkj->nij", J, W[0], J)[better]
        g[better] = np.einsum("nki,k,nk->ni", J, W[0], res[0])[better]
        D = A + lam[:, None, None] * A * np.eye(m)[None] + 1e-30 * np.eye(m)[None]
        step = np.linalg.solve(D, g[..., None])[..., 0]
        if max_step is not None:
            step = np.clip(step, -max_step, max_step)
        q = q_best + np.nan_to_num(step)
        if lower is not None:
            q = np.clip(q, lower, upper)
    return q_best, chi_best, A


def _steps(attr):
    return np.maximum(0.1 * np.sqrt(np.diag(attr.cov)), 1e-10)


def fit_nodes(nodes, attr, obs, n_iter=8):
    """Fit the four attributable parameters at every node (rho, rho-dot held fixed) to `obs`.

    Nodes whose orbit would strike the Earth during the observations get chi2 = inf.
    """
    n = len(nodes)
    nodes.chi2 = np.full(n, np.inf)
    nodes.states = np.full((n, 6), np.nan)
    ok = np.ones(n, bool)
    if ok.any():
        rho, rhodot = nodes.rho[ok], nodes.rhodot[ok]
        build = lambda Q: states_from_attributable(Q, np.tile(rho, len(Q) // len(rho)),
                                                   np.tile(rhodot, len(Q) // len(rho)), attr)
        p, chi, A = _lm(nodes.p[ok], build, obs, attr.t, _steps(attr), n_iter)
        nodes.p[ok], nodes.chi2[ok] = p, chi
        nodes.A = np.zeros((n, 4, 4))
        nodes.A[ok] = A
        nodes.states[ok] = states_from_attributable(p, rho, rhodot, attr)
    return nodes


def profile(attr, obs, s, n_start=5, n_iter=10, q_start=None):
    """The chi-squared valley: at each ln rho in s, fit rho-dot as well as the attributable.

    Several starting rho-dots across the admissible interval are tried and the best is kept,
    unless q_start (len(s), 5) gives one starting point (attributable, rho-dot) per rho.
    Returns dict of arrays: rho, rhodot (best fit), sig_rhodot, chi2, p.
    """
    rho = np.exp(s)
    lo, hi = rhodot_interval(attr, rho)
    good = np.isfinite(lo)
    rho, lo, hi = rho[good], lo[good], hi[good]
    if q_start is not None:
        n_start = 1
        q0 = np.asarray(q_start, float)[good].copy()
        q0[:, 4] = np.clip(q0[:, 4], lo, hi)
    else:
        fr = (np.arange(n_start) + 0.5) / n_start
        q0 = np.column_stack([np.tile(attr.x, (len(rho) * n_start, 1)),
                              (lo[:, None] + fr[None] * (hi - lo)[:, None]).ravel()])
    R = np.repeat(rho, n_start)
    ok = np.ones(len(R), bool)
    q, chi, A = q0.copy(), np.full(len(R), np.inf), np.zeros((len(R), 5, 5))
    if ok.any():
        Rk = R[ok]
        build = lambda Q: states_from_attributable(Q[:, :4], np.tile(Rk, len(Q) // len(Rk)), Q[:, 4], attr)
        h = np.append(_steps(attr), 1e-7)
        cap = np.append(100 * np.sqrt(np.diag(attr.cov)), 0.1 * np.nanmax(hi - lo))
        lower = np.full((ok.sum(), 5), -np.inf)
        upper = np.full((ok.sum(), 5), np.inf)
        lower[:, 4], upper[:, 4] = np.repeat(lo, n_start)[ok], np.repeat(hi, n_start)[ok]
        q[ok], chi[ok], A[ok] = _lm(q0[ok], build, obs, attr.t, h, n_iter, cap, lower, upper)
    # sigma(rho-dot) with the attributable marginalised: 1 / sqrt(Schur complement of the normal
    # matrix). Where chi2 is flat in rho-dot (a single tracklet) the complement vanishes and the
    # valley is as wide as the admissible interval.
    with np.errstate(all="ignore"):
        A4 = A[:, :4, :4] + 1e-30 * np.eye(4)[None]
        schur = A[:, 4, 4] - np.einsum("ni,ni->n", A[:, 4, :4], np.linalg.solve(A4, A[:, :4, 4:5])[..., 0])
        sig = np.where(schur > 0, 1 / np.sqrt(np.abs(schur)), np.inf)
    chi = chi.reshape(len(rho), n_start)
    best = np.argmin(chi, axis=1)
    pick = np.arange(len(rho)) * n_start + best
    return dict(s=np.log(rho), rho=rho, rhodot=q[pick, 4], sig_rhodot=sig[pick],
                chi2=chi[np.arange(len(rho)), best], p=q[pick, :4], lo=lo, hi=hi)


def systematic_ranging(attr, obs, n_s=60, n_rd=24, rho_min=1e-4, rho_max=100.0, zmax=6.0,
                       dchi2=50.0, levels=20, ds_min=1e-6, n_iter=8,
                       dchi2_nodes=50.0):
    """Systematic ranging that follows the chi-squared valley.

    1. Profile: on a grid in ln rho, fit rho-dot and the attributable (5 parameters) to find the
       valley floor rho-dot*(rho) and its width sigma(rho).
    2. Refine the rho grid, halving cells, wherever the profile is within dchi2 of its minimum
       and not yet resolved (chi2 changes by more than 1 between neighbouring cells).
    3. At each rho, place n_rd nodes across rho-dot* +- zmax sigma (clipped to the admissible
       interval) and fit the attributable at each with (rho, rho-dot) fixed.
       Only rho values whose profile is within dchi2_nodes of the minimum get nodes (None: all).
    With one tracklet the valley is as wide as the admissible region; with more it narrows.
    """
    edges = np.linspace(np.log(rho_min), np.log(rho_max), n_s + 1)
    prof = profile(attr, obs, 0.5 * (edges[1:] + edges[:-1]))
    prof["ds"] = np.full(len(prof["s"]), edges[1] - edges[0])
    for _ in range(levels):
        c = prof["chi2"]
        # split cells near the minimum whose chi2 differs from a neighbour's by more than 1
        # (the profile is not yet resolved there), down to a smallest cell of ds_min in ln rho
        with np.errstate(invalid="ignore"):
            jump = np.abs(np.diff(c)) > 1
        jump = np.r_[jump, False] | np.r_[False, jump]
        near = (np.minimum(c, np.minimum(np.r_[c[1:], np.inf], np.r_[np.inf, c[:-1]])) - c.min() < dchi2)
        near &= jump & (prof["ds"] > ds_min)
        if not near.any():
            break
        ds = prof["ds"][near] / 2
        q_par = np.column_stack([prof["p"][near], prof["rhodot"][near]])
        child = profile(attr, obs, np.concatenate([prof["s"][near] - ds / 2, prof["s"][near] + ds / 2]),
                        n_iter=6, q_start=np.concatenate([q_par, q_par]))
        child["ds"] = np.full(len(child["s"]), np.nan)
        lookup = dict(zip(np.round(np.concatenate([prof["s"][near] - ds / 2, prof["s"][near] + ds / 2]), 12),
                          np.concatenate([ds, ds])))
        child["ds"] = np.array([lookup[v] for v in np.round(child["s"], 12)])
        prof = {key: np.concatenate([prof[key][~near], child[key]]) for key in prof}
        order = np.argsort(prof["s"])
        prof = {key: v[order] for key, v in prof.items()}
    # rho-dot nodes around the valley at every rho that matters
    use = np.ones(len(prof["s"]), bool) if dchi2_nodes is None else prof["chi2"] - prof["chi2"].min() < dchi2_nodes
    prof_all = prof
    prof = {key: v[use] for key, v in prof.items()}
    lo = np.where(np.isfinite(prof["chi2"]), np.maximum(prof["lo"], prof["rhodot"] - zmax * prof["sig_rhodot"]), prof["lo"])
    hi = np.where(np.isfinite(prof["chi2"]), np.minimum(prof["hi"], prof["rhodot"] + zmax * prof["sig_rhodot"]), prof["hi"])
    width = hi - lo
    f = (np.arange(n_rd) + 0.5) / n_rd
    nodes = Nodes(np.repeat(prof["rho"], n_rd), (lo[:, None] + f[None] * width[:, None]).ravel(),
                  np.repeat(prof["ds"], n_rd), np.repeat(width / n_rd, n_rd),
                  np.where(np.isfinite(prof["chi2"])[:, None], prof["p"], attr.x[None]).repeat(n_rd, 0))
    nodes = fit_nodes(nodes, attr, obs, n_iter)
    nodes.profile = prof_all
    return nodes


def weights(nodes, attr, allow_unbound=False):
    """Posterior probability of each node: prior cell area times exp(-chi2/2), normalised."""
    w = nodes.area * np.exp(-0.5 * (nodes.chi2 - np.min(nodes.chi2)))
    with np.errstate(invalid="ignore"):
        if not allow_unbound:
            w = np.where(helio_energy(nodes.states, attr.t) < 0, w, 0)
        w = np.where(earth_bound(nodes.states, attr.t), 0, w)
    w = np.nan_to_num(w)
    return w / w.sum()


def sample(nodes, attr, obs, n, rng, oversample=4, n_iter=4):
    """Draw n orbits from the ranging probability.

    1. Pick candidate nodes by probability and a uniform point (rho, rho-dot) within each cell.
    2. Refit the attributable at each candidate point.
    3. Importance-weight each candidate by exp(-(chi2 - chi2_node)/2), the ratio of the probability
       at the point to the probability its cell was drawn with, and resample n of them.
    4. Draw the attributable from its Gaussian uncertainty at that (rho, rho-dot), so that
       the uncertainty of the attributable itself is included too.
    Returns (states (n, 6) at attr.t, rho, rho-dot).
    """
    m = oversample * n
    w = weights(nodes, attr)
    sub = nodes.take(rng.choice(len(nodes), size=m, p=w))
    chi_node = sub.chi2
    cand = Nodes(sub.rho * np.exp((rng.random(m) - 0.5) * sub.ds),
                 sub.rhodot + (rng.random(m) - 0.5) * sub.drd, sub.ds, sub.drd, sub.p.copy())
    cand = fit_nodes(cand, attr, obs, n_iter)
    iw = np.exp(-0.5 * np.clip(cand.chi2 - chi_node, -50, 700))
    with np.errstate(invalid="ignore"):
        iw = np.where(np.isfinite(cand.chi2) & (helio_energy(cand.states, attr.t) < 0)
                      & ~earth_bound(cand.states, attr.t), iw, 0)
    pick = cand.take(rng.choice(m, size=n, p=iw / iw.sum()))
    cov = np.linalg.inv(pick.A + 1e-30 * np.eye(4)[None])
    p = pick.p + np.einsum("nij,nj->ni", np.linalg.cholesky(cov), rng.standard_normal((n, 4)))
    return states_from_attributable(p, pick.rho, pick.rhodot, attr), pick.rho, pick.rhodot


def rho_rhodot_of(state, t0, attr, dt=1e-3):
    """(rho, rho-dot) of a known orbit as seen from the attributable's station and time."""
    from .ephem import propagate
    times = attr.t + np.array([-dt, 0, dt])
    X = propagate(state[None], t0, times)[:, 0]
    sun = body_state("sun", times)[:, :3]
    obs = observer_states([attr.stn] * 3, attr.jd_utc + np.array([-dt, 0, dt]))
    _, dist = topocentric(X, sun, obs[:, :3])
    return dist[1], (dist[2] - dist[0]) / (2 * dt)
