"""Helper module for the notebook "Planet Nine and the Secular Dynamics of Distant TNOs".

It follows Li, Hadden, Payne & Holman (2018, AJ 156, 263; LHPH18): the long-term
(secular) evolution of a distant trans-Neptunian object (TNO) under an eccentric
outer planet ("Planet Nine"), with the four giant planets represented by their
orbit-averaged quadrupole (J2) potential.

The secular equations are evaluated by direct numerical averaging, as LHPH18 do,
but in Gauss's form: the force on the TNO is averaged over both orbits (the TNO's
and Planet Nine's, sampled uniformly in mean anomaly), and the orbit-averaged
rates of change of the TNO's angular-momentum and eccentricity vectors follow from
Gauss's equations. This is equivalent to differentiating LHPH18's averaged
Hamiltonian (their Eq. 1), including the indirect term. Where the two orbits
cross, the integrand is singular; a small softening length smooths it.

Units: au, years and solar masses (G = 4 pi^2). Functions that run in worker
processes live here, not in the notebook, so that multiprocessing works on macOS
and Windows.
"""
import numpy as np
from scipy.integrate import solve_ivp

G = 4 * np.pi**2                     # au^3 / (Msun yr^2)
M_SUN = 1.0
M_EARTH = 3.003489e-6
# The giant planets (mass in Msun, semi-major axis in au), for the J2 term of LHPH18 Eq. 2
GIANTS = {"Jupiter": (9.547919e-4, 5.2026), "Saturn": (2.858860e-4, 9.5549),
          "Uranus": (4.366244e-5, 19.2184), "Neptune": (5.151389e-5, 30.1104)}
SUM_MA2 = sum(m * a**2 for m, a in GIANTS.values())        # sum of m_p a_p^2 / M (au^2)
J2R2 = 0.5 * SUM_MA2                                        # J2 R^2 of the equivalent oblate Sun


def kepler_E(M, e):
    E = M + e * np.sin(M)
    for _ in range(40):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def orbit_points(a, evec, jvec, n):
    """Positions, velocities and time weights at n points on the orbit with eccentricity vector
    evec and dimensionless angular-momentum vector jvec (|j| = sqrt(1-e^2)). The points are
    uniform in eccentric anomaly E, weighted by (1 - e cos E)/n, so that weighted sums are time
    averages; this resolves the pericentre of eccentric orbits much better than uniform steps
    in mean anomaly."""
    e = np.linalg.norm(evec)
    W = jvec / np.linalg.norm(jvec)
    if e > 1e-12:
        P = evec / e
    else:
        P = np.cross([0, 0, 1.0], W)
        P = P / np.linalg.norm(P) if np.linalg.norm(P) > 1e-12 else np.array([1.0, 0, 0])
    Q = np.cross(W, P)
    E = (np.arange(n) + 0.5) * 2 * np.pi / n
    b = np.sqrt(1 - e * e)
    x, y = a * (np.cos(E) - e), a * b * np.sin(E)
    nmot = np.sqrt(G * M_SUN / a**3)
    edot = nmot / (1 - e * np.cos(E))
    vx, vy = -a * np.sin(E) * edot, a * b * np.cos(E) * edot
    r = x[:, None] * P + y[:, None] * Q
    v = vx[:, None] * P + vy[:, None] * Q
    wts = (1 - e * np.cos(E)) / n
    return r, v, wts


def p9_vectors(e9, i9, w9, O9):
    """Eccentricity and angular-momentum unit vectors of Planet Nine's orbit (angles in radians)."""
    cO, sO, cw, sw, ci, si = np.cos(O9), np.sin(O9), np.cos(w9), np.sin(w9), np.cos(i9), np.sin(i9)
    P = np.array([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si])
    W = np.array([sO * si, -cO * si, ci])
    return e9 * P, np.sqrt(1 - e9**2) * W


def p9_precession(a9, e9):
    """LHPH18 Eq. 3: rates of change (rad/yr) of Planet Nine's argument of pericentre and node
    caused by the giant planets."""
    base = np.sqrt(G * M_SUN) / (a9**1.5 * (1 - e9**2) ** 2) * SUM_MA2 / a9**2
    return 1.5 * base, -0.75 * base


class Secular:
    """Orbit-averaged equations for one TNO (semi-major axis a) under Planet Nine (m9, a9, e9,
    i9, w9, O9 at t = 0, precessing at the rates of p9_precession if j2) and, if j2, the giant
    planets' J2 potential."""

    def __init__(self, a, m9=10 * M_EARTH, a9=500.0, e9=0.6, i9=0.0, w9=0.0, O9=0.0, j2=True,
                 n_tno=96, n_p9=96, soft=2.0, n_fine=256, d_fine=15.0):
        self.a, self.m9, self.a9, self.e9, self.i9 = a, m9, a9, e9, i9
        self.w9, self.O9, self.j2 = w9, O9, j2
        self.n_tno, self.n_p9, self.soft2 = n_tno, n_p9, soft**2
        self.n_fine, self.d_fine = n_fine, d_fine
        self.n_refined = 0
        self.dw9, self.dO9 = p9_precession(a9, e9) if j2 else (0.0, 0.0)
        self.L = np.sqrt(G * M_SUN * a)          # per unit mass

    def p9_points(self, t, n):
        ev, jv = p9_vectors(self.e9, self.i9, self.w9 + self.dw9 * t, self.O9 + self.dO9 * t)
        r9, _, w9 = orbit_points(self.a9, ev, jv, n)
        return r9, w9

    def rhs(self, t, y):
        """y = (e vector, j vector) of the TNO; returns their time derivatives (per year).

        The double average uses n_tno x n_p9 points; where the two orbits pass within d_fine au
        of each other, the integrand is sharply peaked and both orbits are resampled with
        n_fine points (the softening length soft regularises the crossing itself)."""
        evec, jvec = y[:3], y[3:]
        F, r, v, wt, dmin = self._force(t, evec, jvec, self.n_tno, self.n_p9)
        if dmin < self.d_fine:
            F, r, v, wt, _ = self._force(t, evec, jvec, self.n_fine, self.n_fine)
            self.n_refined += 1
        if self.j2:
            rn = np.linalg.norm(r, axis=1)
            z = r[:, 2]
            k = G * M_SUN * J2R2
            # Phi = k (3 z^2 / r^2 - 1) / (2 r^3), so F = -grad Phi
            F = F + k * (r * ((7.5 * z**2 / rn**2 - 1.5) / rn**5)[:, None]
                         - np.outer(3 * z / rn**5, [0, 0, 1.0]))
        h = np.cross(r, v)
        hdot = wt @ np.cross(r, F)
        edot = wt @ (np.cross(F, h) + np.cross(v, np.cross(r, F))) / (G * M_SUN)
        return np.concatenate([edot, hdot / self.L])

    def _force(self, t, evec, jvec, n1, n2):
        r, v, wt = orbit_points(self.a, evec, jvec, n1)
        r9, w9 = self.p9_points(t, n2)
        d = r9[None, :, :] - r[:, None, :]
        d2 = np.sum(d * d, axis=2)
        dist3 = (d2 + self.soft2) ** 1.5
        r93 = np.linalg.norm(r9, axis=1) ** 3
        F = G * self.m9 * (np.einsum("nmk,nm->nk", d, w9[None, :] / dist3)
                           - np.sum(w9[:, None] * r9 / r93[:, None], axis=0)[None, :])
        return F, r, v, wt, np.sqrt(d2.min())


def elements_to_vectors(e, i, w, O):
    ev, jv = p9_vectors(e, i, w, O)
    return np.concatenate([ev, jv])


def vectors_to_elements(y):
    """(e, i, w, O, varpi, theta_e) in radians from (e vector, j vector); theta_e is the
    direction of the eccentricity vector projected on the reference plane (LHPH18's
    'geometrical longitude of pericentre')."""
    ev, jv = y[..., :3], y[..., 3:]
    e = np.linalg.norm(ev, axis=-1)
    jn = np.linalg.norm(jv, axis=-1)
    inc = np.arccos(np.clip(jv[..., 2] / jn, -1, 1))
    O = np.arctan2(jv[..., 0], -jv[..., 1])
    nodev = np.stack([np.cos(O), np.sin(O), np.zeros_like(O)], -1)
    w = np.arctan2(np.sum(np.cross(nodev, ev) * jv, -1) / jn, np.sum(nodev * ev, -1))
    theta_e = np.arctan2(ev[..., 1], ev[..., 0])
    return e, inc, w % (2 * np.pi), O % (2 * np.pi), (w + O) % (2 * np.pi), theta_e % (2 * np.pi)


RP_STOP = 10.0     # au: integrations stop if the pericentre falls below this (see evolve)


def evolve(args):
    """Integrate one TNO's secular evolution. args = (a, e, i, w, O, t_end_yr, n_out, kwargs for
    Secular). Returns the output times and the element history (e, i, w, O, varpi, theta_e) and
    Planet Nine's longitude of pericentre at each time.

    The integration stops if the pericentre falls below RP_STOP: such a TNO would be scattered
    by the giant planets, which the secular model cannot follow (LHPH18 Sec. 2.2), and as
    e -> 1 the equations become stiff. Outputs after that time are nan, and t_stop records it
    (t_stop = t_end for TNOs that last the whole run)."""
    a, e, i, w, O, t_end, n_out, kw = args
    sec = Secular(a, **kw)
    y0 = elements_to_vectors(e, i, w, O)
    t_eval = np.linspace(0, t_end, n_out)
    low_peri = lambda t, y: a * (1 - np.linalg.norm(y[:3])) - RP_STOP
    low_peri.terminal = True
    sol = solve_ivp(sec.rhs, (0, t_end), y0, method="DOP853", t_eval=t_eval, rtol=1e-8, atol=1e-10, events=low_peri)
    stopped = sol.status == 1
    y = np.full((n_out, 6), np.nan)
    y[:len(sol.t)] = sol.y.T
    els = vectors_to_elements(y)
    varpi9 = (sec.w9 + sec.O9 + (sec.dw9 + sec.dO9) * t_eval) % (2 * np.pi)
    return dict(t=t_eval, e=els[0], i=els[1], w=els[2], O=els[3], varpi=els[4], theta_e=els[5],
                varpi9=varpi9, O9=(sec.O9 + sec.dO9 * t_eval) % (2 * np.pi), w9=(sec.w9 + sec.dw9 * t_eval) % (2 * np.pi),
                ok=sol.status >= 0, t_stop=float(sol.t_events[0][0]) if stopped else float(t_end))


def anti_aligned_ensemble(n, seed=2018, a_range=(150.0, 550.0), rp_range=(30.0, 50.0), inc_deg=10.0):
    """LHPH18 Sec. 2.2: TNOs with a and pericentre distance uniform in the given ranges,
    i = 10 deg, longitude of pericentre 180 deg (anti-aligned with Planet Nine's, which is
    0), and argument of pericentre uniform (so that Omega = 180 deg - omega)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(*a_range, n)
    rp = rng.uniform(*rp_range, n)
    w = rng.uniform(0, 2 * np.pi, n)
    return dict(a=a, e=1 - rp / a, i=np.full(n, np.radians(inc_deg)), w=w, O=(np.pi - w) % (2 * np.pi))


def flip_run(args):
    """LHPH18 Sec. 4.1 (their Figure 9): does a TNO with semi-major axis a and pericentre rp,
    started at 5 deg inclination with omega = 180 deg and Omega = 0 (anti-aligned), flip past
    90 deg within t_end? args = (a, rp, j2, t_end). Returns (flipped, max inclination in deg)."""
    a, rp, j2, t_end = args
    r = evolve((a, 1 - rp / a, np.radians(5.0), np.pi, 0.0, t_end, 400, dict(j2=j2)))
    imax = float(np.nanmax(np.degrees(r["i"])))
    return bool(imax > 90), imax


def ensemble_member(args):
    """One member of the anti-aligned ensemble: returns its history at n_out times as float32
    arrays (e, i, and the angles relative to Planet Nine's: varpi, theta_e, omega, Omega; nan
    after the integration stops) and the time it stopped."""
    a, e, i, w, O, t_end, n_out = args
    r = evolve((a, e, i, w, O, t_end, n_out, {}))
    rel = lambda x, ref: ((x - ref) % (2 * np.pi)).astype(np.float32)
    return dict(e=r["e"].astype(np.float32), i=r["i"].astype(np.float32), dvarpi=rel(r["varpi"], r["varpi9"]),
                dtheta=rel(r["theta_e"], r["varpi9"]), dw=rel(r["w"], r["w9"]), dO=rel(r["O"], r["O9"]), t_stop=r["t_stop"])
