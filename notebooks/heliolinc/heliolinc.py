"""Helper module for the HelioLinC notebook.

Follows M. J. Holman, M. J. Payne, P. Blankley, R. Janssen & S. Kuindersma
(2018), "HelioLinC: A Novel Approach to the Minor Planet Linking Problem",
AJ 156, 135 ("H18"), which builds on the formalism of Bernstein & Khushalani
(2000, AJ 120, 3323).

Contents:
  * two-body propagation of real MPCORB orbits, and an approximate Earth orbit;
  * a simple simulated survey that turns those orbits into tracklets;
  * the H18 transformation of each tracklet into an "arrow" (alpha, alpha-dot,
    beta, beta-dot) at a common reference time, for assumed (gamma, gamma-dot);
  * KD-tree clustering of arrows with the H18 metric, and scoring against the
    known identities of the simulated tracklets.

Units: au, days and radians. Coordinates are heliocentric ecliptic J2000.
"""
import numpy as np
from scipy.spatial import cKDTree

K_GAUSS = 0.01720209895            # Gaussian gravitational constant (au^1.5 / day)
GM_SUN = K_GAUSS ** 2              # au^3 / day^2
C_AU_PER_DAY = 173.1446326846693   # speed of light
OBLIQUITY = np.radians(23.4392911) # J2000 mean obliquity of the ecliptic
J2000 = 2451545.0
ARCSEC = np.pi / 180 / 3600


# --- Orbits -----------------------------------------------------------------------

def load_orbits(path):
    """Read the MPCORB extract written by tools/fetch_heliolinc_orbits.py.

    Returns (names, orbits, meta): names an array of designations; orbits a dict
    of arrays (a, e, inc, node, peri, M in au and radians, epoch as a JD, H, G);
    meta the `# key = value` lines of the header, as floats.
    """
    import csv
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    meta, rows = {}, []
    with opener(path, "rt") as f:
        lines = []
        for line in f:
            if line.startswith("#"):
                if "=" in line:
                    k, v = line[1:].split("=", 1)
                    try:
                        meta[k.strip()] = float(v)
                    except ValueError:
                        pass
            else:
                lines.append(line)
    reader = csv.reader(lines)
    next(reader)
    for r in reader:
        rows.append(r)
    names = np.array([r[0] for r in rows], dtype=object)
    H, G, epoch, a, e, inc, node, peri, M = np.array([[float(x) for x in r[1:]] for r in rows]).T
    orbits = {"a": a, "e": e, "inc": np.radians(inc), "node": np.radians(node), "peri": np.radians(peri),
              "M": np.radians(M), "epoch": epoch, "H": H, "G": G}
    return names, orbits, meta


def subset(orbits, idx):
    return {k: v[idx] for k, v in orbits.items()}


def true_arrow(orbits, t_ref, frame):
    """The exact (alpha, beta, alpha_dot, beta_dot, gamma, gamma_dot) of two-body
    orbits at t_ref in an H18 frame (rows x, y, z), from H18 Eq. (4)."""
    h = 1e-3
    r0 = kepler_positions(orbits["a"], orbits["e"], orbits["inc"], orbits["node"], orbits["peri"],
                          orbits["M"], orbits["epoch"], t_ref) @ frame.T
    rp = kepler_positions(orbits["a"], orbits["e"], orbits["inc"], orbits["node"], orbits["peri"],
                          orbits["M"], orbits["epoch"], t_ref + h) @ frame.T
    rm = kepler_positions(orbits["a"], orbits["e"], orbits["inc"], orbits["node"], orbits["peri"],
                          orbits["M"], orbits["epoch"], t_ref - h) @ frame.T
    v0 = (rp - rm) / (2 * h)
    z0 = r0[..., 2]
    return np.stack([r0[..., 0] / z0, r0[..., 1] / z0, v0[..., 0] / z0, v0[..., 1] / z0,
                     1 / z0, v0[..., 2] / z0], axis=-1)

def solve_kepler(M, e, tol=1e-13):
    """Eccentric anomaly E from mean anomaly M (radians), for arrays, e < 1."""
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(60):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def kepler_positions(a, e, inc, node, peri, M0, epoch, t, gm=GM_SUN):
    """Heliocentric ecliptic positions, shape (..., 3), of two-body orbits at times t.

    Angles in radians. a, e, ... may be arrays of objects; t a scalar or an array
    that broadcasts against them.
    """
    n = np.sqrt(gm / a ** 3)
    M = np.mod(M0 + n * (t - epoch), 2 * np.pi)
    E = solve_kepler(M, e * np.ones_like(M))
    xp = a * (np.cos(E) - e)
    yp = a * np.sqrt(1 - e * e) * np.sin(E)
    cO, sO, cw, sw, ci, si = np.cos(node), np.sin(node), np.cos(peri), np.sin(peri), np.cos(inc), np.sin(inc)
    x = (cO * cw - sO * sw * ci) * xp + (-cO * sw - sO * cw * ci) * yp
    y = (sO * cw + cO * sw * ci) * xp + (-sO * sw + cO * cw * ci) * yp
    z = (sw * si) * xp + (cw * si) * yp
    return np.stack([x, y, z], axis=-1)


def earth_position(t):
    """Approximate heliocentric ecliptic J2000 position of the Earth-Moon barycentre.

    Uses the Keplerian elements and rates of Standish & Williams, "Approximate
    Positions of the Planets" (JPL; Table 1, valid 1800-2050), good to about 20".
    Here the barycentre stands in for the observer, so diurnal parallax and the
    Moon's offset are ignored.
    """
    T = (np.asarray(t, dtype=float) - J2000) / 36525.0
    a = 1.00000261 + 0.00000562 * T
    e = 0.01671123 - 0.00004392 * T
    inc = np.radians(-0.00001531 - 0.01294668 * T)
    L = np.radians(100.46457166 + 35999.37244981 * T)
    varpi = np.radians(102.93768193 + 0.32327364 * T)
    node = np.zeros_like(T)
    M = np.mod(L - varpi, 2 * np.pi)
    E = solve_kepler(M, e * np.ones_like(M))
    xp, yp = a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E)
    peri = varpi - node
    cw, sw, ci, si = np.cos(peri), np.sin(peri), np.cos(inc), np.sin(inc)
    return np.stack([cw * xp - sw * yp, (sw * xp + cw * yp) * ci, (sw * xp + cw * yp) * si], axis=-1)


def ecliptic_to_equatorial(v):
    c, s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    return np.stack([v[..., 0], c * v[..., 1] - s * v[..., 2], s * v[..., 1] + c * v[..., 2]], axis=-1)


def equatorial_to_ecliptic(v):
    c, s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    return np.stack([v[..., 0], c * v[..., 1] + s * v[..., 2], -s * v[..., 1] + c * v[..., 2]], axis=-1)


def radec(u_eq):
    """RA, Dec (radians) of equatorial unit vectors."""
    return np.mod(np.arctan2(u_eq[..., 1], u_eq[..., 0]), 2 * np.pi), np.arcsin(np.clip(u_eq[..., 2], -1, 1))


def unit_radec(ra, dec):
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


def classify(a, e):
    """Coarse dynamical classes from a (au) and e, by simple cuts."""
    q = a * (1 - e)
    cls = np.full(a.shape, "Other", dtype=object)
    cls[(a > 1.8) & (a < 3.7)] = "MBA"
    cls[(a >= 3.7) & (a < 4.2)] = "Hilda"
    cls[(a >= 5.05) & (a < 5.35)] = "Trojan"
    cls[(a >= 5.5) & (a < 30.1)] = "Centaur"
    cls[a >= 30.1] = "TNO"
    cls[q < 1.3] = "NEO"
    return cls


def hg_magnitude(H, G, r, delta, phase):
    """Apparent V magnitude in the IAU H, G system (Bowell et al. 1989)."""
    th = np.tan(phase / 2)
    phi1 = np.exp(-3.33 * th ** 0.63)
    phi2 = np.exp(-1.87 * th ** 1.22)
    return H + 5 * np.log10(r * delta) - 2.5 * np.log10((1 - G) * phi1 + G * phi2)


def new_moon_jde(k):
    """Meeus, Astronomical Algorithms (2nd ed.), Chapter 49: JDE of new moon k."""
    T = k / 1236.85
    jde = (2451550.09766 + 29.530588861 * k + 0.00015437 * T ** 2
           - 0.000000150 * T ** 3 + 0.00000000073 * T ** 4)
    E = 1 - 0.002516 * T - 0.0000074 * T ** 2
    rad = np.radians
    M = rad(2.5534 + 29.10535670 * k - 0.0000014 * T ** 2 - 0.00000011 * T ** 3)
    Mp = rad(201.5643 + 385.81693528 * k + 0.0107582 * T ** 2 + 0.00001238 * T ** 3 - 0.000000058 * T ** 4)
    F = rad(160.7108 + 390.67050284 * k - 0.0016118 * T ** 2 - 0.00000227 * T ** 3 + 0.000000011 * T ** 4)
    Om = rad(124.7746 - 1.56375588 * k + 0.0020672 * T ** 2 + 0.00000215 * T ** 3)
    corr = (-0.40720 * np.sin(Mp) + 0.17241 * E * np.sin(M) + 0.01608 * np.sin(2 * Mp)
            + 0.01039 * np.sin(2 * F) + 0.00739 * E * np.sin(Mp - M) - 0.00514 * E * np.sin(Mp + M)
            + 0.00208 * E * E * np.sin(2 * M) - 0.00111 * np.sin(Mp - 2 * F) - 0.00057 * np.sin(Mp + 2 * F)
            + 0.00056 * E * np.sin(2 * Mp + M) - 0.00042 * np.sin(3 * Mp) + 0.00042 * E * np.sin(M + 2 * F)
            + 0.00038 * E * np.sin(M - 2 * F) - 0.00024 * E * np.sin(2 * Mp - M) - 0.00017 * np.sin(Om)
            - 0.00007 * np.sin(Mp + 2 * M) + 0.00004 * np.sin(2 * Mp - 2 * F) + 0.00004 * np.sin(3 * M)
            + 0.00003 * np.sin(Mp + M - 2 * F) + 0.00003 * np.sin(2 * Mp + 2 * F) - 0.00003 * np.sin(Mp + M + 2 * F)
            + 0.00003 * np.sin(Mp - M + 2 * F) - 0.00002 * np.sin(Mp - M - 2 * F) - 0.00002 * np.sin(3 * Mp + M)
            + 0.00002 * np.sin(4 * Mp))
    A = [299.77 + 0.107408 * k - 0.009173 * T ** 2, 251.88 + 0.016321 * k, 251.83 + 26.651886 * k,
         349.42 + 36.412478 * k, 84.66 + 18.206239 * k, 141.74 + 53.303771 * k, 207.14 + 2.453732 * k,
         154.84 + 7.306860 * k, 34.52 + 27.261239 * k, 207.19 + 0.121824 * k, 291.34 + 1.844379 * k,
         161.72 + 24.198154 * k, 239.56 + 25.513099 * k, 331.55 + 3.592518 * k]
    W = [325, 165, 164, 126, 110, 62, 60, 56, 47, 42, 40, 37, 35, 23]
    corr += sum(w * 1e-6 * np.sin(rad(a)) for w, a in zip(W, A))
    return jde + corr


def nearest_new_moon(jd):
    """JDE (TT) of the new moon nearest to JD `jd`."""
    k = np.round((jd - 2451550.09766) / 29.530588861)
    return new_moon_jde(k)


# --- Simulated observations --------------------------------------------------------

def observe(orbits, t, n_light=2):
    """Astrometric (light-time corrected) geocentric directions of every orbit at times t.

    `orbits` is a dict of arrays (a, e, inc, node, peri, M, epoch in au, rad, JD).
    t may be a scalar, or an array matching the orbits. Returns the ecliptic unit
    vectors, heliocentric distance r, geocentric distance delta and phase angle.
    """
    t = np.asarray(t, dtype=float)
    RE = earth_position(t)
    tau = 0.0
    for _ in range(n_light + 1):
        r_vec = kepler_positions(orbits["a"], orbits["e"], orbits["inc"], orbits["node"],
                                 orbits["peri"], orbits["M"], orbits["epoch"], t - tau)
        rho_vec = r_vec - RE
        delta = np.linalg.norm(rho_vec, axis=-1)
        tau = delta / C_AU_PER_DAY
    r = np.linalg.norm(r_vec, axis=-1)
    cos_phase = np.sum(r_vec * rho_vec, axis=-1) / (r * delta)
    return rho_vec / delta[..., None], r, delta, np.arccos(np.clip(cos_phase, -1, 1))


def simulate_survey(orbits, nights, field, *, spacing_min=20.0, n_per_tracklet=3,
                    v_limit=21.5, v_width=0.15, sigma_arcsec=0.1, seed=1):
    """Tracklets of every orbit that falls in a field on the given nights.

    `field` = (lam0, beta0, half_width) in radians: a square of side 2*half_width
    in gnomonic (tangent-plane) coordinates about ecliptic longitude lam0 and
    latitude beta0. On each night, an object in the field is detected with
    probability 1 / (1 + exp((V - v_limit) / v_width)) and, if detected, yields
    one tracklet of `n_per_tracklet` detections `spacing_min` minutes apart, each
    with Gaussian astrometric noise of `sigma_arcsec` in each coordinate.

    Returns a dict of per-detection arrays: t (JD), ra, dec (radians),
    tracklet (index), obj (index into the orbits) and V.
    """
    rng = np.random.default_rng(seed)
    lam0, beta0, half = field
    centre = unit_radec(lam0, beta0)          # same formula, in ecliptic coordinates
    ex = np.array([-np.sin(lam0), np.cos(lam0), 0.0])
    ey = np.cross(centre, ex)
    out = {k: [] for k in ("t", "ra", "dec", "tracklet", "obj", "V")}
    n_tracklets = 0
    nobj = len(orbits["a"])
    for t0 in nights:
        u, r, delta, phase = observe(orbits, t0)
        cz = u @ centre
        gx, gy = (u @ ex) / cz, (u @ ey) / cz
        V = hg_magnitude(orbits["H"], orbits["G"], r, delta, phase)
        inside = (cz > 0) & (np.abs(gx) < half) & (np.abs(gy) < half)
        p_det = 1 / (1 + np.exp(np.clip((V - v_limit) / v_width, -50, 50)))
        seen = np.flatnonzero(inside & (rng.random(nobj) < p_det))
        if seen.size == 0:
            continue
        sub = {k: v[seen] for k, v in orbits.items()}
        for j in range(n_per_tracklet):
            tj = t0 + j * spacing_min / 1440.0
            uj, *_ = observe(sub, np.full(seen.size, tj))
            ra, dec = radec(ecliptic_to_equatorial(uj))
            dec_n = dec + rng.normal(0, sigma_arcsec * ARCSEC, seen.size)
            ra_n = ra + rng.normal(0, sigma_arcsec * ARCSEC, seen.size) / np.cos(dec)
            out["t"].append(np.full(seen.size, tj)); out["ra"].append(ra_n); out["dec"].append(dec_n)
            out["tracklet"].append(n_tracklets + np.arange(seen.size)); out["obj"].append(seen)
            out["V"].append(V[seen])
        n_tracklets += seen.size
    det = {k: np.concatenate(v) for k, v in out.items()}
    order = np.lexsort((det["t"], det["tracklet"]))
    return {k: v[order] for k, v in det.items()}


def tracklet_table(det):
    """One row per tracklet: object, mean time, mean RA/Dec, night (integer JD) and
    sky-plane rates dra (RA rate times cos Dec) and ddec, in radians per day."""
    tid = det["tracklet"]
    n = tid.max() + 1
    cnt = np.bincount(tid, minlength=n)
    u = unit_radec(det["ra"], det["dec"])
    um = np.stack([np.bincount(tid, u[:, k], n) for k in range(3)], axis=-1)
    ra, dec = radec(um / np.linalg.norm(um, axis=-1, keepdims=True))
    obj = np.zeros(n, dtype=int); obj[tid] = det["obj"]
    t = np.bincount(tid, det["t"], n) / cnt
    first = np.r_[True, tid[1:] != tid[:-1]]
    last = np.r_[tid[1:] != tid[:-1], True]
    span = det["t"][last] - det["t"][first]
    dra = np.angle(np.exp(1j * (det["ra"][last] - det["ra"][first]))) * np.cos(dec) / span
    ddec = (det["dec"][last] - det["dec"][first]) / span
    return {"obj": obj, "t": t, "ra": ra, "dec": dec, "night": np.floor(t - 0.5).astype(int), "n": cnt,
            "dra": dra, "ddec": ddec}


def select_tracklets(det, keep):
    """The detections of the tracklets where `keep` (one flag per tracklet) is
    true, with tracklets renumbered from 0."""
    keep = np.asarray(keep, dtype=bool)
    new_id = np.cumsum(keep) - 1
    m = keep[det["tracklet"]]
    out = {k: v[m] for k, v in det.items()}
    out["tracklet"] = new_id[out["tracklet"]]
    return out


# --- The HelioLinC transformation (H18 Section 2 and Appendix) ---------------------

def heliocentric_from_topocentric(u, R_obs, r):
    """H18 Appendix, Eqs. (A1)-(A4): heliocentric position of an object seen along
    unit vector u from heliocentric observer position R_obs, at assumed
    heliocentric distance r. Returns NaN where no positive-distance solution exists."""
    robs = np.linalg.norm(R_obs, axis=-1)
    cos_f = -np.sum(u * R_obs, axis=-1) / robs        # cos f = -rhat . rhat_obs
    disc = r ** 2 - robs ** 2 * (1 - cos_f ** 2)
    rho = robs * cos_f + np.sqrt(np.where(disc >= 0, disc, np.nan))
    rho = np.where(rho > 0, rho, np.nan)
    return R_obs + rho[..., None] * u


def reference_frame(z_dir):
    """H18 frame: z toward the reference direction, x along increasing ecliptic
    longitude, y completing a right-handed set. Returns the 3x3 matrix whose rows
    are x, y, z."""
    z = z_dir / np.linalg.norm(z_dir)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    return np.array([x, np.cross(z, x), z])


def prepare(det, t_ref, gamma, z_dir=None):
    """Quantities for fitting arrows that depend only on the observations and gamma.

    The reference direction defaults to the mean heliocentric direction of the
    detections at the assumed distance 1/gamma (H18 use HEALPix centres instead).
    Returns a dict with theta_x, theta_y, the observer position in the frame and
    the times relative to t_ref.
    """
    u = equatorial_to_ecliptic(unit_radec(det["ra"], det["dec"]))
    RE = earth_position(det["t"])
    if z_dir is None:
        h = heliocentric_from_topocentric(u, RE, 1 / gamma)
        z_dir = np.nanmean(h / np.linalg.norm(h, axis=-1, keepdims=True), axis=0)
    F = reference_frame(z_dir)
    uf, Ef = u @ F.T, RE @ F.T
    return {"tid": det["tracklet"], "theta_x": uf[:, 0] / uf[:, 2], "theta_y": uf[:, 1] / uf[:, 2],
            "E": Ef, "t": det["t"] - t_ref, "frame": F, "gamma": gamma}


def heliocentric_tangent(p, gamma_dot, gravity=True):
    """The right-hand sides of H18 Eq. (6) for every detection: its position in the
    heliocentric tangent plane, (x/z0, y/z0), at the gamma used in `prepare` and
    the given gamma_dot. With gravity, only the dominant term g_z = -sigma t^2 / 2,
    with sigma = GM gamma^3, is kept (H18's second option). Also returns t'."""
    g = p["gamma"]
    tp = p["t"] - 1 / (C_AU_PER_DAY * g)        # H18: t' ~ t - 1/(c gamma)
    gz = -0.5 * GM_SUN * g ** 3 * tp ** 2 if gravity else 0.0
    bracket = 1 + gamma_dot * tp + g * gz - g * p["E"][:, 2]
    return p["theta_x"] * bracket + g * p["E"][:, 0], p["theta_y"] * bracket + g * p["E"][:, 1], tp


def fit_arrows(p, gamma_dot, gravity=True):
    """H18 Eq. (6), fitted by least squares for each tracklet: alpha + alpha_dot t'
    and beta + beta_dot t' against the heliocentric tangent-plane positions.

    Returns an (N_tracklets, 4) array of arrows (alpha, beta, alpha_dot, beta_dot).
    """
    yx, yy, tp = heliocentric_tangent(p, gamma_dot, gravity)
    tid = p["tid"]
    n = tid.max() + 1
    S0 = np.bincount(tid, minlength=n).astype(float)
    S1 = np.bincount(tid, tp, n)
    S2 = np.bincount(tid, tp * tp, n)
    det = S0 * S2 - S1 ** 2
    out = np.empty((n, 4))
    for k, y in enumerate((yx, yy)):
        Sy, Sty = np.bincount(tid, y, n), np.bincount(tid, tp * y, n)
        out[:, k] = (S2 * Sy - S1 * Sty) / det           # alpha or beta
        out[:, k + 2] = (S0 * Sty - S1 * Sy) / det       # alpha_dot or beta_dot
    return out


# --- Clustering (H18 Section 3.1.3, Eq. 11) ----------------------------------------

def find_clusters(arrows, d, dt, min_size=3):
    """Clusters of arrows: for each arrow, all arrows within distance d of it under
    the H18 metric d^2 = dalpha^2 + dbeta^2 + dt^2 (dalpha_dot^2 + dbeta_dot^2),
    kept when they number at least `min_size`. Returns a set of frozensets of
    tracklet indices."""
    pts = np.column_stack([arrows[:, 0], arrows[:, 1], dt * arrows[:, 2], dt * arrows[:, 3]])
    ok = np.all(np.isfinite(pts), axis=1)
    idx = np.flatnonzero(ok)
    tree = cKDTree(pts[ok])
    out = set()
    for nb in tree.query_ball_point(pts[ok], r=d):
        if len(nb) >= min_size:
            out.add(frozenset(idx[nb].tolist()))
    return out


def remove_subsets(clusters):
    """Drop clusters that are proper subsets of other clusters (H18 Section 5.1)."""
    by_member = {}
    for c in clusters:
        for m in c:
            by_member.setdefault(m, []).append(c)
    keep = []
    for c in clusters:
        first = min(c)
        if not any(len(o) > len(c) and c < o for o in by_member[first]):
            keep.append(c)
    return keep


def all_arrows(det, t_ref, gammas, gamma_dots, z_dir=None):
    """Arrows for every (gamma, gamma_dot) hypothesis, as a list of arrays.

    `gamma_dots` is a list of values, or a function of gamma returning them."""
    out = []
    for g in np.atleast_1d(gammas):
        p = prepare(det, t_ref, g, z_dir)
        for gd in np.atleast_1d(gamma_dots(g) if callable(gamma_dots) else gamma_dots):
            out.append(fit_arrows(p, gd))
    return out


def cluster_arrows(arrow_sets, d, dt, min_size=3):
    """Clusters from every set of arrows, merged, with proper subsets removed."""
    clusters = set()
    for arrows in arrow_sets:
        clusters |= find_clusters(arrows, d, dt, min_size)
    return remove_subsets(clusters)


def run_heliolinc(det, t_ref, gammas, gamma_dots, d, dt, min_size=3, z_dir=None):
    """The whole H18 search: transform, fit arrows, cluster, merge."""
    return cluster_arrows(all_arrows(det, t_ref, gammas, gamma_dots, z_dir), d, dt, min_size)


def score(clusters, tracklet_obj, min_size=3):
    """Score clusters against the true identities of their tracklets.

    Follows the H18 dispositions: a cluster is "pure" if all its tracklets belong
    to one object, "valid" if it contains at least `min_size` tracklets of one
    object plus others, and "erroneous" otherwise. Returns the number of objects
    found in at least one pure cluster, the number with at least `min_size`
    tracklets in a pure or valid cluster, the number of erroneous clusters, and
    the total number of clusters.
    """
    pure, valid, n_err = set(), set(), 0
    for c in clusters:
        objs, counts = np.unique(tracklet_obj[list(c)], return_counts=True)
        real = objs >= 0                     # false tracklets are labelled -1
        if objs.size == 1 and real[0]:
            pure.add(objs[0]); valid.add(objs[0])
        elif np.any(counts[real] >= min_size):
            valid.update(objs[real & (counts >= min_size)].tolist())
        else:
            n_err += 1
    return {"pure_objects": len(pure), "valid_objects": len(valid),
            "erroneous": n_err, "clusters": len(clusters)}


def findable(tracklet_obj, tracklet_night, min_size=3):
    """Objects with at least `min_size` tracklets (on different nights) in the data."""
    pairs = np.unique(np.column_stack([tracklet_obj, tracklet_night]), axis=0)
    objs, counts = np.unique(pairs[:, 0], return_counts=True)
    return objs[counts >= min_size]


# --- False tracklets (not part of H18) -------------------------------------------

def add_false_tracklets(det, n_false, field, nights, *, rate_source=None, spacing_min=20.0,
                        n_per_tracklet=3, sigma_arcsec=0.1, seed=2):
    """Append `n_false` false tracklets, labelled obj = -1, to a set of detections.

    Each is placed uniformly at random in the field on a random night, and moves
    in a straight line on the sky at a rate drawn at random from the real
    tracklets' sky-plane rates (`rate_source`, defaulting to those of `det`),
    as an unlinkable artefact with realistic motion would.
    """
    rng = np.random.default_rng(seed)
    lam0, beta0, half = field
    centre = unit_radec(lam0, beta0)
    ex = np.array([-np.sin(lam0), np.cos(lam0), 0.0])
    ey = np.cross(centre, ex)
    src = det if rate_source is None else rate_source
    # sky-plane rates (gnomonic, radians/day) of the real tracklets
    u = equatorial_to_ecliptic(unit_radec(src["ra"], src["dec"]))
    gx, gy = (u @ ex) / (u @ centre), (u @ ey) / (u @ centre)
    tid = src["tracklet"]
    first = np.r_[True, tid[1:] != tid[:-1]]
    last = np.r_[tid[1:] != tid[:-1], True]
    span = src["t"][last] - src["t"][first]
    rates = np.column_stack([(gx[last] - gx[first]) / span, (gy[last] - gy[first]) / span])
    pick = rates[rng.integers(0, len(rates), n_false)]
    x0, y0 = rng.uniform(-half, half, n_false), rng.uniform(-half, half, n_false)
    t0 = np.asarray(nights)[rng.integers(0, len(nights), n_false)]
    new = {k: [] for k in ("t", "ra", "dec", "tracklet", "obj", "V")}
    base = det["tracklet"].max() + 1
    for j in range(n_per_tracklet):
        dtj = j * spacing_min / 1440.0
        X = x0 + pick[:, 0] * dtj + rng.normal(0, sigma_arcsec * ARCSEC, n_false)
        Y = y0 + pick[:, 1] * dtj + rng.normal(0, sigma_arcsec * ARCSEC, n_false)
        v = centre + X[:, None] * ex + Y[:, None] * ey
        ra, dec = radec(ecliptic_to_equatorial(v / np.linalg.norm(v, axis=-1, keepdims=True)))
        new["t"].append(t0 + dtj); new["ra"].append(ra); new["dec"].append(dec)
        new["tracklet"].append(base + np.arange(n_false)); new["obj"].append(np.full(n_false, -1))
        new["V"].append(np.full(n_false, np.nan))
    out = {k: np.concatenate([det[k]] + new[k]) for k in det}
    order = np.lexsort((out["t"], out["tracklet"]))
    return {k: v[order] for k, v in out.items()}
