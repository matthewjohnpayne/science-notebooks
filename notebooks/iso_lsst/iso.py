"""Helper module for the notebook "Interstellar objects seen by the LSST".

It reproduces the Monte Carlo of Hoover, Seligman & Payne (2022, PSJ 3, 71;
HSP22): a population of 'Oumuamua-like interstellar objects (ISOs) is drawn
from the Galactic kinematics of nearby stars, each object is followed through
the inner Solar System, and we ask when it would be bright enough, high
enough and in a dark enough sky for the Vera C. Rubin Observatory's Legacy
Survey of Space and Time (LSST) to see it.

Units: au, days and km/s (converted where needed). Positions are in the J2000
ecliptic frame unless a name says otherwise.

The functions that run in worker processes (`simulate_chunk`) live here,
rather than in the notebook, so that multiprocessing works on macOS and
Windows.
"""
import numpy as np

# ---------------------------------------------------------------- constants
AU_KM = 1.495978707e8
DAY_S = 86400.0
KMS_TO_AUD = DAY_S / AU_KM             # km/s -> au/day
GM_SUN = 1.32712440018e11 * DAY_S**2 / AU_KM**3   # au^3/day^2
V_ESC_1AU = np.sqrt(2 * GM_SUN) / KMS_TO_AUD      # 42.1 km/s
OBLIQUITY = np.radians(23.4392911)     # J2000 mean obliquity (IAU 1980 value)
JD2000 = 2451545.0

# Rubin Observatory, Cerro Pachón
RUBIN_LAT = np.radians(-30.2446)

# The ICRS -> Galactic rotation matrix (Hipparcos catalogue, ESA 1997, Vol. 1, Sec. 1.5.3).
# Galactic (x towards l=0, y towards l=90 deg, z towards the north Galactic pole) = A_G @ ICRS.
A_G = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                [+0.4941094279, -0.4448296300, +0.7469822445],
                [-0.8676661490, -0.1980763734, +0.4559837762]])

# Equatorial (ICRS ~ J2000) -> ecliptic J2000: rotate about x by the obliquity.
_ce, _se = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
EQ_TO_ECL = np.array([[1, 0, 0], [0, _ce, _se], [0, -_se, _ce]])
GAL_TO_ECL = EQ_TO_ECL @ A_G.T
GAL_TO_EQ = A_G.T

# HSP22 Table 1: Schwarzschild velocity dispersions (km/s) and vertex deviations (deg),
# after Binney & Merrifield (1998).
POPULATIONS = {
    "M":   dict(sigma=(31.0, 23.0, 16.0), vertex=7.0),
    "G":   dict(sigma=(26.0, 18.0, 15.0), vertex=12.0),
    "O/B": dict(sigma=(12.0, 11.0, 9.0), vertex=36.0),
}

# The Sun's motion relative to the Local Standard of Rest, (U, V, W) in km/s.
# HSP22 do not quote the value they used; we adopt Schönrich, Binney & Dehnen (2010)
# and test Dehnen & Binney (1998) as an alternative.
SOLAR_MOTION = {
    "Schönrich et al. (2010)": (11.1, 12.24, 7.25),
    "Dehnen & Binney (1998)": (10.0, 5.25, 7.17),
}
DEFAULT_SOLAR = "Schönrich et al. (2010)"

R_START = 5.0          # au: every object starts on the 5 au sphere (HSP22 Sec. 2)
B_MAX = 5.0            # au: impact parameters drawn from p(b) ~ b^2 on (0, 5) au
BETA_SMOOTH = 2.0      # km/s: the smoothing beta in HSP22 Sec. 2
H_OUMUAMUA = 22.4      # 'Oumuamua's absolute magnitude, used for every object
T_SPAN = 3 * 365.25    # days: HSP22 integrate each object for 3 years
M_BRIGHT, M_FAINT = 16.0, 24.0        # LSST saturation and limiting magnitudes
DEC_MAX = np.radians(12.0)            # the Wide-Fast-Deep's nominal northern limit
ALT_MIN = np.radians(30.0)            # object altitude
SUN_ALT_MAX = np.radians(-18.0)       # astronomical twilight
T_GRID = np.round(np.arange(16.0, 24.51, 0.25), 2)   # thresholds for the web export


# ---------------------------------------------------------------- kinematics
def focusing(v_inf):
    """HSP22 Sec. 2: the gravitational-focusing enhancement xi(v_inf), v_inf in km/s."""
    return 1.0 + (V_ESC_1AU / (v_inf + BETA_SMOOTH)) ** 2


def schwarzschild_lsr(v_gal, pop="G"):
    """Unnormalised Schwarzschild density of Galactic velocities v_gal (..., 3) km/s,
    relative to the LSR, for one stellar population (HSP22 Sec. 2 and Table 1)."""
    p = POPULATIONS[pop]
    lv = np.radians(p["vertex"])
    # The principal axes are the (U, V) axes rotated by the vertex deviation about W.
    u = np.cos(lv) * v_gal[..., 0] + np.sin(lv) * v_gal[..., 1]
    w = -np.sin(lv) * v_gal[..., 0] + np.cos(lv) * v_gal[..., 1]
    s1, s2, s3 = p["sigma"]
    return np.exp(-0.5 * ((u / s1) ** 2 + (w / s2) ** 2 + (v_gal[..., 2] / s3) ** 2))


def incoming_density(v_helio_gal, pop="G", solar=DEFAULT_SOLAR, focus=True):
    """Relative probability that an ISO arrives with heliocentric Galactic velocity v (km/s):
    the Schwarzschild density, shifted by the solar motion, times the focusing factor."""
    v_sun = np.array(SOLAR_MOTION[solar])
    f = schwarzschild_lsr(v_helio_gal + v_sun, pop)
    if focus:
        f = f * focusing(np.linalg.norm(v_helio_gal, axis=-1))
    return f


def sample_velocities(n, rng, pop="G", solar=DEFAULT_SOLAR, focus=True, vmax=100.0):
    """Draw n heliocentric Galactic velocities (km/s) from incoming_density, by rejection.

    HSP22 tabulate the density on a 100^3 grid spanning -100 to 100 km/s in each
    component; we keep their box but sample the continuous density exactly.
    """
    p = POPULATIONS[pop]
    v_sun = np.array(SOLAR_MOTION[solar])
    lv = np.radians(p["vertex"])
    rot = np.array([[np.cos(lv), -np.sin(lv), 0], [np.sin(lv), np.cos(lv), 0], [0, 0, 1]])
    fmax = focusing(0.0) if focus else 1.0
    out = []
    have = 0
    while have < n:
        m = max(4 * (n - have), 100_000) if not focus else max(400 * (n - have), 1_000_000)
        w = rng.normal(size=(m, 3)) * np.array(p["sigma"])     # principal-axis frame
        v = w @ rot.T - v_sun                                      # heliocentric, Galactic frame
        keep = np.all(np.abs(v) <= vmax, axis=1)
        if focus:
            keep &= rng.random(m) * fmax < focusing(np.linalg.norm(v, axis=1))
        out.append(v[keep])
        have += keep.sum()
    return np.concatenate(out)[:n]


def unit_to_radec(u):
    """Unit vectors (..., 3) in the equatorial frame -> RA, Dec in degrees."""
    ra = np.degrees(np.arctan2(u[..., 1], u[..., 0])) % 360
    dec = np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1)))
    return ra, dec


def solar_apex(solar=DEFAULT_SOLAR):
    """RA, Dec (deg) of the solar apex: the direction of the Sun's motion relative to the LSR."""
    v = GAL_TO_EQ @ np.array(SOLAR_MOTION[solar])
    return unit_to_radec(v / np.linalg.norm(v))


def radiant_map(pop="G", solar=DEFAULT_SOLAR, focus=True, n_ra=180, n_dec=90, n_v=200, vmax=100.0):
    """Probability per unit solid angle that an ISO arrives FROM each direction (HSP22 Fig. 1).

    For each sky direction u (the radiant) the heliocentric velocity is v = -s u; we
    integrate the density over speed s in spherical coordinates (weight s^2).
    Returns RA edges, Dec edges (deg) and the map normalised to a maximum of 1.
    """
    ra_e = np.linspace(0, 360, n_ra + 1)
    dec_e = np.linspace(-90, 90, n_dec + 1)
    ra = np.radians(0.5 * (ra_e[1:] + ra_e[:-1]))
    dec = np.radians(0.5 * (dec_e[1:] + dec_e[:-1]))
    RA, DEC = np.meshgrid(ra, dec)
    u_eq = np.stack([np.cos(DEC) * np.cos(RA), np.cos(DEC) * np.sin(RA), np.sin(DEC)], -1)
    u_gal = u_eq @ A_G.T
    s = np.linspace(0, vmax, n_v + 1)[1:]
    dens = np.zeros(RA.shape)
    for sk in s:
        dens += incoming_density(-sk * u_gal, pop, solar, focus) * sk**2
    return ra_e, dec_e, dens / dens.max()


# ---------------------------------------------------------------- initial conditions
def initial_conditions(n, seed, pop="G", solar=DEFAULT_SOLAR, focus=True, b_power=2):
    """HSP22 Sec. 2: start n objects on the 5 au sphere, heading inwards.

    Each object gets a velocity at infinity from the Galactic distribution, an offset
    b perpendicular to it with p(b) ~ b^b_power on (0, 5) au, an azimuth psi for that
    offset, and a start date drawn uniformly through 2021. Its speed is raised to
    sqrt(v_inf^2 + v_esc(5 au)^2), keeping the direction of v_inf.
    Returns a dict of arrays; positions in au and velocities in au/day, ecliptic frame.
    """
    rng = np.random.default_rng(seed)
    v_gal = sample_velocities(n, rng, pop, solar, focus)
    b = B_MAX * rng.random(n) ** (1.0 / (b_power + 1))
    psi = 2 * np.pi * rng.random(n)
    jd0 = 2459215.5 + 365.0 * rng.random(n)     # 2021 January 1.0 to December 31
    r0, v0 = place(v_gal, b, psi)
    v_inf = np.linalg.norm(v_gal, axis=1)
    radiant_eq = (-v_gal @ GAL_TO_EQ.T) / v_inf[:, None]
    ra_rad, dec_rad = unit_to_radec(radiant_eq)
    return dict(r0=r0, v0=v0, v_inf=v_inf, b=b, jd0=jd0, ra_radiant=ra_rad, dec_radiant=dec_rad)


def place(v_gal, b, psi):
    """Starting state on the 5 au sphere for velocities at infinity v_gal (n, 3) in km/s
    (Galactic frame), offsets b (au) and azimuths psi (radians). Returns r0 (au) and
    v0 (au/day) in the ecliptic frame."""
    v_ecl = v_gal @ GAL_TO_ECL.T
    v_inf = np.linalg.norm(v_ecl, axis=1)
    vhat = v_ecl / v_inf[:, None]
    # Two unit vectors perpendicular to vhat; psi is the azimuth of b around vhat.
    ref = np.where(np.abs(vhat[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    e1 = np.cross(vhat, ref)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(vhat, e1)
    bvec = b[:, None] * (np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2)
    r0 = -np.sqrt(R_START**2 - b**2)[:, None] * vhat + bvec
    v_esc5 = V_ESC_1AU / np.sqrt(R_START)
    speed0 = np.sqrt(v_inf**2 + v_esc5**2)
    return r0, vhat * (speed0 * KMS_TO_AUD)[:, None]


# ---------------------------------------------------------------- two-body orbits
def elements(r0, v0):
    """Conic elements of heliocentric states (au, au/day): a, e, q, i (deg), the unit
    vectors P (to perihelion) and Q, and the time of perihelion passage relative to t=0."""
    r = np.linalg.norm(r0, axis=1)
    v2 = np.sum(v0**2, axis=1)
    h = np.cross(r0, v0)
    hn = np.linalg.norm(h, axis=1)
    evec = np.cross(v0, h) / GM_SUN - r0 / r[:, None]
    e = np.linalg.norm(evec, axis=1)
    a = 1.0 / (2.0 / r - v2 / GM_SUN)            # negative for hyperbolae
    q = a * (1 - e)
    inc = np.degrees(np.arccos(np.clip(h[:, 2] / hn, -1, 1)))
    P = evec / e[:, None]
    Q = np.cross(h / hn[:, None], P)
    # Hyperbolic anomaly now; inbound objects have F < 0.
    coshF = (1 + r / -a) / e
    F = np.arccosh(np.maximum(coshF, 1.0)) * np.sign(np.sum(r0 * v0, axis=1))
    n_mot = np.sqrt(GM_SUN / (-a) ** 3)
    t_peri = -(e * np.sinh(F) - F) / n_mot
    return dict(a=a, e=e, q=q, inc=inc, P=P, Q=Q, n=n_mot, t_peri=t_peri)


def solve_hyperbolic(M, e):
    """Solve Kepler's equation e sinh F - F = M for F (vectorised, safeguarded Newton).

    g(F) = e sinh F - F - M increases monotonically, and for M >= 0 its root lies between
    asinh(M/e) and asinh(M/(e-1)) (since sinh F >= F); M < 0 follows by symmetry. Newton
    steps that leave the bracket are replaced by bisection, so orbits with e close to 1
    (very slow objects) converge too.
    """
    M, e = np.broadcast_arrays(M, e)
    s = np.sign(M)
    Ma = np.abs(M)
    lo = np.arcsinh(Ma / e)
    hi = np.arcsinh(Ma / (e - 1))
    F = np.where(Ma > 6 * e, np.minimum(np.log(2 * Ma / e + 1.8), hi), lo)
    for _ in range(100):
        g = e * np.sinh(F) - F - Ma
        lo = np.where(g < 0, F, lo)
        hi = np.where(g > 0, F, hi)
        Fn = F - g / (e * np.cosh(F) - 1)
        bad = ~((Fn > lo) & (Fn < hi))
        Fn = np.where(bad, 0.5 * (lo + hi), Fn)
        done = np.abs(Fn - F) <= 1e-13 * (1 + Fn)
        F = Fn
        if done.all():
            break
    return s * F


def positions(el, t):
    """Heliocentric positions (n, len(t), 3) in au at times t (days, shape (n, T) or (T,))."""
    t = np.broadcast_to(t, (el["e"].size, np.shape(t)[-1]))
    M = el["n"][:, None] * (t - el["t_peri"][:, None])
    e = el["e"][:, None]
    F = solve_hyperbolic(M, e)
    absa = -el["a"][:, None]
    x = absa * (e - np.cosh(F))
    y = absa * np.sqrt(e**2 - 1) * np.sinh(F)
    return x[..., None] * el["P"][:, None, :] + y[..., None] * el["Q"][:, None, :]


# ---------------------------------------------------------------- Earth and Sun
def earth_position(jd):
    """Heliocentric ecliptic J2000 position of the Earth (au), from the low-precision solar
    coordinates of the Astronomical Almanac (good to about 0.01 deg, 1950-2050)."""
    n = jd - JD2000
    L = np.radians(280.460 + 0.9856474 * n)
    g = np.radians(357.528 + 0.9856003 * n)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    lam -= np.radians(1.3970 * n / 36525.0)        # precession: equinox of date -> J2000
    R = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)
    sun = R[..., None] * np.stack([np.cos(lam), np.sin(lam), np.zeros_like(lam)], -1)
    return -sun


def ecl_to_radec(x):
    """Ecliptic J2000 vectors (..., 3) -> equatorial RA, Dec (radians)."""
    xe = x @ EQ_TO_ECL        # the transpose of EQ_TO_ECL is ECL_TO_EQ, applied on the right
    d = np.linalg.norm(xe, axis=-1)
    return np.arctan2(xe[..., 1], xe[..., 0]), np.arcsin(xe[..., 2] / d)


# ---------------------------------------------------------------- brightness and visibility
def phase_integral(theta):
    """HSP22 Sec. 3: the phase integral p(theta) = (2/3) [(1 - theta/pi) cos theta + sin(theta)/pi]."""
    return (2.0 / 3.0) * ((1 - theta / np.pi) * np.cos(theta) + np.sin(theta) / np.pi)


def apparent_magnitude(H, d_bs, d_bo, d_os):
    """HSP22 Sec. 3: m = H + 2.5 log10( d_BS^2 d_BO^2 / (p(theta) d_OS^4) )."""
    cos_t = (d_bo**2 + d_bs**2 - d_os**2) / (2 * d_bo * d_bs)
    theta = np.arccos(np.clip(cos_t, -1, 1))
    return H + 2.5 * np.log10(d_bs**2 * d_bo**2 / (phase_integral(theta) * d_os**4))


def half_arc(alt_limit, dec):
    """Half-width in hour angle (radians) of the arc on which a body at declination dec
    stands above altitude alt_limit at Rubin; 0 if never, pi if always."""
    c = (np.sin(alt_limit) - np.sin(RUBIN_LAT) * np.sin(dec)) / (np.cos(RUBIN_LAT) * np.cos(dec))
    return np.arccos(np.clip(c, -1, 1)) * (c < 1)


def observable_tonight(ra, dec, ra_sun, dec_sun):
    """True if, at some moment that night, the object is above 30 deg altitude while the
    Sun is below -18 deg (HSP22 Sec. 4, criteria 2 and 3, applied over the whole night).

    In local sidereal time the object is up on an arc centred on its RA and the sky is
    dark on an arc centred on RA_sun + pi; the two arcs overlap if their centres are
    closer than the sum of their half-widths.
    """
    up = half_arc(ALT_MIN, dec)
    dark = np.pi - half_arc(SUN_ALT_MAX, dec_sun)
    sep = np.abs((ra - ra_sun - np.pi + np.pi) % (2 * np.pi) - np.pi)
    return (up > 0) & (sep < up + dark) & (dec < DEC_MAX)


# ---------------------------------------------------------------- the known interstellar objects
# Heliocentric osculating elements from the JPL Small-Body Database (ecliptic J2000;
# retrieved 2026-10-15): perihelion q (au), eccentricity e, inclination i, node Om and
# argument of perihelion w (deg).
KNOWN = {
    "1I/ʻOumuamua": dict(q=0.2559115812959116, e=1.201133796102373, i=122.7417062847286,
                         Om=24.59690955523242, w=241.8105360304898),
    "2I/Borisov": dict(q=2.006520878500843, e=3.356475782676596, i=44.05264247909138,
                       Om=308.1477292269942, w=209.1236864378081),
    "3I/ATLAS": dict(q=1.356481057231181, e=6.141351449317625, i=175.1164570850441,
                     Om=322.1696089290778, w=128.0228697185194),
}


def perifocal_vectors(i, Om, w):
    """Unit vectors P (towards perihelion) and Q in the ecliptic frame, from angles in degrees."""
    i, Om, w = np.radians([i, Om, w])
    P = np.array([np.cos(Om) * np.cos(w) - np.sin(Om) * np.sin(w) * np.cos(i),
                  np.sin(Om) * np.cos(w) + np.cos(Om) * np.sin(w) * np.cos(i),
                  np.sin(w) * np.sin(i)])
    Q = np.array([-np.cos(Om) * np.sin(w) - np.sin(Om) * np.cos(w) * np.cos(i),
                  -np.sin(Om) * np.sin(w) + np.cos(Om) * np.cos(w) * np.cos(i),
                  np.cos(w) * np.sin(i)])
    return P, Q


def asymptote(q, e, i, Om, w, **_):
    """Speed at infinity (km/s) and radiant RA, Dec (deg) of a hyperbolic orbit.

    The incoming asymptote lies at true anomaly -nu_inf, where cos(nu_inf) = -1/e; the
    radiant is the direction from which the object approached, far from the Sun.
    """
    P, Q = perifocal_vectors(i, Om, w)
    nu = np.arccos(-1.0 / e)
    u_ecl = np.cos(nu) * P - np.sin(nu) * Q
    ra, dec = ecl_to_radec(u_ecl)
    v_inf = np.sqrt(GM_SUN * (e - 1) / q) / KMS_TO_AUD
    return v_inf, np.degrees(ra) % 360, np.degrees(dec)


# ---------------------------------------------------------------- the Monte Carlo
H_LIST = (22.4, 23.4, 24.4, 25.4)      # HSP22 Table 2's four absolute magnitudes
LIM_LIST = (24.0, 22.0)                # limiting magnitudes for the sky maps (HSP22 Fig. 4)
Q_CUT = 2.5     # au: objects with perihelion beyond this can never reach m = 24 at H = 22.4
                # (r * Delta > 2.5 * 1.48 > 1.71, the most an m = 24 detection allows)


def simulate_chunk(args):
    """Follow one chunk of objects for 3 years, one sample per night.

    args = (n, seed, pop, solar, focus, b_power). Objects whose perihelion lies beyond
    Q_CUT are not propagated: they cannot get brighter than about m = 25.
    Returns a dict: per-object initial quantities for all n objects, and detailed
    results for the objects that ever reach m <= 24.5 on an observable night.
    """
    n, seed, pop, solar, focus, b_power = args
    ic = initial_conditions(n, seed, pop, solar, focus, b_power)
    el = elements(ic["r0"], ic["v0"])
    allq = dict(v_inf=ic["v_inf"].astype(np.float32), b=ic["b"].astype(np.float32),
                q=el["q"].astype(np.float32), e=el["e"].astype(np.float32),
                inc=el["inc"].astype(np.float32),
                ra_radiant=ic["ra_radiant"].astype(np.float32),
                dec_radiant=ic["dec_radiant"].astype(np.float32),
                m_min=np.full(n, np.inf, np.float32))
    ra_e, dec_e = ecl_to_radec(ic["r0"])
    allq["ra_entry"] = (np.degrees(ra_e) % 360).astype(np.float32)
    allq["dec_entry"] = np.degrees(dec_e).astype(np.float32)
    days = np.arange(0.0, T_SPAN + 1)          # one sample per night
    rows = []
    idx_all = np.flatnonzero(el["q"] < Q_CUT)
    for start in range(0, idx_all.size, 1500):
        idx = idx_all[start:start + 1500]
        sub = {k: v[idx] for k, v in el.items()}
        r = positions(sub, days)                                  # (k, T, 3)
        jd = ic["jd0"][idx, None] + days[None, :]
        earth = earth_position(jd)
        geo = r - earth
        d_bs = np.linalg.norm(r, axis=-1)
        d_bo = np.linalg.norm(geo, axis=-1)
        d_os = np.linalg.norm(earth, axis=-1)
        m = apparent_magnitude(H_OUMUAMUA, d_bs, d_bo, d_os)
        allq["m_min"][idx] = m.min(axis=1)

        ra, dec = ecl_to_radec(geo)
        ra_s, dec_s = ecl_to_radec(-earth)
        obs = observable_tonight(ra, dec, ra_s, dec_s)
        m_obs = np.where(obs, m, np.inf)
        keep = m_obs.min(axis=1) <= T_GRID[-1]
        if not keep.any():
            continue
        k = np.flatnonzero(keep)
        m, m_obs, obs, d_bo, ra, dec = m[k], m_obs[k], obs[k], d_bo[k], ra[k], dec[k]

        # Closest approach to the Earth: the nightly minimum, refined by a parabola.
        j = np.clip(d_bo.argmin(axis=1), 1, days.size - 2)
        rr = np.arange(k.size)
        y0, y1, y2 = d_bo[rr, j - 1], d_bo[rr, j], d_bo[rr, j + 1]
        den = y0 - 2 * y1 + y2
        off = np.where(den > 0, 0.5 * (y0 - y2) / np.where(den > 0, den, 1), 0.0)
        off = np.clip(off, -1, 1)
        t_ca = days[j] + off
        d_min = y1 - 0.25 * (y0 - y2) * off

        def first(cond):
            any_ = cond.any(axis=1)
            return np.where(any_, cond.argmax(axis=1), -1)

        res = dict(id=(idx[k] + 0).astype(np.int32), t_ca=t_ca.astype(np.float32),
                   d_min=d_min.astype(np.float32), m_obs_min=m_obs.min(axis=1).astype(np.float32),
                   jd0=ic["jd0"][idx[k]])
        # Exact detections (with saturation) for HSP22 Table 2 and the sky maps.
        for H in H_LIST:
            dm = H - H_OUMUAMUA
            res[f"det_{H}"] = first(obs & (m + dm <= M_FAINT) & (m + dm > M_BRIGHT)).astype(np.int16)
        for lim in LIM_LIST:
            jdet = first(obs & (m <= lim) & (m > M_BRIGHT))
            ok = jdet >= 0
            jj = np.where(ok, jdet, 0)
            res[f"ra_det_{lim}"] = np.where(ok, np.degrees(ra[rr, jj]) % 360, np.nan).astype(np.float32)
            res[f"dec_det_{lim}"] = np.where(ok, np.degrees(dec[rr, jj]), np.nan).astype(np.float32)
        res["det_19"] = first(obs & (m <= 19.0) & (m > M_BRIGHT)).astype(np.int16)
        # First observable night at or below each threshold, ignoring saturation (web export).
        cm = np.minimum.accumulate(m_obs, axis=1)
        res["first_T"] = np.where(cm.min(axis=1)[:, None] <= T_GRID[None, :],
                                  (cm[:, :, None] > T_GRID[None, None, :]).sum(axis=1), -1).astype(np.int16)
        rows.append(res)
    out = {k: np.concatenate([r_[k] for r_ in rows]) for k in rows[0]} if rows else {}
    return allq, out


RUBIN_LON = np.radians(-70.7494)


def detect_at_instants(args):
    """Detectability judged as HSP22 describe it: at 100 evenly spaced instants in the
    3-year window, using the altitudes of the object and the Sun at that instant.

    args = (n, seed, pop, solar, focus, b_power). Returns, for the objects that reach
    m <= 24 at some instant, a boolean array "detectable at one of the instants", and
    the same objects' verdict from the whole-night test at the same instants.
    """
    n, seed, pop, solar, focus, b_power = args
    ic = initial_conditions(n, seed, pop, solar, focus, b_power)
    el = elements(ic["r0"], ic["v0"])
    idx = np.flatnonzero(el["q"] < Q_CUT)
    sub = {k: v[idx] for k, v in el.items()}
    t = np.linspace(0.0, T_SPAN, 100)
    r = positions(sub, t)
    jd = ic["jd0"][idx, None] + t[None, :]
    earth = earth_position(jd)
    geo = r - earth
    m = apparent_magnitude(H_OUMUAMUA, np.linalg.norm(r, axis=-1), np.linalg.norm(geo, axis=-1),
                           np.linalg.norm(earth, axis=-1))
    ra, dec = ecl_to_radec(geo)
    ra_s, dec_s = ecl_to_radec(-earth)
    lst = np.radians(280.46061837 + 360.98564736629 * (jd - JD2000)) + RUBIN_LON

    def alt(ra_, dec_):
        return np.arcsin(np.sin(RUBIN_LAT) * np.sin(dec_) + np.cos(RUBIN_LAT) * np.cos(dec_) * np.cos(lst - ra_))

    bright = (m <= M_FAINT) & (m > M_BRIGHT)
    instant = bright & (alt(ra, dec) > ALT_MIN) & (dec < DEC_MAX) & (alt(ra_s, dec_s) < SUN_ALT_MAX)
    night = bright & observable_tonight(ra, dec, ra_s, dec_s)
    reach = bright.any(axis=1)
    return instant.any(axis=1)[reach], night.any(axis=1)[reach], n


def run(n_total, n_chunks, pool=None, seed0=2022, pop="G", solar=DEFAULT_SOLAR, focus=True, b_power=2):
    """Run the Monte Carlo in chunks (optionally on a multiprocessing pool) and merge."""
    n_each = n_total // n_chunks
    jobs = [(n_each, seed0 + k, pop, solar, focus, b_power) for k in range(n_chunks)]
    parts = pool.map(simulate_chunk, jobs) if pool is not None else list(map(simulate_chunk, jobs))
    allq = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    det = {}
    for kk, (aq, d) in enumerate(parts):
        if not d:
            continue
        d = dict(d)
        d["id"] = d["id"] + kk * n_each
        for k, v in d.items():
            det.setdefault(k, []).append(v)
    det = {k: np.concatenate(v) for k, v in det.items()}
    return allq, det


def track(obj_id, n_each, seed0=2022, pop="G", solar=DEFAULT_SOLAR, focus=True, b_power=2):
    """Rebuild one object of a run() from its id, and return its nightly track: days,
    heliocentric and Earth positions (au), magnitude, observability, RA and Dec (deg)."""
    k, j = divmod(int(obj_id), n_each)
    ic = initial_conditions(n_each, seed0 + k, pop, solar, focus, b_power)
    el = elements(ic["r0"][j:j + 1], ic["v0"][j:j + 1])
    days = np.arange(0.0, T_SPAN + 1)
    r = positions(el, days)[0]
    earth = earth_position(ic["jd0"][j] + days)
    geo = r - earth
    m = apparent_magnitude(H_OUMUAMUA, np.linalg.norm(r, axis=-1), np.linalg.norm(geo, axis=-1),
                           np.linalg.norm(earth, axis=-1))
    ra, dec = ecl_to_radec(geo)
    ra_s, dec_s = ecl_to_radec(-earth)
    obs = observable_tonight(ra, dec, ra_s, dec_s)
    return dict(days=days, r=r, earth=earth, m=m, obs=obs, ra=np.degrees(ra) % 360, dec=np.degrees(dec),
                jd0=ic["jd0"][j], v_inf=ic["v_inf"][j], b=ic["b"][j], q=el["q"][0], e=el["e"][0])


def rate_factor(v_typ_kms, n0=0.1):
    """HSP22 Sec. 5: objects crossing the 5 au sphere per year, for number density n0 (au^-3)
    and typical speed v (km/s), using the mean chord 4/3 x 5 au."""
    n_inside = n0 * 4.0 / 3.0 * np.pi * R_START**3
    t_cross_yr = (4.0 / 3.0 * R_START) / (v_typ_kms * KMS_TO_AUD * 365.25)
    return n_inside / t_cross_yr
