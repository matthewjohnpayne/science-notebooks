"""Helper module for the notebook "Shift-and-Stack on Detection Catalogues".

It follows Stetzler et al. (2025, AJ 170, 352; S25): instead of shifting and adding
the pixels of many images along trial trajectories, shift and stack the *catalogues*
of low-signal-to-noise detections from those images. Here everything is simulated:
difference images with white noise and a Gaussian PSF, faint moving objects injected
into them, PSF-matched detection, and both the catalogue-based and an image-based
search.

Units: pixels and hours. The PSF is a circular Gaussian of standard deviation `a`
pixels (S25's PSF size a).
"""
import numpy as np
from scipy import ndimage, special, integrate


# ---------------------------------------------------------------- the survey
def make_survey(n_epochs=30, size=200, a=2.0, t_span=4.0, n_obj=0, snr_range=(1.0, 4.0),
                v_range=(2.0, 20.0), seed=1):
    """Simulated difference images: white noise of unit variance per pixel, plus n_obj moving
    point sources on straight trajectories. Each object has a per-epoch PSF-matched
    signal-to-noise ratio nu drawn log-uniformly from snr_range, a speed uniform in v_range
    (pixels/hour) and a random direction. Returns times (hours), the image cube and the truth."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_span, n_epochs)
    nu = np.exp(rng.uniform(*np.log(snr_range), n_obj))
    speed = rng.uniform(*v_range, n_obj)
    ang = rng.uniform(0, 2 * np.pi, n_obj)
    vx, vy = speed * np.cos(ang), speed * np.sin(ang)
    # start positions chosen so that the whole track stays inside the image (with a margin)
    m = 4 * a
    x0 = rng.uniform(m - np.minimum(vx * t_span, 0), size - m - np.maximum(vx * t_span, 0))
    y0 = rng.uniform(m - np.minimum(vy * t_span, 0), size - m - np.maximum(vy * t_span, 0))
    # A point source of flux f gives a PSF-matched S/N of f / (sqrt(4 pi) a) in unit-variance noise
    flux = nu * np.sqrt(4 * np.pi) * a
    yy, xx = np.mgrid[0:size, 0:size]
    imgs = rng.normal(size=(n_epochs, size, size))
    for k in range(n_obj):
        for i, ti in enumerate(t):
            cx, cy = x0[k] + vx[k] * ti, y0[k] + vy[k] * ti
            sl = (slice(max(0, int(cy - 5 * a)), int(cy + 5 * a) + 1), slice(max(0, int(cx - 5 * a)), int(cx + 5 * a) + 1))
            imgs[i][sl] += flux[k] / (2 * np.pi * a * a) * np.exp(-((xx[sl] - cx) ** 2 + (yy[sl] - cy) ** 2) / (2 * a * a))
    truth = dict(x0=x0, y0=y0, vx=vx, vy=vy, nu=nu)
    return t, imgs, truth


def snr_image(img, a):
    """PSF-matched filter: the S/N of a point source centred on each pixel (unit-variance noise)."""
    return ndimage.gaussian_filter(img, a, mode="constant") * np.sqrt(4 * np.pi) * a


def detect(t, imgs, a, nu_th):
    """Peaks of each S/N image above nu_th (local maxima in 3x3), with sub-pixel centroids
    from a parabola through the peak. Returns a catalogue: arrays x, y, t, snr, epoch."""
    rows = []
    for i, img in enumerate(imgs):
        s = snr_image(img, a)
        peak = (s == ndimage.maximum_filter(s, size=3)) & (s >= nu_th)
        peak[:1, :] = peak[-1:, :] = peak[:, :1] = peak[:, -1:] = False
        y, x = np.nonzero(peak)
        def sub(c, l, r):
            den = l - 2 * c + r
            return np.where(den < 0, 0.5 * (l - r) / np.where(den < 0, den, -1), 0.0)
        dx = sub(s[y, x], s[y, x - 1], s[y, x + 1])
        dy = sub(s[y, x], s[y - 1, x], s[y + 1, x])
        rows.append(np.stack([x + dx, y + dy, np.full(x.size, t[i]), s[y, x], np.full(x.size, i)], 1))
    c = np.concatenate(rows)
    return dict(x=c[:, 0], y=c[:, 1], t=c[:, 2], snr=c[:, 3], epoch=c[:, 4].astype(int))


# ---------------------------------------------------------------- noise-peak statistics (S25 Sec. 2.2)
def _B(s, b):
    """Budavari et al. (2017) Appendix: B(s, b) = sqrt(pi/b) exp(s^2/2b) [1 + erf(s/sqrt(2b))]."""
    return np.sqrt(np.pi / b) * np.exp(s * s / (2 * b)) * (1 + special.erf(s / np.sqrt(2 * b)))


def peak_density(z, a):
    """Surface density of noise peaks of height z (in sigma) per unit area and unit z, for white
    noise smoothed with a Gaussian of standard deviation a (a PSF-matched filter). This is
    Budavari, Szalay & Loredo (2017), Appendix, Eq. for n_pk with their shape and scale parameters
    gamma^2 = 1/2 and R^2 = a^2/2 (so s = z):
        n_pk = exp(-z^2) / (4 pi^2 a^2) [ z/2 + (z^2 - 1)/2 B(z, 1) + B(z, 2) ].
    (S25 quote this result in a different form, which does not match our simulations; see the
    notebook.)"""
    return np.exp(-z * z) / (4 * np.pi**2 * a * a) * (z / 2 + (z * z - 1) / 2 * _B(z, 1) + _B(z, 2))


def peak_density_s25(z, a):
    """The noise-peak density exactly as printed in S25 Sec. 2.2, for comparison."""
    s = np.sqrt(2) / 2 * z
    Bp = lambda s_, b: np.pi / b * np.exp(s_ * s_ / (2 * b)) * (1 + special.erf(s_ / np.sqrt(2 * b)))
    return np.exp(-z * z / 2) / (2 * np.pi**2 * a * a) * (np.sqrt(2) / 4 * z + Bp(s, 1) * (0.5 + 0.75 * z) + Bp(s, 2))


def peaks_above(nu_th, a, form=None):
    """lambda(nu_th): the surface density of noise peaks above nu_th (per unit area)."""
    f = form or peak_density
    return integrate.quad(lambda z: f(z, a), nu_th, 12)[0]


def expected_counts(nu, nu_th, dx_a, n_epochs):
    """S25 Sec. 2.2: expected signal and noise detections in one stack bin of side dx (in units of
    a), for an object of per-epoch S/N nu and a catalogue thresholded at nu_th, over n_epochs."""
    p = 0.5 * special.erfc((nu_th - nu) / np.sqrt(2)) * special.erf(dx_a * nu / 4) ** 2
    lam = peaks_above(nu_th, 1.0)               # per a^2
    return n_epochs * p, n_epochs * lam * dx_a**2


# ---------------------------------------------------------------- the catalogue search (S25 Sec. 2.1)
def velocity_grid(v_min, v_max, dv):
    """All trial velocities on a square grid of spacing dv with v_min <= |v| <= v_max."""
    g = np.arange(-v_max, v_max + dv / 2, dv)
    vx, vy = np.meshgrid(g, g)
    s = np.hypot(vx, vy)
    keep = (s >= v_min - dv / 2) & (s <= v_max + dv / 2)
    return vx[keep], vy[keep]


def catalog_search(cat, dx, v_min, v_max, n_min, max_results=100, size=None):
    """S25's algorithm, steps 1-3: project the catalogue to t0 for every trial velocity (spacing
    dx / dt, their Eq. for |V|), histogram the projected positions in bins of dx, and repeatedly
    take the fullest bin of the joint histogram, removing its detections, until the fullest bin
    holds fewer than n_min. Returns the clusters (indices into the catalogue), their trial
    velocities and the number of trial velocities."""
    t0 = cat["t"].min()
    dt = cat["t"].max() - t0
    vxs, vys = velocity_grid(v_min, v_max, dx / dt)
    tt = cat["t"] - t0
    alive = np.ones(cat["x"].size, bool)
    lo = -v_max * dt - dx
    nb = int(np.ceil(((size or (cat["x"].max() + 1)) + 2 * v_max * dt + 2 * dx) / dx))
    def best_for(j):
        idx = np.flatnonzero(alive)
        px = ((cat["x"][idx] - vxs[j] * tt[idx] - lo) // dx).astype(int)
        py = ((cat["y"][idx] - vys[j] * tt[idx] - lo) // dx).astype(int)
        cnt = np.bincount(px * nb + py, minlength=nb * nb)
        k = cnt.argmax()
        return cnt[k], k, idx, px * nb + py
    maxima = np.array([best_for(j)[0] for j in range(vxs.size)])
    results = []
    while len(results) < max_results:
        j = int(maxima.argmax())
        n, k, idx, cells = best_for(j)
        if n != maxima[j]:                 # stale after earlier removals: refresh and retry
            maxima[j] = n
            continue
        if n < n_min:
            break
        members = idx[cells == k]
        results.append(dict(members=members, vx=vxs[j], vy=vys[j], n=int(n)))
        alive[members] = False
        maxima[j] = best_for(j)[0]
        # other velocities whose best bin used these detections are refreshed lazily (above)
    return results, vxs.size


# ---------------------------------------------------------------- robust line fit (S25 Sec. 2.1)
def mcd(X, h=None, n_starts=30, rng=None):
    """Minimum Covariance Determinant estimate of the mean and covariance of the rows of X, by
    the C-steps of FAST-MCD (Rousseeuw & Van Driessen 1999) from random starting subsets.
    h defaults to (n + p + 1) // 2, the most outlier-resistant choice."""
    rng = rng or np.random.default_rng(0)
    n, p = X.shape
    h = h or (n + p + 1) // 2
    best = None
    for _ in range(n_starts):
        sub = rng.choice(n, p + 1, replace=False)
        for _ in range(50):
            mu = X[sub].mean(0)
            C = np.cov(X[sub].T) + 1e-12 * np.eye(p)
            d = np.einsum("ij,jk,ik->i", X - mu, np.linalg.inv(C), X - mu)
            new = np.argsort(d)[:h]
            if set(new) == set(sub):
                break
            sub = new
        det = np.linalg.det(np.cov(X[sub].T) + 1e-12 * np.eye(p))
        if best is None or det < best[0]:
            best = (det, X[sub].mean(0), np.cov(X[sub].T), sub)
    return best[1], best[2], best[3]


def fit_track(cat, members, robust=True):
    """Multivariate regression of (x, y) on t from the (robust) mean and covariance:
    B = S_tt^-1 S_t(xy), c = mu_(xy) - B mu_t (S25's Eqs.). Returns x0, y0 (at t = 0), vx, vy."""
    X = np.stack([cat["t"][members], cat["x"][members], cat["y"][members]], 1)
    if robust and len(members) >= 6:
        mu, C, _ = mcd(X)
    else:
        mu, C = X.mean(0), np.cov(X.T)
    v = C[0, 1:] / C[0, 0]
    x0, y0 = mu[1:] - v * mu[0]
    return x0, y0, v[0], v[1]


def gather(cat, track, dx):
    """Detections within dx of a fitted track (S25 step 5)."""
    x0, y0, vx, vy = track
    d = np.hypot(cat["x"] - (x0 + vx * cat["t"]), cat["y"] - (y0 + vy * cat["t"]))
    return np.flatnonzero(d <= dx)


def matches(track, truth, t, tol):
    """Indices of injected objects whose positions agree with a track to within tol pixels at
    half or more of the epochs (S25's matching rule, with tol in place of 1 arcsec)."""
    x0, y0, vx, vy = track
    px, py = x0 + vx * t[:, None], y0 + vy * t[:, None]
    tx = truth["x0"][None, :] + truth["vx"][None, :] * t[:, None]
    ty = truth["y0"][None, :] + truth["vy"][None, :] * t[:, None]
    ok = np.hypot(px - tx, py - ty) < tol
    return np.flatnonzero(ok.mean(0) >= 0.5)


# ---------------------------------------------------------------- image-based shift-and-stack
def image_search(t, imgs, a, vxs, vys, nu_detect=5.0):
    """Classical shift-and-stack: for each trial velocity, shift each epoch's S/N image back
    along the trajectory (to the nearest pixel) and average, so that a source moving with that
    velocity adds coherently; the stacked S/N is the sum over epochs divided by sqrt(N).
    Returns, for each velocity, the peak stacked S/N and its position."""
    S = np.array([snr_image(img, a) for img in imgs])
    n = len(t)
    best = np.zeros(len(vxs))
    pos = np.zeros((len(vxs), 2), int)
    for j, (vx, vy) in enumerate(zip(vxs, vys)):
        acc = np.zeros(S.shape[1:])
        for i in range(n):
            acc += np.roll(S[i], (-int(round(vy * t[i])), -int(round(vx * t[i]))), axis=(0, 1))
        acc /= np.sqrt(n)
        k = acc.argmax()
        best[j] = acc.flat[k]
        pos[j] = np.unravel_index(k, acc.shape)[::-1]
    return best, pos


def image_search_peaks(t, imgs, a, v_min, v_max, dv, threshold):
    """Blind image-based shift-and-stack: for every trial velocity on a grid of spacing dv
    (pixels/hour), shift each epoch's S/N image back along the trajectory (nearest pixel), sum,
    divide by sqrt(N), and keep every local maximum of the stack above threshold. Returns a list
    of candidates (x0, y0, vx, vy, stacked S/N), the number of trial velocities and the run time."""
    import time
    S = np.array([snr_image(img, a) for img in imgs])
    vxs, vys = velocity_grid(v_min, v_max, dv)
    n = len(t)
    t_start = time.time()
    cands = []
    for vx, vy in zip(vxs, vys):
        acc = np.zeros(S.shape[1:])
        for i in range(n):
            acc += np.roll(S[i], (-int(round(vy * t[i])), -int(round(vx * t[i]))), axis=(0, 1))
        acc /= np.sqrt(n)
        pk = (acc >= threshold) & (acc == ndimage.maximum_filter(acc, size=5))
        for y, x in zip(*np.nonzero(pk)):
            cands.append((x, y, vx, vy, acc[y, x]))
    return cands, vxs.size, time.time() - t_start


def merge_candidates(cands, radius, dv_tol):
    """Keep the strongest candidate among those within radius pixels and dv_tol (pixels/hour) of
    each other: neighbouring trial velocities see the same object."""
    order = sorted(cands, key=lambda c: -c[4])
    kept = []
    for c in order:
        if all(np.hypot(c[0] - k[0], c[1] - k[1]) > radius or np.hypot(c[2] - k[2], c[3] - k[3]) > dv_tol for k in kept):
            kept.append(c)
    return kept
