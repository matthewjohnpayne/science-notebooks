"""Full six-parameter least-squares orbit fit (differential corrections) and sampling."""
from dataclasses import dataclass

import numpy as np

from .predict import residuals


@dataclass
class OrbitFit:
    t: float               # epoch (TDB)
    state: np.ndarray      # barycentric ICRF (6,)
    cov: np.ndarray        # 6x6
    chi2: float
    nobs: int
    res: np.ndarray        # (M, 2) residuals, radians
    used: np.ndarray = None  # (M,) False for observations rejected as outliers


def fit_orbit(state, t, obs, n_iter=20, h=(1e-8, 1e-10), tol=1e-6):
    """Levenberg-Marquardt fit of a Cartesian state at epoch t to the astrometry `obs`.

    Partial derivatives are central finite differences (steps h: au, au/day), computed as
    13 test particles in one ASSIST run per iteration.
    """
    x = np.asarray(state, float).copy()
    step = np.array([h[0]] * 3 + [h[1]] * 3)
    W = obs.weights.ravel()
    lam, chi_best = 1e-3, np.inf
    for _ in range(n_iter):
        X = np.vstack([x] + [x + step[k] * np.eye(6)[k] for k in range(6)] + [x - step[k] * np.eye(6)[k] for k in range(6)])
        res = residuals(X, t, obs).reshape(13, -1)
        chi = np.sum(res[0] ** 2 * W)
        if chi < chi_best:
            converged = chi_best - chi < tol * chi
            x_best, chi_best, r_best = x.copy(), chi, res[0]
            J = -(res[1:7] - res[7:13]).T / (2 * step)
            A = J.T @ (J * W[:, None])
            g = J.T @ (W * r_best)
            lam /= 10
            if converged:
                break
        else:
            lam *= 10
        x = x_best + np.linalg.solve(A + lam * np.diag(np.diag(A)), g)
    cov = np.linalg.inv(A)
    return OrbitFit(t, x_best, cov, chi_best, len(obs), r_best.reshape(-1, 2))


def fit_orbit_robust(state, t, obs, chi2_reject=8.0, chi2_recover=7.0, max_rounds=6, **kw):
    """fit_orbit with automatic outlier rejection (Carpino, Milani & Chesley 2003).

    An observation whose chi-squared (of its two coordinates) exceeds chi2_reject is dropped,
    and a dropped one is restored when its chi-squared falls below chi2_recover; repeat until
    the set of observations no longer changes.
    """
    used = np.ones(len(obs), bool)
    x = np.asarray(state, float)
    for _ in range(max_rounds):
        f = fit_orbit(x, t, obs.select(np.where(used)[0]), **kw)
        x = f.state
        chi_i = np.sum(residuals(x[None], t, obs)[0] ** 2 * obs.weights, axis=1)
        new = np.where(used, chi_i <= chi2_reject, chi_i < chi2_recover)
        if np.array_equal(new, used):
            break
        used = new
    f.used = used
    return f


def sample_gaussian(fit, n, rng):
    """n orbits drawn from the linearised (Gaussian) uncertainty of a fit."""
    return rng.multivariate_normal(fit.state, fit.cov, size=n)
