"""Helpers for the occultation notebook: fits with and without occultations, uncertainty frames.

The orbit machinery is in the `orbunc` toolkit at the top of this repository.
"""
import numpy as np

from orbunc import ephem, fit, frames, mpc, predict, simulate

KM = frames.AU_KM
STAGE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]   # categorical slots 1, 2, 3, 7
INK, MUTED = "#0b0b0b", "#8a8984"


def load(desig, cache_dir="data"):
    """All observations, occultations included, their submission times, and the MPC orbit."""
    obs = mpc.from_ades(mpc.fetch_ades(desig, cache_dir), occultations=True)
    t0, x0, cov = mpc.orbit_state(mpc.fetch_orbit(desig, cache_dir))
    return obs, mpc.submission_times(obs), (t0, x0, cov)


def stages(obs, submitted, truth, sigma_floor=0.1, t_fit=None):
    """Fits with the optical observations submitted before the first occultation, plus 0, 1, 2, ...
    of the occultations in time order. The optical data are the same at every stage, so the
    differences are due to the occultations alone."""
    occ = np.where(obs.kind == "occultation")[0]
    occ = occ[np.argsort(obs.t[occ])]
    first = np.nanmin(submitted[occ])
    optical = np.where((obs.kind == "optical") & (submitted < first))[0]
    base = obs.with_floor(sigma_floor).deweight_nights()
    t0, x0, _ = truth
    t_fit = obs.t[occ[0]] if t_fit is None else t_fit
    x_start = ephem.propagate(x0[None], t0, [t_fit])[0, 0]
    out = []
    for k in range(len(occ) + 1):
        O = base.select(np.concatenate([optical, occ[:k]]))
        f = fit.fit_orbit_robust(x_start, t_fit, O)
        out.append(dict(n_occ=k, obs=O, fit=f))
    return out, occ


def propagate_cov(state, cov, t0, times, h=(1e-8, 1e-10)):
    """States (M, 6) and 6x6 covariances (M, 6, 6) at `times`, by linear mapping (finite differences)."""
    step = np.array([h[0]] * 3 + [h[1]] * 3)
    X = np.vstack([state] + [state + step[k] * np.eye(6)[k] for k in range(6)]
                  + [state - step[k] * np.eye(6)[k] for k in range(6)])
    P = ephem.propagate(X, t0, times)                       # (M, 13, 6)
    J = (P[:, 1:7] - P[:, 7:13]) / (2 * step)[None, :, None]  # (M, 6 in, 6 out)
    J = np.transpose(J, (0, 2, 1))
    return P[:, 0], J @ cov @ np.transpose(J, (0, 2, 1))


def view_frame(state, t, observer=None):
    """Unit vectors (along-track on the sky, across-track on the sky, line of sight) for an
    asteroid at barycentric `state` seen from `observer` (default: the Earth's centre)."""
    obs = ephem.body_state("earth", t)[0] if observer is None else observer
    los = state[:3] - obs[:3]
    los /= np.linalg.norm(los)
    v = state[3:] - obs[3:]
    along = v - (v @ los) * los
    along /= np.linalg.norm(along)
    return np.array([along, np.cross(los, along), los])


def sigmas_km(states, covs, times):
    """1-sigma position uncertainty (km) along-track, across-track (both on the sky, as seen from
    the Earth's centre) and along the line of sight, at each time."""
    out = np.empty((len(times), 3))
    for k, t in enumerate(times):
        F = view_frame(states[k], t)
        out[k] = np.sqrt(np.einsum("ij,jk,ik->i", F, covs[k][:3, :3], F)) * KM
    return out


def occultation_error_km(obs, k):
    """The 1-sigma error ellipse of occultation k as (semi-axes in km, position angle of the major
    axis in degrees east of north), at the asteroid's distance from the observer."""
    sa, sd, r = obs.sig_ra[k], obs.sig_dec[k], obs.corr[k]
    C = np.array([[sa * sa, r * sa * sd], [r * sa * sd, sd * sd]])   # arcsec^2, (east, north)
    vals, vecs = np.linalg.eigh(C)
    return np.sqrt(vals), np.degrees(np.arctan2(vecs[0, 1], vecs[1, 1])) % 180


def sky_cov_mas(state, cov, t0, obs):
    """Observed minus predicted (RA cos Dec, Dec) in mas at each observation, and the 2x2 covariance (mas^2)
    of the prediction."""
    step = np.array([1e-8] * 3 + [1e-10] * 3)
    X = np.vstack([state] + [state + step[k] * np.eye(6)[k] for k in range(6)]
                  + [state - step[k] * np.eye(6)[k] for k in range(6)])
    R = predict.residuals(X, t0, obs) / frames.ARCSEC * 1e3          # (13, M, 2)
    J = -(R[1:7] - R[7:13]) / (2 * step)[:, None, None]               # (6, M, 2)
    C = np.einsum("imk,ij,jml->mkl", J, cov, J)
    return R[0], C


def occ_cov_mas(obs, k):
    """2x2 covariance (mas^2, east/north) of observation k from its uncertainties and correlation."""
    sa, sd, r = obs.sig_ra[k] * 1e3, obs.sig_dec[k] * 1e3, obs.corr[k]
    return np.array([[sa * sa, r * sa * sd], [r * sa * sd, sd * sd]])


def inflation_factor(misses, prior_covs, occ_covs):
    """The factor k by which a prior's sky-plane covariance must be scaled so that the misses at real
    occultations are statistically consistent: median of chi2 = d^T (k^2 C_prior + C_occ)^-1 d equal to
    that of a chi-squared with two degrees of freedom (2 ln 2). The median resists odd events."""
    def med(k):
        return np.median([d @ np.linalg.solve(k * k * Cp + Co, d) for d, Cp, Co in zip(misses, prior_covs, occ_covs)])
    lo, hi = 0.1, 100.0
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if med(mid) > 2 * np.log(2) else (lo, mid)
    return np.sqrt(lo * hi)


def history_fit(x_true, t_true, years, stn, sigma_arcsec, rng):
    """A least-squares orbit from a simulated observing history (see orbunc.simulate.survey_history)."""
    O = simulate.survey_history(x_true, t_true, list(years), stn, sigma_arcsec, rng)
    x_start = ephem.propagate(x_true[None], t_true, [O.t[0]])[0, 0]
    return O, fit.fit_orbit(x_start, O.t[0], O)


def position_cloud(state, cov, t0, t, n, rng):
    """n samples of the position offset (km) at time t in the frame (along-track, across-track,
    line of sight) seen from the Earth's centre, from a Gaussian orbit estimate (state, cov at t0)."""
    X, C = propagate_cov(state, cov, t0, [t])
    F = view_frame(X[0], t)
    P = F @ C[0][:3, :3] @ F.T * KM ** 2
    return rng.multivariate_normal(np.zeros(3), P, size=n)


def sigmas_orbit_km(states, covs, times):
    """1-sigma position uncertainty (km) along the orbit, radially (from the Sun) and out of the
    orbital plane, at each time: a frame that, unlike the view from the Earth, changes smoothly."""
    out = np.empty((len(times), 3))
    sun = ephem.body_state("sun", times)
    for k in range(len(times)):
        r, v = states[k, :3] - sun[k, :3], states[k, 3:] - sun[k, 3:]
        along = v / np.linalg.norm(v)
        normal = np.cross(r, v)
        normal /= np.linalg.norm(normal)
        F = np.array([along, np.cross(normal, along), normal])
        out[k] = np.sqrt(np.einsum("ij,jk,ik->i", F, covs[k][:3, :3], F)) * KM
    return out
