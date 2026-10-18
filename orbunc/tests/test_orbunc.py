"""Tests of the orbunc toolkit.

Tests that propagate orbits need the ASSIST ephemeris files (see orbunc.ephem); they are
skipped if the files are not present. MPC data are read from the NEOCP notebook's cache.
"""
import copy
import pathlib

import numpy as np
import pytest

from orbunc import ephem, fit, frames, mpc, predict, ranging

CACHE = pathlib.Path(__file__).resolve().parents[2] / "notebooks" / "neocp_uncertainty" / "data"


def have_ephem():
    try:
        ephem.get_ephem(download=False)
        return True
    except FileNotFoundError:
        return False


needs_ephem = pytest.mark.skipif(not have_ephem(), reason="ASSIST ephemeris files not available")


# --- pure geometry ---------------------------------------------------------

def test_radec_round_trip():
    rng = np.random.default_rng(0)
    ra, dec = rng.uniform(0, 2 * np.pi, 100), rng.uniform(-1.5, 1.5, 100)
    ra2, dec2 = frames.unit_to_radec(frames.radec_to_unit(ra, dec))
    assert np.allclose(ra, ra2) and np.allclose(dec, dec2)


def test_gnomonic_small_offsets():
    xi, eta = frames.gnomonic(1.0 + 1e-6 / np.cos(0.3), 0.3 + 2e-6, 1.0, 0.3)
    assert np.isclose(xi, 1e-6, rtol=1e-4) and np.isclose(eta, 2e-6, rtol=1e-4)


def test_elements_circular_and_eccentric():
    # circular orbit at 1 au in the ecliptic
    v = np.sqrt(frames.GM_SUN)
    el = frames.elements(np.array([[1.0, 0, 0, 0, v, 0]]))
    assert np.isclose(el["a"][0], 1) and el["e"][0] < 1e-12 and el["i"][0] < 1e-9
    # at perihelion of an a = 2, e = 0.5 orbit inclined by 30 degrees
    a, e, inc = 2.0, 0.5, np.radians(30)
    q = a * (1 - e)
    vp = np.sqrt(frames.GM_SUN * (1 + e) / q)
    el = frames.elements(np.array([[q, 0, 0, 0, vp * np.cos(inc), vp * np.sin(inc)]]))
    assert np.isclose(el["a"][0], a) and np.isclose(el["e"][0], e) and np.isclose(el["i"][0], 30)
    assert np.isclose(el["q"][0], q) and np.isclose(el["M"][0] % 360, 0, atol=1e-6)


def test_ecliptic_rotation_is_orthonormal():
    assert np.allclose(frames.EQ_TO_EC @ frames.EC_TO_EQ, np.eye(3))


# --- MPC data, propagation and fitting ---------------------------------------

@pytest.fixture(scope="module")
def yr4():
    if not have_ephem():
        pytest.skip("ASSIST ephemeris files not available")
    obs = mpc.from_ades(mpc.fetch_ades("2024 YR4", CACHE))
    t0, x0, cov = mpc.orbit_state(mpc.fetch_orbit("2024 YR4", CACHE))
    return obs, t0, x0, cov


@needs_ephem
def test_mpc_orbit_reproduces_observations(yr4):
    """End-to-end check of time scales, frames, light time and observatory positions."""
    obs, t0, x0, _ = yr4
    res = predict.residuals(x0[None], t0, obs)[0] / frames.ARCSEC
    assert np.median(np.abs(res)) < 0.15
    assert np.all(np.abs(res.mean(axis=0)) < 0.05)


@needs_ephem
def test_propagation_forward_backward(yr4):
    _, t0, x0, _ = yr4
    x1 = ephem.propagate(x0[None], t0, [t0 + 30])[0, 0]
    x2 = ephem.propagate(x1[None], t0 + 30, [t0])[0, 0]
    assert np.linalg.norm(x2[:3] - x0[:3]) < 1e-11


@needs_ephem
def test_strikes_earth():
    t = 9000.0
    e = ephem.body_state("earth", t)[0]
    towards = np.r_[e[:3] + [0.001, 0, 0], e[3:] + [-0.01, 0, 0]]   # 0.001 au away, closing at 17 km/s
    away = np.r_[e[:3] + [0.001, 0, 0], e[3:] + [0.01, 0, 0]]
    hit = ephem.strikes_earth(np.array([towards, away]), t, np.array([t, t + 1]))
    assert hit.tolist() == [True, False]
    out = ephem.propagate(np.array([towards, away]), t, [t + 1])
    assert np.all(np.isnan(out[0, 0])) and np.all(np.isfinite(out[0, 1]))


@needs_ephem
def test_attributable_round_trip(yr4):
    """(rho, rho-dot) -> state -> (rho, rho-dot), and the attributable of the true orbit."""
    obs, t0, x0, _ = yr4
    attr = ranging.fit_attributable(obs.select(np.where(obs.trk == "00000IVTma")[0]))
    rho, rhodot = ranging.rho_rhodot_of(x0, t0, attr)
    X = ranging.states_from_attributable(attr.x, rho, rhodot, attr)
    rho2, rhodot2 = ranging.rho_rhodot_of(X[0], attr.t, attr)
    assert np.isclose(rho, rho2, rtol=1e-6) and np.isclose(rhodot, rhodot2, rtol=1e-4, atol=1e-8)
    # the bound interval of rho-dot at the true distance contains the true rho-dot
    lo, hi = ranging.rhodot_interval(attr, rho)
    assert lo[0] < rhodot < hi[0]
    # the end points of the interval are orbits of zero heliocentric energy
    Xe = ranging.states_from_attributable(np.tile(attr.x, (2, 1)), [rho, rho], [lo[0], hi[0]], attr)
    assert np.all(np.abs(ranging.helio_energy(Xe, attr.t)) < 1e-6)


@needs_ephem
def test_least_squares_fit_recovers_synthetic_truth(yr4):
    """Fit noisy synthetic observations; the truth should lie within the fitted uncertainty."""
    obs, t0, x0, _ = yr4
    keep = np.isin(obs.trk, ["00000IVTma", "00000IVnqT", "00000IW3Ek", "00000IWrpA"])
    sub = obs.select(np.where(keep)[0]).with_floor(0.2)
    xt = ephem.propagate(x0[None], t0, [sub.t[0]])[0, 0]
    ra, dec, _ = predict.predict(xt[None], sub.t[0], sub.t, sub.observer)
    rng = np.random.default_rng(1)
    m2 = []
    for _ in range(4):
        syn = copy.copy(sub)
        syn.ra = ra[0] + rng.normal(0, 1, len(sub)) * sub.sig_ra * frames.ARCSEC / np.cos(dec[0])
        syn.dec = dec[0] + rng.normal(0, 1, len(sub)) * sub.sig_dec * frames.ARCSEC
        f = fit.fit_orbit(xt + np.r_[1e-6, 0, 0, 0, 0, 0], sub.t[0], syn)
        d = f.state - xt
        m2.append(d @ np.linalg.solve(f.cov, d))
        assert f.chi2 < 3 * 2 * len(sub)
    assert np.mean(m2) < 20   # chi-squared with 6 degrees of freedom has mean 6


@needs_ephem
def test_ranging_brackets_truth(yr4):
    obs, t0, x0, _ = yr4
    obs = obs.with_floor(0.2)
    groups = [np.where(obs.trk == k)[0] for k in ["00000IVTma", "00000IVnqT"]]
    attr = ranging.fit_attributable(obs.select(groups[0]))
    rho, _ = ranging.rho_rhodot_of(x0, t0, attr)
    nodes = ranging.systematic_ranging(attr, obs.select(np.concatenate(groups)), n_s=30, n_rd=12)
    w = ranging.weights(nodes, attr)
    q = np.interp([0.001, 0.999], np.cumsum(w[np.argsort(nodes.rho)]), np.sort(nodes.rho))
    assert q[0] < rho < q[1]


def test_with_floor():
    o = object.__new__(mpc.Observations)
    for name in mpc.Observations.__dataclass_fields__:
        setattr(o, name, np.zeros(3))
    o.sig_ra = np.array([0.05, 0.3, 1.0])
    o.sig_dec = np.array([0.1, 0.1, 2.0])
    f = o.with_floor(0.2)
    assert f.sig_ra.tolist() == [0.2, 0.3, 1.0] and f.sig_dec.tolist() == [0.2, 0.2, 2.0]
    assert o.sig_ra[0] == 0.05   # the original is unchanged


def test_submission_time_from_obsid():
    """Decoded submission times match the MPC's submission_block_id for the same observations."""
    from astropy.time import Time
    for obsid, iso in [("Llv7bTFW0000GVfI010000001", "2024-12-27T08:07:11.962"),
                       ("Lmt722000000Gg2M010000001", "2025-02-25T07:30:34.000"),
                       ("LuBIjL0J0000Bwvd010000001", "2026-05-12T20:00:03.019")]:
        assert abs(mpc.submission_time_from_obsid(obsid) - Time(iso, scale="utc").jd) < 1e-8
    assert np.isnan(mpc.submission_time_from_obsid("00000000_00_000000a"))
