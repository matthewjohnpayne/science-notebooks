"""Space-based observers: trajectories from JPL Horizons.

Positions are geocentric ICRF vectors in km (the convention of ADES space-based observations), so
they can be used directly as `Observations.geo_km`.
"""
import gzip
import json
import pathlib

import numpy as np
import requests

from .ephem import data_dir

HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"


def fetch_vectors(target, start, stop, step="6 h", cache_dir=None):
    """Geocentric ICRF state vectors of a Horizons target: (jd_tdb (M,), pos_km (M, 3), vel_km_s (M, 3)).

    `target` is a Horizons COMMAND (for example "-211", the Nancy Grace Roman Space Telescope);
    start/stop are ISO dates. The response is cached as gzip JSON in cache_dir (default
    data_dir()/horizons).
    """
    d = pathlib.Path(cache_dir) if cache_dir else data_dir() / "horizons"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{target}_{start}_{stop}_{step.replace(' ', '')}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as f:
            rows = json.load(f)
    else:
        r = requests.get(HORIZONS, timeout=120, params=dict(
            format="json", COMMAND=f"'{target}'", EPHEM_TYPE="VECTORS", CENTER="'500@399'",
            START_TIME=f"'{start}'", STOP_TIME=f"'{stop}'", STEP_SIZE=f"'{step}'", REF_PLANE="FRAME",
            REF_SYSTEM="ICRF", OUT_UNITS="KM-S", VEC_TABLE="2", CSV_FORMAT="YES", TIME_TYPE="TDB"))
        r.raise_for_status()
        res = r.json()["result"]
        body = res[res.index("$$SOE") + 5:res.index("$$EOE")]
        rows = [[float(v) for i, v in enumerate(line.split(",")[:8]) if i != 1]
                for line in body.strip().splitlines()]
        with gzip.open(path, "wt") as f:
            json.dump(rows, f)
    a = np.array(rows)
    return a[:, 0], a[:, 1:4], a[:, 4:7]


class Trajectory:
    """Cubic Hermite interpolation of a spacecraft's geocentric position (km) in TDB Julian date."""

    def __init__(self, jd_tdb, pos_km, vel_km_s):
        self.t, self.p, self.v = jd_tdb, pos_km, vel_km_s * 86400.0   # km/day

    def __call__(self, jd_tdb):
        jd = np.atleast_1d(jd_tdb)
        if jd.min() < self.t[0] or jd.max() > self.t[-1]:
            raise ValueError("time outside the spacecraft trajectory")
        k = np.clip(np.searchsorted(self.t, jd) - 1, 0, len(self.t) - 2)
        h = (self.t[k + 1] - self.t[k])[:, None]
        s = ((jd - self.t[k])[:, None]) / h
        h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
        return h00 * self.p[k] + h10 * h * self.v[k] + h01 * self.p[k + 1] + h11 * h * self.v[k + 1]


def roman(start="2026-12-01", stop="2028-10-01", step="6 h", cache_dir=None):
    """The Nancy Grace Roman Space Telescope (Horizons -211). Its post-launch trajectory is a
    prediction until it reaches its orbit around the Sun-Earth L2 point, and will be revised."""
    return Trajectory(*fetch_vectors("-211", start, stop, step, cache_dir))
