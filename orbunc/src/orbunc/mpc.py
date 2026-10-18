"""Observations, observatory positions and orbits from the Minor Planet Center's public APIs.

API documentation: https://docs.minorplanetcenter.net/mpc-ops-docs/apis/
Responses are cached as JSON (in `cache_dir`, default data_dir()/mpc) so a notebook can be re-run offline.
"""
import gzip
import json
import pathlib
from dataclasses import dataclass, field

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from .ephem import body_state, data_dir
from .frames import ARCSEC, AU_KM, R_EARTH_KM, EC_TO_EQ, mjd_tt_to_tdb, rotate_states, utc_jd_to_tdb

API = "https://data.minorplanetcenter.net/api"


def _cached(name, fetch, cache_dir=None):
    d = pathlib.Path(cache_dir) if cache_dir else data_dir() / "mpc"
    d.mkdir(parents=True, exist_ok=True)
    path = d / (name.replace(" ", "_").replace("/", "_") + ".json.gz")
    if path.exists():
        with gzip.open(path, "rt") as f:
            return json.load(f)
    data = fetch()
    with gzip.open(path, "wt") as f:
        json.dump(data, f)
    return data


def _get(endpoint, payload):
    r = requests.get(f"{API}/{endpoint}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def fetch_ades(desig, cache_dir=None):
    """All optical observations of an object, as a list of ADES records (dicts)."""
    return _cached(f"obs_{desig}", lambda: _get("get-obs", {"desigs": [desig], "output_format": ["ADES_DF"]})[0]["ADES_DF"], cache_dir)


def fetch_orbit(desig, cache_dir=None):
    """The MPC's current orbit for an object, in mpc_orb JSON form."""
    def fetch():
        o = _get("get-orb", {"desig": desig})
        o = o[0] if isinstance(o, list) else o
        return o["mpc_orb"][0]
    return _cached(f"orb_{desig}", fetch, cache_dir)


def fetch_obscode(code, cache_dir=None):
    return _cached(f"obscode_{code}", lambda: _get("obscodes", {"obscode": code}), cache_dir)


# --- observatories -------------------------------------------------------

def observer_states(codes, jd_utc, cache_dir=None):
    """Barycentric ICRF position and velocity (N, 6) of ground stations at UTC Julian dates.

    The Earth's centre comes from the ASSIST (DE440) ephemeris; the station's offset from it
    from its MPC parallax constants, rotated into the GCRS by astropy (precession, nutation,
    Earth rotation and polar motion).
    """
    codes = np.asarray(codes)
    jd_utc = np.asarray(jd_utc, float)
    out = np.empty((len(codes), 6))
    for code in np.unique(codes):
        k = np.where(codes == code)[0]
        oc = fetch_obscode(str(code), cache_dir)
        if oc.get("rhocosphi") in (None, "") or oc.get("longitude") in (None, ""):
            raise ValueError(f"station {code} has no parallax constants (space-based or roving?)")
        lon = np.radians(float(oc["longitude"]))
        rc, rs = float(oc["rhocosphi"]), float(oc["rhosinphi"])
        loc = EarthLocation.from_geocentric(rc * np.cos(lon) * R_EARTH_KM, rc * np.sin(lon) * R_EARTH_KM,
                                            rs * R_EARTH_KM, unit=u.km)
        pos, vel = loc.get_gcrs_posvel(Time(jd_utc[k], format="jd", scale="utc"))
        out[k, :3] = pos.xyz.to_value(u.km).T / AU_KM
        out[k, 3:] = vel.xyz.to_value(u.km / u.day).T / AU_KM
    t = utc_jd_to_tdb(jd_utc)
    return out + np.array([body_state("earth", tk)[0] for tk in np.atleast_1d(t)])


# --- observations --------------------------------------------------------

@dataclass
class Observations:
    """Optical astrometry as parallel arrays (angles in radians, uncertainties in arcsec)."""
    jd_utc: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    sig_ra: np.ndarray      # uncertainty of RA*cos(Dec), arcsec
    sig_dec: np.ndarray     # arcsec
    stn: np.ndarray
    trk: np.ndarray         # tracklet identifier
    mag: np.ndarray
    band: np.ndarray
    obsid: np.ndarray = None   # MPC observation identifiers
    t: np.ndarray = field(init=False)          # TDB days since J2000
    observer: np.ndarray = field(init=False)   # (N, 6) barycentric station states

    def __post_init__(self):
        self.t = utc_jd_to_tdb(self.jd_utc) if len(self.jd_utc) else np.empty(0)
        self.observer = observer_states(self.stn, self.jd_utc) if len(self.jd_utc) else np.empty((0, 6))

    def __len__(self):
        return len(self.jd_utc)

    def select(self, index):
        """A subset (boolean mask or index array), without recomputing station positions."""
        new = object.__new__(Observations)
        for name in self.__dataclass_fields__:
            setattr(new, name, getattr(self, name)[index])
        return new

    @staticmethod
    def concat(parts):
        new = object.__new__(Observations)
        for name in Observations.__dataclass_fields__:
            setattr(new, name, np.concatenate([getattr(p, name) for p in parts]))
        return new

    def tracklets(self):
        """Index arrays, one per tracklet identifier, in time order."""
        groups = [np.where(self.trk == k)[0] for k in dict.fromkeys(self.trk)]
        return sorted(groups, key=lambda g: self.t[g].min())

    def with_floor(self, sigma_floor):
        """A copy whose uncertainties are at least sigma_floor arcsec in each coordinate.

        Reported uncertainties often omit errors shared by a whole tracklet (star-catalogue
        and timing errors), so the smallest of them are not to be trusted.
        """
        new = self.select(np.arange(len(self)))
        new.sig_ra = np.maximum(self.sig_ra, sigma_floor)
        new.sig_dec = np.maximum(self.sig_dec, sigma_floor)
        return new

    def deweight_nights(self, n0=4):
        """A copy in which a station's batch of more than n0 positions in one night is down-weighted.

        Uncertainties are scaled by sqrt(N/n0) for a batch of N > n0, as proposed by
        Vereš et al. (2017), so that many positions from one night, which share systematic
        errors, do not count as many independent measurements. A night runs from local noon to
        local noon at the station.
        """
        lon = np.array([float(fetch_obscode(str(s)).get("longitude") or 0) for s in self.stn])
        night = np.floor(self.jd_utc + lon / 360.0).astype(int)
        key = np.char.add(self.stn.astype(str), night.astype(str))
        _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        factor = np.sqrt(np.maximum(counts[inv] / n0, 1.0))
        new = self.select(np.arange(len(self)))
        new.sig_ra = self.sig_ra * factor
        new.sig_dec = self.sig_dec * factor
        return new

    def before(self, jd_utc, submitted=None):
        """The observations available before a UTC Julian date.

        By default this means observations *made* before that date. Observations often reach the
        MPC days or weeks after they are made, so to reconstruct what was known at a given date,
        pass `submitted`, an array of the UTC Julian dates at which each observation was submitted
        (see submission_times); observations are then selected by submission time.
        """
        when = self.jd_utc if submitted is None else submitted
        return self.select(np.where(when < jd_utc)[0])

    @property
    def weights(self):
        """Inverse variances (N, 2) in rad^-2 for (RA*cos Dec, Dec)."""
        return np.stack([1 / (self.sig_ra * ARCSEC) ** 2, 1 / (self.sig_dec * ARCSEC) ** 2], axis=-1)


def from_ades(records, default_sigma=1.0, start=None, end=None):
    """Build Observations from ADES records; missing uncertainties get `default_sigma` arcsec.

    Space-based stations and records without a position are dropped; so are records outside
    [start, end] if given (ISO date strings, compared with the ADES obsTime).
    """
    keep = []
    for o in records:
        if (start and o.get("obstime", "") < start) or (end and o.get("obstime", "") > end):
            continue
        if o.get("ra") in (None, "") or o.get("obstime") in (None, ""):
            continue
        if o.get("sys") not in (None, "") or o.get("pos1") not in (None, ""):
            continue  # satellite or roving observer
        keep.append(o)
    keep.sort(key=lambda o: o["obstime"])
    f = lambda k, d: np.array([float(o[k]) if o.get(k) not in (None, "") else d for o in keep])
    times = Time([o["obstime"].rstrip("Z") for o in keep], format="isot", scale="utc").jd
    return Observations(
        jd_utc=times,
        ra=np.radians(f("ra", np.nan)), dec=np.radians(f("dec", np.nan)),
        sig_ra=f("rmsra", default_sigma), sig_dec=f("rmsdec", default_sigma),
        stn=np.array([o["stn"] for o in keep]),
        trk=np.array([o.get("trkid") or o.get("trksub") or f"{o['stn']}{o['obstime'][:10]}" for o in keep]),
        mag=f("mag", np.nan), band=np.array([o.get("band") or "" for o in keep]),
        obsid=np.array([o.get("obsid") or "" for o in keep]),
    )


def submission_times(obs, path):
    """UTC Julian dates at which each observation was submitted to the MPC (NaN if unknown).

    `path` is a CSV with columns obsid, submission_block_id, as exported from the MPC's
    observation database (submission_block_id begins with the ISO time of the submission).
    """
    import csv
    with open(path) as f:
        when = {r["obsid"]: r["submission_block_id"].split("_")[0] for r in csv.DictReader(f)}
    iso = [when.get(k) for k in obs.obsid]
    out = np.full(len(obs), np.nan)
    known = [i for i, v in enumerate(iso) if v]
    if known:
        out[known] = Time([iso[i] for i in known], format="isot", scale="utc").jd
    return out


# --- orbits --------------------------------------------------------------

def orbit_state(mpc_orb):
    """(t_epoch, state, cov): the MPC Cartesian orbit as a barycentric ICRF state and 6x6 covariance.

    mpc_orb 'CAR' elements are heliocentric, J2000 ecliptic, at an epoch in TT.
    """
    car = mpc_orb["CAR"]
    x = np.array(car["coefficient_values"][:6], float)
    cov = np.zeros((6, 6))
    for i in range(6):
        for j in range(i, 6):
            cov[i, j] = cov[j, i] = car["covariance"][f"cov{i}{j}"]
    t = float(mjd_tt_to_tdb(mpc_orb["epoch_data"]["epoch"]))
    R = np.zeros((6, 6))
    R[:3, :3] = R[3:, 3:] = EC_TO_EQ
    state = rotate_states(x, EC_TO_EQ) + body_state("sun", t)[0]
    return t, state, R @ cov @ R.T
