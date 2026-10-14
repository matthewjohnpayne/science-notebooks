#!/usr/bin/env python3
"""Extract the MPCORB orbits used by notebooks/heliolinc/heliolinc.ipynb.

    python tools/fetch_heliolinc_orbits.py [path/to/MPCORB.DAT.gz]

Without an argument, MPCORB.DAT.gz (about 90 MB) is downloaded from the Minor
Planet Center into ~/.cache/science-notebooks/. The script then selects every
object that could appear in the notebook's survey field during its 30-day
window, plus (10606) Crocco, and writes notebooks/heliolinc/data/orbits.csv.gz
with the MPCORB header that the MPC asks redistributors to include.
"""
import datetime as dt
import gzip
import pathlib
import sys
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "heliolinc"))
import heliolinc as hl  # noqa: E402

URL = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"
OUT = ROOT / "notebooks" / "heliolinc" / "data" / "orbits.csv.gz"

# The survey window and field, also defined in the notebook.
NEAR_JD = 2461200.5          # 2026 June 9, the MPCORB epoch at the time of writing
FIELD_HALF_DEG = 8.0         # the largest field the notebook uses
MARGIN_DEG = 1.5             # checked daily, so objects moving up to 1.5 deg/day are kept
V_KEEP = 22.5                # one magnitude fainter than the survey limit


def unpack_epoch(s):
    """MPC packed date, e.g. K2669 -> 2026 June 9 (0h TT), as a JD."""
    century = {"I": 1800, "J": 1900, "K": 2000}[s[0]]
    code = "0123456789ABCDEFGHIJKLMNOPQRSTUV"
    d = dt.date(century + int(s[1:3]), code.index(s[3]), code.index(s[4]))
    return d.toordinal() + 1721424.5


def read_mpcorb(path):
    rows = []
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as f:
        body = False
        for line in f:
            if line.startswith("-----"):
                body = True
                continue
            if not body or len(line) < 103 or not line[20:25].strip():
                continue
            try:
                H = float(line[8:13]) if line[8:13].strip() else np.nan
                G = float(line[14:19]) if line[14:19].strip() else 0.15
                rows.append((line[166:194].strip() or line[0:7].strip(), H, G, unpack_epoch(line[20:25]),
                             float(line[92:103]), float(line[70:79]), float(line[59:68]),
                             float(line[48:57]), float(line[37:46]), float(line[26:35])))
            except (ValueError, KeyError):
                continue
    names = np.array([r[0] for r in rows], dtype=object)
    arr = np.array([r[1:] for r in rows], dtype=float)
    return names, arr


def main():
    if len(sys.argv) > 1:
        path = pathlib.Path(sys.argv[1])
    else:
        path = pathlib.Path.home() / ".cache" / "science-notebooks" / "MPCORB.DAT.gz"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {URL} ...")
            urllib.request.urlretrieve(URL, path)
    names, arr = read_mpcorb(path)
    H, G, epoch, a, e, inc, node, peri, M = arr.T
    good = np.isfinite(H) & (e < 1) & (a > 0)
    print(f"Read {len(names)} orbits; {good.sum()} usable")
    orbits = {"a": a, "e": e, "inc": np.radians(inc), "node": np.radians(node),
              "peri": np.radians(peri), "M": np.radians(M), "epoch": epoch, "H": H, "G": G}
    orbits = {k: v[good] for k, v in orbits.items()}
    names = names[good]

    t_c = hl.nearest_new_moon(NEAR_JD) - 69.2 / 86400   # JDE (TT) to roughly UTC; irrelevant at this precision
    RE = hl.earth_position(t_c)
    lam0 = np.arctan2(-RE[1], -RE[0])                # opposition point, on the ecliptic
    print(f"Window centre: new moon at JD {t_c:.4f}; opposition longitude {np.degrees(lam0):.3f} deg")

    lim = np.radians(FIELD_HALF_DEG + MARGIN_DEG)
    centre = hl.unit_radec(lam0, 0.0)
    ex = np.array([-np.sin(lam0), np.cos(lam0), 0.0])
    ey = np.cross(centre, ex)
    keep = np.zeros(len(names), dtype=bool)
    for t in t_c + np.arange(-15, 16):
        u, r, delta, phase = hl.observe(orbits, t)
        cz = u @ centre
        near = (cz > 0) & (np.abs((u @ ex) / cz) < np.tan(lim)) & (np.abs((u @ ey) / cz) < np.tan(lim))
        V = hl.hg_magnitude(orbits["H"], orbits["G"], r, delta, phase)
        keep |= near & (V < V_KEEP)
    keep |= names == "(10606) Crocco"
    print(f"Kept {keep.sum()} orbits")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as f:
        header = []
        for line in f:
            if line.startswith("Des'n"):
                break
            header.append(line.rstrip("\n"))
    with gzip.open(OUT, "wt") as f:
        f.write("# Extract of MPCORB.DAT, the Minor Planet Center orbit database, made by\n")
        f.write(f"# tools/fetch_heliolinc_orbits.py on {dt.date.today().isoformat()}. MPCORB header follows.\n#\n")
        for line in header:
            f.write(("# " + line).rstrip() + "\n")
        f.write(f"#\n# window_centre_jd = {t_c:.6f}\n# opposition_longitude_deg = {np.degrees(lam0):.6f}\n")
        f.write("name,H,G,epoch_jd,a_au,e,inc_deg,node_deg,peri_deg,M_deg\n")
        for i in np.flatnonzero(keep):
            f.write(f"\"{names[i]}\",{orbits['H'][i]:.2f},{orbits['G'][i]:.2f},{orbits['epoch'][i]:.1f},"
                    f"{orbits['a'][i]:.7f},{orbits['e'][i]:.7f},{np.degrees(orbits['inc'][i]):.5f},"
                    f"{np.degrees(orbits['node'][i]):.5f},{np.degrees(orbits['peri'][i]):.5f},"
                    f"{np.degrees(orbits['M'][i]):.5f}\n")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
