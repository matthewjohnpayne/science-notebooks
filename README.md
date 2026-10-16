# Science notebooks

Jupyter notebooks by **Matthew Payne** that explore the dynamics and methods behind his research papers, with each result reproduced by direct computation.

These notebooks complement, and do not duplicate, the [Minor Planet Center tutorials](https://docs.minorplanetcenter.net/tutorials/), which cover working with MPC data and services. Notebooks here are tied to research papers rather than to MPC services.

## Notebooks

| Notebook | Based on | Run time |
|---|---|---|
| [Resonance overlap: when do two close planets become chaotic?](notebooks/resonance_overlap/resonance_overlap.ipynb) | Deck, Payne & Holman (2013), ApJ 774, 129 | Seconds from the cached results in `cache/`; about 2 h on 12 cores to recompute all ~470,000 integrations; a few minutes with `QUICK = True` |
| [HelioLinC: linking asteroid detections by guessing their distance](notebooks/heliolinc/heliolinc.ipynb) | Holman, Payne, Blankley, Janssen & Kuindersma (2018), AJ 156, 135 | About 3 minutes |
| [Transit timing variations from inclined and retrograde companions](notebooks/ttv_inclined/ttv_inclined.ipynb) | Payne, Ford & Veras (2010), ApJL 712, L86 | Seconds from the cached results in `cache/`; about 50 minutes on 12 cores to recompute all ~100,000 integrations; a few minutes with `QUICK = True` |
| [Interstellar objects in the LSST era: which will Rubin see?](notebooks/iso_lsst/iso_lsst.ipynb) | Hoover, Seligman & Payne (2022), PSJ 3, 71 | About 6 minutes on 12 cores (a million objects, plus five variants); about a minute with `QUICK = True` |
| [Shift-and-stack on detection catalogues](notebooks/shift_and_stack/shift_and_stack.ipynb) | Stetzler et al. (2025), AJ 170, 352 | About 3 minutes |
| [The Laplace resonance of GJ 876](notebooks/laplace_gj876/laplace_gj876.ipynb) | Nelson et al. (2016), MNRAS 455, 2484 | About a minute |
| [Fitting interacting planets: N-body radial velocities and differential-evolution MCMC](notebooks/rv_demcmc/rv_demcmc.ipynb) | Nelson, Ford & Payne (2014), ApJS 210, 11 | About 30 minutes on 12 cores; a few minutes with `QUICK = True` |
| [Can moons survive in tightly packed planetary systems?](notebooks/moons_stips/moons_stips.ipynb) | Payne, Deck, Holman & Perets (2013), ApJL 775, L44 | Seconds from the cached results in `cache/`; about 20 minutes on 12 cores to recompute; a few minutes with `QUICK = True` |
| [How planets lose their moons around white dwarfs](notebooks/exomoons_wd/exomoons_wd.ipynb) | Payne, Veras, Holman & Gänsicke (2016), MNRAS 457, 217; Payne, Veras, Gänsicke & Holman (2017), MNRAS 464, 2557 | A few minutes from the cached results in `cache/`; about 2 h on 12 cores to recompute; `QUICK = True` for 1,000 moons per encounter |
| [Planet Nine and the Cassini ranging data](notebooks/cassini_planet9/cassini_planet9.ipynb) | Holman & Payne (2016), AJ 152, 94 | Under a minute |
| [2010 JO179: a dwarf planet in the 21:5 resonance with Neptune](notebooks/jo179_resonance/jo179_resonance.ipynb) | Holman, Payne, Fraser et al. (2018), ApJL 855, L6 | Seconds from the cached results in `cache/`; about 2 h on 12 cores to recompute; a few minutes with `QUICK = True` |

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/resonance_overlap/resonance_overlap.ipynb
```

Each notebook lives in its own folder with any helper module it imports. The notebooks are committed with their outputs, so they can be read on GitHub without running them.

## Editing

Notebooks are generated from a plain-text builder in `tools/`, so prose and code can be reviewed together:

```bash
python tools/build_resonance_overlap.py                                   # write the .ipynb
python tools/execute.py notebooks/resonance_overlap/resonance_overlap.ipynb   # run it and store outputs
```

The HelioLinC notebook reads real orbits from `notebooks/heliolinc/data/orbits.csv.gz`, an extract of the MPC's MPCORB file. To regenerate it from a fresh MPCORB (about 90 MB, downloaded automatically), run `python tools/fetch_heliolinc_orbits.py`.

## License

- **Code** (helper modules, builders and the code in notebook cells) is released under the MIT License (`LICENSE`).
- **Text and figures** in the notebooks are © Matthew Payne, all rights reserved.
