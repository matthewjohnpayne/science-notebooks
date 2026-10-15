# Science notebooks

Jupyter notebooks by **Matthew Payne** that explore the dynamics and methods behind his research papers, with each result reproduced by direct computation.

These notebooks complement, and do not duplicate, the [Minor Planet Center tutorials](https://docs.minorplanetcenter.net/tutorials/), which cover working with MPC data and services. Notebooks here are tied to research papers rather than to MPC services.

## Notebooks

| Notebook | Based on | Run time |
|---|---|---|
| [Resonance overlap: when do two close planets become chaotic?](notebooks/resonance_overlap/resonance_overlap.ipynb) | Deck, Payne & Holman (2013), ApJ 774, 129 | Seconds from the cached results in `cache/`; about 2 h on 12 cores to recompute all ~470,000 integrations; a few minutes with `QUICK = True` |
| [HelioLinC: linking asteroid detections by guessing their distance](notebooks/heliolinc/heliolinc.ipynb) | Holman, Payne, Blankley, Janssen & Kuindersma (2018), AJ 156, 135 | About 3 minutes |
| [Transit timing variations from inclined and retrograde companions](notebooks/ttv_inclined/ttv_inclined.ipynb) | Payne, Ford & Veras (2010), ApJL 712, L86 | Seconds from the cached results in `cache/`; about 50 minutes on 12 cores to recompute all ~100,000 integrations; a few minutes with `QUICK = True` |

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
