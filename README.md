# Science notebooks

Jupyter notebooks by **Matthew Payne** that explore the dynamics and methods behind his research papers, with each result reproduced by direct computation.

These notebooks complement, and do not duplicate, the [Minor Planet Center tutorials](https://docs.minorplanetcenter.net/tutorials/), which cover working with MPC data and services. Notebooks here are tied to research topics that are not specific to the MPC.

## Notebooks

| Notebook | Based on | Run time |
|---|---|---|
| [Resonance overlap: when do two close planets become chaotic?](notebooks/resonance_overlap/resonance_overlap.ipynb) | Deck, Payne & Holman (2013), ApJ 774, 129 | About 2.5 min on 12 cores; seconds with `QUICK = True` |

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

## License

- **Code** (helper modules, builders and the code in notebook cells) is released under the MIT License (`LICENSE`).
- **Text and figures** in the notebooks are © Matthew Payne, all rights reserved.
