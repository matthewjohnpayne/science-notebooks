# Instructions for agents

This repository holds Matthew Payne's research-based Jupyter notebooks. His personal website (`matthewjohnpayne.github.io`, local copy `~/git_repos/ScientistWebsite`) links to them; its `docs/ROADMAP.md` lists further notebook ideas, each tied to one of his papers.

## Rules

1. **Real, verified physics.** Every figure is computed in the notebook. Check numerical results against the source paper and say where they differ. Never adjust results to match expectations.
2. **Cite precisely.** Formulas and coefficients come from the paper's text; quote equation forms exactly and give ADS links. Check bibliographic details against NASA ADS. Never invent references.
3. **Do not invent claims** about the papers, their authors or their reception.
4. **Structure:**
   - Each notebook gets its own folder `notebooks/<name>/`, containing `<name>.ipynb` and an optional helper module.
   - Multiprocessing workers must live in the helper module, not in the notebook, so that spawn-based platforms (macOS, Windows) work.
   - Generate the notebook from `tools/build_<name>.py`, then execute it with `tools/execute.py`, and commit it with outputs.
5. **Style:** follow the MPC tutorial style that Matthew uses.
   - Open with a title, author line, motivation and a numbered "We will" list.
   - Use numbered sections, a Summary, Going Further and References.
   - Offer a `QUICK` flag when a full run takes more than about a minute.
6. Do not push to GitHub, or create the GitHub repository, without Matthew's go-ahead.

## Orbital-uncertainty series

The notebooks listed in `docs/orbit_uncertainty/PLAN.md` are an exception to "tied to a paper". They explore orbital and on-sky uncertainty (NEOCP tracklets, impact probability, impact corridors, occultations, Roman), share the toolkit in `orbunc/` (tests in `orbunc/tests`), and may later move to `generic_mpcsoftware_dev/mpc-research`. Keep `orbunc` self-contained, keep PLAN.md up to date, and re-run the tests after changing it.
