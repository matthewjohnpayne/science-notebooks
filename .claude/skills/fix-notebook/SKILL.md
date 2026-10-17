---
name: fix-notebook
description: Repair a science notebook that the weekly check (.github/workflows/notebooks.yml) reports as failing, rebuild and re-execute it, commit it with outputs and close its GitHub issue. Use when asked to fix a failing notebook or given a "Notebook failing: <name>" issue.
argument-hint: <notebook name, e.g. ttv_inclined>
disable-model-invocation: true
---

# Fix a failing notebook: $ARGUMENTS

The weekly check executes every notebook with the newest dependency versions that
`requirements.txt` allows, with `QUICK = True` where a notebook has that flag, and opens an issue
titled `Notebook failing: <name>` (label `claude-task`) when one fails. Follow `AGENTS.md`
throughout: real, verified physics; precise citations, never invented; one folder per notebook;
build from `tools/build_<name>.py`, execute with `tools/execute.py`, commit with outputs.

1. **Read the issue.** `gh issue list --repo matthewjohnpayne/science-notebooks --label claude-task --search "Notebook failing: $ARGUMENTS in:title"`, then `gh issue view <n>` for the error and the package versions in the linked run.
2. **Reproduce it locally**, QUICK first, in a clean environment with the newest allowed versions:
   ```bash
   python -m venv /tmp/nbcheck && /tmp/nbcheck/bin/pip install -U -r requirements.txt
   MPLBACKEND=Agg /tmp/nbcheck/bin/python tools/ci_execute.py notebooks/$ARGUMENTS/$ARGUMENTS.ipynb --quick
   ```
   `tools/ci_execute.py` never writes to the notebook. If it passes locally, compare package versions with the run's log before changing anything.
3. **Fix the cause in the helper module** (`notebooks/$ARGUMENTS/*.py`) **or the builder** (`tools/build_$ARGUMENTS.py`). Never hand-edit the `.ipynb`. Prefer code that works across versions over pinning; pin a dependency in `requirements.txt` only if a fix is impossible, and say why in a comment.
4. **Keep the science unchanged.** Do not alter physics, parameters, figures or citations unless the fix requires it. If it does, explain exactly what changed and why in the commit message and in your summary to Matthew, and update any numbers quoted in the notebook text.
5. **Rebuild and re-execute:**
   ```bash
   python tools/build_$ARGUMENTS.py
   python tools/ci_execute.py notebooks/$ARGUMENTS/$ARGUMENTS.ipynb --quick     # quick check
   python tools/execute.py notebooks/$ARGUMENTS/$ARGUMENTS.ipynb                # full run, outputs stored
   ```
   If the change could affect cached results (`cache/`), re-run with `RECOMPUTE = True` in a scratch copy and refresh the cache, noting the run time. Long runs: start them in the background and check on them rather than waiting.
6. **Check the outputs** against the previous commit (`git diff --stat`, and look at figures and quoted numbers). Results should agree unless step 4 explained why not.
7. **Commit** the builder, helper module, notebook with outputs and any cache, with a message saying what broke and how it was fixed, ending with the Co-Authored-By line the session uses.
8. **Close the issue** with a comment summarising the fix and the commit hash (noting that the commit is local until pushed): `gh issue close <n> --comment "..."`.
9. **Ask Matthew before pushing.** Never push without his explicit go-ahead.
