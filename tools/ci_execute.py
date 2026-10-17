#!/usr/bin/env python3
"""Execute a notebook as a health check, without writing its outputs back.

Used by the weekly workflow (.github/workflows/notebooks.yml) and by the /fix-notebook skill.

    python tools/ci_execute.py notebooks/<name>/<name>.ipynb            # as committed
    python tools/ci_execute.py notebooks/<name>/<name>.ipynb --quick    # QUICK = True

--quick sets the notebook's `QUICK = False` line to `QUICK = True` (in memory only), so the
expensive parts are recomputed at reduced size into cache_quick/ (git-ignored) instead of being
read from the committed cache/. Notebooks without a QUICK flag run as committed.

On failure, prints the failing cell and the tail of its traceback, writes the same text to
--error-file (if given), and exits with status 1. The notebook file itself is never modified.
"""
import argparse
import pathlib
import re
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

ap = argparse.ArgumentParser()
ap.add_argument("notebook")
ap.add_argument("--quick", action="store_true", help="run with QUICK = True when the notebook has the flag")
ap.add_argument("--timeout", type=int, default=3600, help="seconds allowed per cell (default 3600)")
ap.add_argument("--error-file", help="write a failure summary here")
args = ap.parse_args()

path = pathlib.Path(args.notebook).resolve()
nb = nbformat.read(path, as_version=4)

QUICK_LINE = re.compile(r"^QUICK\s*=\s*False\b", re.M)
mode = "as committed"
if args.quick:
    for cell in nb.cells:
        if cell.cell_type == "code" and QUICK_LINE.search(cell.source):
            cell.source = QUICK_LINE.sub("QUICK = True", cell.source, count=1)
            mode = "QUICK = True"
            break
    else:
        mode = "as committed (no QUICK flag)"
print(f"Executing {path.relative_to(pathlib.Path.cwd()) if path.is_relative_to(pathlib.Path.cwd()) else path} ({mode})", flush=True)

t0 = time.time()
client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                        resources={"metadata": {"path": str(path.parent)}})
try:
    client.execute()
except (CellExecutionError, CellTimeoutError) as err:
    lines = re.sub(r"\x1b\[[0-9;]*m", "", str(err)).splitlines()   # drop terminal colour codes
    summary = "\n".join(lines[-60:])
    msg = f"{path.name} failed after {time.time() - t0:.0f} s ({mode}).\n\n{summary}\n"
    print(msg, file=sys.stderr)
    if args.error_file:
        pathlib.Path(args.error_file).write_text(msg)
    sys.exit(1)
print(f"OK in {time.time() - t0:.0f} s ({mode})")
