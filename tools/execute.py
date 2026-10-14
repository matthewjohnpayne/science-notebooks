#!/usr/bin/env python3
"""Execute a notebook in place, from its own folder, so outputs are stored with it.

    python tools/execute.py notebooks/<name>/<name>.ipynb
"""
import pathlib
import sys

import nbformat
from nbclient import NotebookClient

path = pathlib.Path(sys.argv[1]).resolve()
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
nbformat.write(nb, path)
print(f"Executed {path}")
