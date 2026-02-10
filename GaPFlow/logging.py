#
# Copyright 2026 Hannes Holey
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def _default_filename_for(name: str) -> str:
    # map logger names like 'gapflow.problem' -> 'gapflow_problem.log'
    base = name.replace('.', '_')
    return f"{base}.log"


def get_logger(name: str,
               outdir: Optional[str] = None,
               filename: Optional[str] = None,
               level: int = logging.INFO,
               force: bool = False) -> logging.Logger:
    """Return a standardised logger for GaPFlow modules.

    Parameters
    ----------
    name : str
        Name of the logger.
    outdir : str, optional
        Output directory to write logfiles (the default is None, which only writes to stdout)
    filename : str, optional
        Output filename of the logger (the default is None, which uses a default filename)
    level : int, optional
        Log level (the default is logging.INFO)
    force : bool, optional
        If true, replace existing handlers to allow reconfiguration (the default is False)

    Returns
    -------
    logging.Logger
        The logger object
    """

    logger = logging.getLogger(name)

    if filename is None:
        filename = _default_filename_for(name)

    if outdir is None:
        filepath = None
    else:
        filepath = os.path.join(outdir, filename)

    if force:
        # Remove existing handlers so we can reconfigure logfile location
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    # If handlers already exist and force is False, return the logger unchanged
    if logger.handlers and not force:
        return logger

    logger.setLevel(level)

    # Stream handler to stdout (always present)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)

    # Only add FileHandler if an outdir (filepath) is specified
    if filepath is not None:
        # Ensure output directory exists
        try:
            os.makedirs(outdir, exist_ok=True)
        except Exception:
            # If creation fails, fall back to CWD (but don't create a file there on initial call)
            pass

        fh = logging.FileHandler(filepath)
        fh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)

    logger.propagate = False

    return logger
