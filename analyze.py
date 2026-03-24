#!/usr/bin/env python3
"""Post-processing script: recomputes S_MFU/S_MBU from sweep results and plots them."""

import glob
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Optional


def find_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    """Return the Path matching pattern with the latest name (lexicographic), or None."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        warnings.warn(f"Multiple files match {pattern} in {directory}; using latest: {matches[-1].name}")
    return matches[-1]
