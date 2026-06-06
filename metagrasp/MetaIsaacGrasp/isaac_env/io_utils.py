# SPDX-License-Identifier: BSD-3-Clause
"""Pickle I/O helpers removed from isaaclab.utils.io in Isaac Lab >= 2.3."""

from __future__ import annotations

import os
import pickle
from typing import Any


def load_pickle(filename: str) -> Any:
    """Load a pickle file (matches legacy ``isaaclab.utils.io.load_pickle``)."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename, "rb") as f:
        return pickle.load(f)


def dump_pickle(filename: str, data: Any) -> None:
    """Save data to a pickle file (matches legacy ``isaaclab.utils.io.dump_pickle``)."""
    if not filename.endswith("pkl"):
        filename += ".pkl"
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)
