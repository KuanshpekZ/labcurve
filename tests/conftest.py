"""Fixtures shared by the test modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cooling_csv() -> Path:
    """Path to the synthetic cooling data set shipped with the examples."""
    return REPO_ROOT / "examples" / "cooling.csv"


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator, so that every test run sees the same numbers."""
    return np.random.default_rng(12345)
