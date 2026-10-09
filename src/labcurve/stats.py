"""Descriptive statistics for a sample of measurements."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as scipy_stats

from labcurve.errors import DataError


@dataclass(frozen=True)
class Summary:
    """Descriptive statistics of a one-dimensional sample.

    ``std`` is the sample standard deviation (``ddof=1``), ``sem`` the standard
    error of the mean, and ``ci_low``/``ci_high`` bound the confidence interval
    of the mean obtained from Student's t distribution. For a single value
    these four quantities are undefined and reported as ``nan``.
    """

    n: int
    mean: float
    std: float
    sem: float
    median: float
    minimum: float
    maximum: float
    confidence: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> dict[str, float | int]:
        """Return the statistics as a plain dictionary."""
        return asdict(self)


def summarize(values: ArrayLike, confidence: float = 0.95) -> Summary:
    """Compute descriptive statistics of a sample.

    Parameters
    ----------
    values
        One-dimensional collection of finite numbers.
    confidence
        Confidence level of the interval for the mean, strictly between 0 and 1.

    Returns
    -------
    Summary
        The statistics of the sample.

    Raises
    ------
    DataError
        If the sample is empty, not one-dimensional, contains non-finite
        values, or ``confidence`` is outside ``(0, 1)``.
    """
    if not 0.0 < confidence < 1.0:
        raise DataError(f"confidence must be between 0 and 1, got {confidence}")

    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1:
        raise DataError("values must be one-dimensional")
    if sample.size == 0:
        raise DataError("cannot summarise an empty sample")
    if not np.all(np.isfinite(sample)):
        raise DataError("values must be finite; clean the data first")

    n = int(sample.size)
    mean = float(np.mean(sample))
    if n > 1:
        std = float(np.std(sample, ddof=1))
        sem = std / math.sqrt(n)
        half_width = float(scipy_stats.t.ppf(0.5 + confidence / 2.0, df=n - 1)) * sem
        ci_low, ci_high = mean - half_width, mean + half_width
    else:
        std = sem = ci_low = ci_high = math.nan

    return Summary(
        n=n,
        mean=mean,
        std=std,
        sem=sem,
        median=float(np.median(sample)),
        minimum=float(np.min(sample)),
        maximum=float(np.max(sample)),
        confidence=confidence,
        ci_low=ci_low,
        ci_high=ci_high,
    )
