"""Exceptions raised by labcurve.

Both exceptions derive from :class:`LabcurveError`, so callers that do not care
about the difference can catch a single type.
"""

from __future__ import annotations


class LabcurveError(Exception):
    """Base class for all errors raised deliberately by labcurve."""


class DataError(LabcurveError, ValueError):
    """The input data is missing, malformed or unsuitable for analysis."""


class FitError(LabcurveError, RuntimeError):
    """The requested model could not be fitted to the data."""
