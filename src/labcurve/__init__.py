"""labcurve: clean, summarise, fit and plot experimental measurements."""

from labcurve.cleaning import CleaningReport, clean_measurements
from labcurve.errors import DataError, FitError, LabcurveError
from labcurve.io import load_measurements

__version__ = "0.1.0"

__all__ = [
    "CleaningReport",
    "DataError",
    "FitError",
    "LabcurveError",
    "__version__",
    "clean_measurements",
    "load_measurements",
]
