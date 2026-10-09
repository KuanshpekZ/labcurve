"""labcurve: clean, summarise, fit and plot experimental measurements."""

from labcurve.cleaning import CleaningReport, clean_measurements
from labcurve.errors import DataError, FitError, LabcurveError
from labcurve.io import load_measurements
from labcurve.models import MODELS, Model, get_model
from labcurve.stats import Summary, summarize

__version__ = "0.1.0"

__all__ = [
    "MODELS",
    "CleaningReport",
    "DataError",
    "FitError",
    "LabcurveError",
    "Model",
    "Summary",
    "__version__",
    "clean_measurements",
    "get_model",
    "load_measurements",
    "summarize",
]
