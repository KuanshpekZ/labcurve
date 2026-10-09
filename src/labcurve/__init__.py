"""labcurve: clean, summarise, fit and plot experimental measurements.

Typical use from Python::

    from labcurve import load_measurements, clean_measurements, fit_curve

    raw = load_measurements("cooling.csv", x="time_s", y="temp_c")
    data, report = clean_measurements(raw, x="time_s", y="temp_c")
    fit = fit_curve(data["time_s"], data["temp_c"], model="exponential")
"""

from labcurve.cleaning import CleaningReport, clean_measurements
from labcurve.errors import DataError, FitError, LabcurveError
from labcurve.fitting import FitResult, fit_curve, flag_outliers
from labcurve.io import load_measurements
from labcurve.models import MODELS, Model, get_model
from labcurve.plotting import plot_fit, save_figure
from labcurve.stats import Summary, summarize

__version__ = "0.1.0"

__all__ = [
    "MODELS",
    "CleaningReport",
    "DataError",
    "FitError",
    "FitResult",
    "LabcurveError",
    "Model",
    "Summary",
    "__version__",
    "clean_measurements",
    "fit_curve",
    "flag_outliers",
    "get_model",
    "load_measurements",
    "plot_fit",
    "save_figure",
    "summarize",
]
