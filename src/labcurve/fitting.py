"""Least-squares curve fitting with optional robust outlier rejection."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeWarning, curve_fit

from labcurve.errors import DataError, FitError
from labcurve.models import FloatArray, Model, get_model

#: Scales the median absolute deviation to the standard deviation of a normal
#: distribution (Iglewicz and Hoaglin, 1993).
_MAD_SCALE = 0.6745

#: Cut-off for the modified z-score recommended by Iglewicz and Hoaglin (1993).
DEFAULT_OUTLIER_THRESHOLD = 3.5


@dataclass(frozen=True)
class FitResult:
    """Outcome of fitting a model to data.

    Attributes
    ----------
    model, formula
        Name and formula of the fitted model.
    parameters
        Names of the model parameters.
    values, errors
        Best-fit value and one-standard-deviation uncertainty of each
        parameter. An uncertainty is ``nan`` if it could not be estimated.
    r_squared
        Coefficient of determination of the points used in the fit. ``nan``
        if all ``y`` values are equal.
    rmse
        Root-mean-square of the residuals of the points used in the fit.
    reduced_chi2
        Chi-square per degree of freedom. Only available when uncertainties
        were supplied, otherwise ``None``.
    n_points, n_outliers
        Number of points used in the fit and number of points rejected.
    outliers
        Boolean mask over the input points; ``True`` marks a rejected point.
    """

    model: str
    formula: str
    parameters: tuple[str, ...]
    values: tuple[float, ...]
    errors: tuple[float, ...]
    r_squared: float
    rmse: float
    reduced_chi2: float | None
    n_points: int
    n_outliers: int
    outliers: NDArray[np.bool_] = field(repr=False, compare=False)

    def predict(self, x: ArrayLike) -> FloatArray:
        """Evaluate the fitted model at ``x``."""
        function = get_model(self.model).function
        result: FloatArray = function(np.asarray(x, dtype=float), *self.values)
        return result

    def as_dict(self) -> dict[str, Any]:
        """Return the result as a dictionary of plain Python values."""
        return {
            "model": self.model,
            "formula": self.formula,
            "parameters": {
                name: {"value": value, "error": error}
                for name, value, error in zip(
                    self.parameters, self.values, self.errors, strict=True
                )
            },
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "reduced_chi2": self.reduced_chi2,
            "n_points": self.n_points,
            "n_outliers": self.n_outliers,
        }


def flag_outliers(
    residuals: ArrayLike,
    threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    *,
    reference: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Flag residuals whose modified z-score exceeds ``threshold``.

    The modified z-score ``0.6745 * (r - median) / MAD`` uses the median and
    the median absolute deviation (MAD), which, unlike the mean and standard
    deviation, are not distorted by the outliers being searched for.

    Parameters
    ----------
    residuals
        Residuals to classify.
    threshold
        Positive cut-off for the absolute modified z-score.
    reference
        Residuals from which the median and MAD are computed. Defaults to
        ``residuals`` itself.

    Returns
    -------
    numpy.ndarray
        Boolean mask with ``True`` for outliers. If the MAD is zero (more than
        half of the reference residuals are identical) no point is flagged,
        because the scale of the scatter cannot be estimated.
    """
    if threshold <= 0:
        raise DataError(f"outlier threshold must be positive, got {threshold}")
    values = np.asarray(residuals, dtype=float)
    basis = values if reference is None else np.asarray(reference, dtype=float)
    if basis.size == 0:
        return np.zeros(values.shape, dtype=bool)

    median = float(np.median(basis))
    mad = float(np.median(np.abs(basis - median)))
    if mad == 0.0:
        return np.zeros(values.shape, dtype=bool)
    flags: NDArray[np.bool_] = np.abs(_MAD_SCALE * (values - median) / mad) > threshold
    return flags


def _validate(
    x: ArrayLike, y: ArrayLike, yerr: ArrayLike | None, model: Model
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.ndim != 1 or ys.ndim != 1:
        raise DataError("x and y must be one-dimensional")
    if xs.size != ys.size:
        raise DataError(f"x and y must have the same length, got {xs.size} and {ys.size}")
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        raise DataError("x and y must be finite; clean the data first")
    if model.requires_positive_x and np.any(xs <= 0):
        raise DataError(
            f"model {model.name!r} is only defined for x > 0, "
            f"but the smallest x is {float(xs.min()):g}"
        )

    sigma: FloatArray | None = None
    if yerr is not None:
        sigma = np.asarray(yerr, dtype=float)
        if sigma.shape != ys.shape:
            raise DataError("yerr must have the same length as y")
        if not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise DataError("yerr must be finite and strictly positive")

    _require_enough_points(int(xs.size), model)
    return xs, ys, sigma


def _require_enough_points(n_points: int, model: Model) -> None:
    needed = len(model.parameters) + 1
    if n_points < needed:
        raise DataError(
            f"model {model.name!r} needs at least {needed} points, but only {n_points} are usable"
        )


def _least_squares(
    model: Model, x: FloatArray, y: FloatArray, sigma: FloatArray | None
) -> tuple[FloatArray, FloatArray]:
    """Run the optimiser and return the parameters and their covariance."""
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            values, covariance = curve_fit(
                model.function,
                x,
                y,
                p0=model.initial_guess(x, y),
                sigma=sigma,
                absolute_sigma=sigma is not None,
                maxfev=20_000,
            )
    except (RuntimeError, ValueError) as exc:
        raise FitError(f"fit of model {model.name!r} did not converge: {exc}") from exc
    if not np.all(np.isfinite(values)):
        raise FitError(f"fit of model {model.name!r} produced non-finite parameters")
    return np.asarray(values, dtype=float), np.asarray(covariance, dtype=float)


def fit_curve(
    x: ArrayLike,
    y: ArrayLike,
    model: str = "linear",
    *,
    yerr: ArrayLike | None = None,
    outlier_threshold: float | None = None,
    max_iterations: int = 10,
) -> FitResult:
    """Fit a model to data by non-linear least squares.

    When ``outlier_threshold`` is given, the fit is repeated after discarding
    points whose residual has a modified z-score above the threshold (see
    :func:`flag_outliers`) until the set of rejected points stops changing or
    ``max_iterations`` is reached. A rejected point can be re-admitted in a
    later iteration if the improved fit describes it well.

    Parameters
    ----------
    x, y
        Measured values; one-dimensional, finite and of equal length.
    model
        Name of a model registered in :data:`labcurve.models.MODELS`.
    yerr
        Optional standard uncertainties of ``y``. They weight the fit and are
        treated as absolute, so they also determine the parameter errors.
    outlier_threshold
        Modified z-score above which a point is rejected. ``None`` disables
        outlier rejection.
    max_iterations
        Upper bound on the number of rejection passes.

    Returns
    -------
    FitResult
        Best-fit parameters, their uncertainties and goodness-of-fit measures.

    Raises
    ------
    DataError
        If the inputs are invalid or too few points are available.
    FitError
        If the optimiser does not converge.
    """
    chosen = get_model(model)
    xs, ys, sigma = _validate(x, y, yerr, chosen)
    if max_iterations < 1:
        raise DataError(f"max_iterations must be at least 1, got {max_iterations}")

    keep = np.ones(xs.size, dtype=bool)
    for _ in range(max_iterations):
        _require_enough_points(int(keep.sum()), chosen)
        values, covariance = _least_squares(
            chosen, xs[keep], ys[keep], None if sigma is None else sigma[keep]
        )
        if outlier_threshold is None:
            break
        residuals = ys - chosen.function(xs, *values)
        new_keep = ~flag_outliers(residuals, outlier_threshold, reference=residuals[keep])
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    else:
        # The last pass changed the selection, so the parameters must be refreshed.
        _require_enough_points(int(keep.sum()), chosen)
        values, covariance = _least_squares(
            chosen, xs[keep], ys[keep], None if sigma is None else sigma[keep]
        )

    residuals = ys[keep] - chosen.function(xs[keep], *values)
    n_used = int(keep.sum())
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((ys[keep] - np.mean(ys[keep])) ** 2))
    variances = np.diag(covariance)
    errors = np.where(np.isfinite(variances) & (variances >= 0), variances, np.nan) ** 0.5

    reduced_chi2: float | None = None
    if sigma is not None:
        dof = n_used - len(chosen.parameters)
        reduced_chi2 = float(np.sum((residuals / sigma[keep]) ** 2)) / dof

    return FitResult(
        model=chosen.name,
        formula=chosen.formula,
        parameters=chosen.parameters,
        values=tuple(float(value) for value in values),
        errors=tuple(float(error) for error in errors),
        r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan,
        rmse=math.sqrt(ss_res / n_used),
        reduced_chi2=reduced_chi2,
        n_points=n_used,
        n_outliers=int(xs.size - n_used),
        outliers=~keep,
    )
