"""Publication-style figure of the data, the fitted curve and the residuals."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from labcurve.errors import DataError
from labcurve.fitting import FitResult

_INK = "#52514e"
_MUTED = "#8a8985"
_GRID = "#e4e3df"
_FIT = "#2a78d6"
_OUTLIER = "#eb6834"


def format_parameter(value: float, error: float, pm: str = "±") -> str:
    """Format ``value ± error`` with a sensible number of digits.

    The uncertainty is rounded to two significant digits and the value to the
    same decimal place. If the uncertainty is unknown, only the value is shown.
    """
    if not math.isfinite(error) or error <= 0:
        return f"{value:.5g}"
    decimals = max(0, 1 - math.floor(math.log10(error)))
    if decimals > 12:
        return f"{value:.5g} {pm} {error:.2g}"
    return f"{value:.{decimals}f} {pm} {error:.{decimals}f}"


def plot_fit(
    x: ArrayLike,
    y: ArrayLike,
    result: FitResult,
    *,
    yerr: ArrayLike | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str | None = None,
) -> Figure:
    """Draw the measurements with the fitted curve above a residual panel.

    The figure is created with the non-interactive Agg backend and does not
    touch ``matplotlib.pyplot``, so it works on machines without a display
    (for example continuous-integration runners) and leaves no global state.

    Parameters
    ----------
    x, y
        The points that were passed to :func:`labcurve.fitting.fit_curve`.
    result
        The corresponding fit result.
    yerr
        Optional uncertainties of ``y``, drawn as error bars.
    xlabel, ylabel
        Axis labels, ideally including units.
    title
        Heading of the figure. Defaults to the formula of the model.

    Returns
    -------
    matplotlib.figure.Figure
        The figure; save it with :func:`save_figure`.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.shape != ys.shape or xs.shape != result.outliers.shape:
        raise DataError("x and y must be the same points that were used for the fit")
    sigma = None if yerr is None else np.asarray(yerr, dtype=float)
    if sigma is not None and sigma.shape != ys.shape:
        raise DataError("yerr must have the same length as y")

    keep = ~result.outliers
    residuals = ys - result.predict(xs)

    figure = Figure(figsize=(7.0, 5.2), dpi=100, layout="constrained")
    FigureCanvasAgg(figure)
    top, bottom = figure.subplots(2, 1, sharex=True, height_ratios=[3, 1])

    for axis, values in ((top, ys), (bottom, residuals)):
        axis.errorbar(
            xs[keep],
            values[keep],
            yerr=None if sigma is None else sigma[keep],
            fmt="o",
            markersize=4,
            color=_INK,
            ecolor=_MUTED,
            elinewidth=0.8,
            capsize=0,
            label="data",
            zorder=2,
        )
        if result.n_outliers:
            axis.plot(
                xs[~keep],
                values[~keep],
                linestyle="none",
                marker="x",
                markersize=7,
                markeredgewidth=1.8,
                color=_OUTLIER,
                label=f"rejected ({result.n_outliers})",
                zorder=3,
            )
        axis.grid(True, color=_GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(_MUTED)
        axis.tick_params(colors=_INK, labelsize=9)

    grid = np.linspace(float(xs.min()), float(xs.max()), 400)
    top.plot(grid, result.predict(grid), color=_FIT, linewidth=2, label="fit", zorder=4)
    bottom.axhline(0.0, color=_FIT, linewidth=1.5, zorder=1)

    parameters = ",  ".join(
        f"{name} = {format_parameter(value, error)}"
        for name, value, error in zip(result.parameters, result.values, result.errors, strict=True)
    )
    quality = f"R² = {result.r_squared:.4f}"
    if result.reduced_chi2 is not None:
        quality += f",  χ²/dof = {result.reduced_chi2:.2f}"
    top.set_title(f"{parameters}\n{quality}", loc="left", fontsize=9, color=_INK)
    figure.suptitle(title or result.formula, x=0.02, ha="left", fontsize=12, fontweight="bold")

    top.set_ylabel(ylabel, fontsize=10)
    # Legend order: data, fit, rejected - independent of the drawing order.
    handles, labels = top.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: (labels[i] != "data", labels[i] != "fit"))
    top.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=9)
    bottom.set_ylabel("residual", fontsize=10)
    bottom.set_xlabel(xlabel, fontsize=10)
    return figure


def save_figure(figure: Figure, path: str | Path, dpi: int = 150) -> Path:
    """Write a figure to disk, creating the parent folder if needed.

    The file format follows the extension of ``path`` (for example ``.png``,
    ``.pdf`` or ``.svg``).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi)
    return target
