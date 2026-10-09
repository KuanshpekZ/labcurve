"""Model functions that can be fitted to measurements.

Each model bundles the function itself with the names of its parameters and a
data-driven initial guess, so that non-linear fits start close to the solution
and the user does not have to supply starting values.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from labcurve.errors import DataError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Model:
    """A parametric function ``y = f(x, *parameters)``."""

    name: str
    formula: str
    description: str
    parameters: tuple[str, ...]
    function: Callable[..., FloatArray]
    initial_guess: Callable[[FloatArray, FloatArray], tuple[float, ...]]
    #: Whether the function is only defined for strictly positive ``x``.
    requires_positive_x: bool = False


def linear(x: FloatArray, a: float, b: float) -> FloatArray:
    """Evaluate the straight line ``a * x + b``."""
    return a * x + b


def _guess_linear(x: FloatArray, y: FloatArray) -> tuple[float, ...]:
    span = float(np.ptp(x))
    if span == 0.0:
        return 0.0, float(np.mean(y))
    slope = float((y[np.argmax(x)] - y[np.argmin(x)]) / span)
    return slope, float(np.mean(y) - slope * np.mean(x))


def exponential(x: FloatArray, a: float, tau: float, c: float) -> FloatArray:
    """Evaluate the exponential relaxation ``a * exp(-x / tau) + c``."""
    return a * np.exp(-x / tau) + c


def _guess_exponential(x: FloatArray, y: FloatArray) -> tuple[float, ...]:
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    span = float(xs[-1] - xs[0])
    tail = max(1, xs.size // 10)
    offset = float(np.mean(ys[-tail:]))
    amplitude = float(ys[0] - offset)
    if span == 0.0 or amplitude == 0.0:
        return 1.0, 1.0, offset

    # The time constant is where the signal has decayed to 1/e of its amplitude.
    decayed = np.flatnonzero(np.abs(ys - offset) <= abs(amplitude) / math.e)
    tau = float(xs[decayed[0]] - xs[0]) if decayed.size else span
    if tau <= 0.0:
        tau = span / 3.0
    return amplitude, tau, offset


def power(x: FloatArray, a: float, b: float) -> FloatArray:
    """Evaluate the power law ``a * x**b`` for positive ``x``."""
    return a * np.power(x, b)


def _guess_power(x: FloatArray, y: FloatArray) -> tuple[float, ...]:
    # A power law is a straight line in log-log coordinates.
    same_sign = bool(np.all(y > 0) or np.all(y < 0))
    if not same_sign or np.any(x <= 0) or np.ptp(x) == 0.0:
        return 1.0, 1.0
    exponent, log_scale = np.polyfit(np.log(x), np.log(np.abs(y)), 1)
    return float(np.sign(y[0]) * np.exp(log_scale)), float(exponent)


MODELS: dict[str, Model] = {
    model.name: model
    for model in (
        Model(
            name="linear",
            formula="y = a * x + b",
            description="Straight line, e.g. calibration curves or Hooke's law.",
            parameters=("a", "b"),
            function=linear,
            initial_guess=_guess_linear,
        ),
        Model(
            name="exponential",
            formula="y = a * exp(-x / tau) + c",
            description="Exponential relaxation, e.g. cooling or capacitor discharge.",
            parameters=("a", "tau", "c"),
            function=exponential,
            initial_guess=_guess_exponential,
        ),
        Model(
            name="power",
            formula="y = a * x^b",
            description="Power law, e.g. pendulum period or scaling relations (needs x > 0).",
            parameters=("a", "b"),
            function=power,
            initial_guess=_guess_power,
            requires_positive_x=True,
        ),
    )
}


def get_model(name: str) -> Model:
    """Look up a model by name.

    Raises
    ------
    DataError
        If no model with this name is registered.
    """
    try:
        return MODELS[name]
    except KeyError:
        available = ", ".join(sorted(MODELS))
        raise DataError(f"unknown model {name!r}. Available models: {available}") from None
