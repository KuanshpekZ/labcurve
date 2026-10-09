from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from labcurve import (
    DataError,
    FitError,
    clean_measurements,
    fit_curve,
    flag_outliers,
    load_measurements,
)
from labcurve.models import MODELS, Model


def test_exact_line_is_recovered() -> None:
    x = np.arange(10.0)
    result = fit_curve(x, 2.5 * x - 4.0, "linear")
    assert result.values == pytest.approx((2.5, -4.0))
    assert result.r_squared == pytest.approx(1.0)
    assert result.rmse == pytest.approx(0.0, abs=1e-9)
    assert result.reduced_chi2 is None
    assert (result.n_points, result.n_outliers) == (10, 0)
    assert not result.outliers.any()


def test_linear_fit_agrees_with_numpy_polyfit(rng: np.random.Generator) -> None:
    x = np.linspace(-3.0, 8.0, 40)
    y = 1.7 * x + 0.4 + rng.normal(0.0, 0.3, x.size)
    coefficients, covariance = np.polyfit(x, y, 1, cov=True)
    result = fit_curve(x, y, "linear")
    assert result.values == pytest.approx(tuple(coefficients), rel=1e-6)
    assert result.errors == pytest.approx(tuple(np.sqrt(np.diag(covariance))), rel=1e-4)


def test_noisy_exponential_is_recovered_within_its_uncertainties(rng: np.random.Generator) -> None:
    truth = (40.0, 12.0, 20.0)
    x = np.linspace(0.0, 80.0, 120)
    y = truth[0] * np.exp(-x / truth[1]) + truth[2] + rng.normal(0.0, 0.4, x.size)
    result = fit_curve(x, y, "exponential", yerr=np.full(x.size, 0.4))
    for value, error, expected in zip(result.values, result.errors, truth, strict=True):
        assert abs(value - expected) < 4.0 * error
    assert result.r_squared > 0.99
    assert result.reduced_chi2 == pytest.approx(1.0, abs=0.5)
    assert result.rmse == pytest.approx(0.4, rel=0.25)


def test_input_order_does_not_matter(rng: np.random.Generator) -> None:
    x = np.linspace(0.0, 50.0, 60)
    y = 30.0 * np.exp(-x / 9.0) + 5.0 + rng.normal(0.0, 0.2, x.size)
    shuffled = rng.permutation(x.size)
    ordered = fit_curve(x, y, "exponential")
    scrambled = fit_curve(x[shuffled], y[shuffled], "exponential")
    assert scrambled.values == pytest.approx(ordered.values, rel=1e-5)


def test_uncertainties_weight_the_fit() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 30.0])
    trusted = fit_curve(x, y, "linear", yerr=[0.1, 0.1, 0.1, 100.0])
    assert trusted.values == pytest.approx((1.0, 0.0), abs=0.01)
    assert fit_curve(x, y, "linear").values[0] > 5.0


def test_absolute_uncertainties_scale_the_parameter_errors(rng: np.random.Generator) -> None:
    x = np.linspace(0.0, 10.0, 30)
    y = 2.0 * x + rng.normal(0.0, 1.0, x.size)
    small = fit_curve(x, y, "linear", yerr=np.full(x.size, 1.0))
    large = fit_curve(x, y, "linear", yerr=np.full(x.size, 3.0))
    assert large.errors == pytest.approx(tuple(3.0 * e for e in small.errors))
    assert large.reduced_chi2 == pytest.approx(small.reduced_chi2 / 9.0)


def test_predict_evaluates_the_fitted_model() -> None:
    x = np.arange(6.0)
    result = fit_curve(x, 3.0 * x + 1.0, "linear")
    np.testing.assert_allclose(result.predict([10.0, 20.0]), [31.0, 61.0])


def test_constant_data_has_undefined_r_squared() -> None:
    result = fit_curve(np.arange(5.0), np.full(5, 2.0), "linear")
    assert result.values == pytest.approx((0.0, 2.0), abs=1e-9)
    assert math.isnan(result.r_squared)


def test_as_dict_contains_plain_values() -> None:
    summary = fit_curve(np.arange(5.0), np.arange(5.0) * 2.0, "linear").as_dict()
    assert set(summary["parameters"]) == {"a", "b"}
    assert summary["parameters"]["a"]["value"] == pytest.approx(2.0)
    assert summary["model"] == "linear"
    assert "outliers" not in summary


def test_power_law_recovers_the_pendulum_relation(rng: np.random.Generator) -> None:
    # Period of a simple pendulum: T = 2 * pi * sqrt(L / g) = (2 * pi / sqrt(g)) * L**0.5
    gravity = 9.81
    length = np.linspace(0.1, 2.0, 40)
    period = 2.0 * math.pi * np.sqrt(length / gravity) + rng.normal(0.0, 0.005, length.size)
    result = fit_curve(length, period, "power", yerr=np.full(length.size, 0.005))
    scale, exponent = result.values
    assert abs(exponent - 0.5) < 4.0 * result.errors[1]
    assert abs(scale - 2.0 * math.pi / math.sqrt(gravity)) < 4.0 * result.errors[0]
    assert (2.0 * math.pi / scale) ** 2 == pytest.approx(gravity, rel=0.01)
    assert result.r_squared > 0.999


@pytest.mark.parametrize("smallest", [0.0, -1.5])
def test_power_law_rejects_non_positive_x(smallest: float) -> None:
    x = np.array([smallest, 1.0, 2.0, 3.0])
    with pytest.raises(DataError, match=r"only defined for x > 0.*smallest x is"):
        fit_curve(x, np.array([1.0, 2.0, 3.0, 4.0]), "power")


# --- outlier rejection -------------------------------------------------------


def test_flag_outliers_finds_the_obvious_one() -> None:
    residuals = np.array([0.1, -0.2, 0.05, 0.15, -0.1, 8.0, -0.05])
    assert flag_outliers(residuals).tolist() == [False] * 5 + [True, False]


def test_flag_outliers_threshold_controls_the_strictness() -> None:
    residuals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    assert flag_outliers(residuals, 10.0).sum() == 0
    assert flag_outliers(residuals, 1.5).sum() >= 1


def test_flag_outliers_with_zero_scatter_flags_nothing() -> None:
    assert not flag_outliers([0.0, 0.0, 0.0, 0.0, 5.0]).any()


def test_flag_outliers_handles_empty_input() -> None:
    assert flag_outliers([]).shape == (0,)
    assert not flag_outliers([1.0, 2.0], reference=[]).any()


def test_flag_outliers_uses_the_reference_sample() -> None:
    flags = flag_outliers([0.0, 4.0], reference=[-0.1, 0.0, 0.1, 0.05, -0.05])
    assert flags.tolist() == [False, True]


@pytest.mark.parametrize("threshold", [0.0, -1.0])
def test_flag_outliers_rejects_non_positive_thresholds(threshold: float) -> None:
    with pytest.raises(DataError, match="positive"):
        flag_outliers([1.0, 2.0], threshold)


def test_rejection_removes_planted_outliers_and_improves_the_fit(
    rng: np.random.Generator,
) -> None:
    x = np.linspace(0.0, 10.0, 50)
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.2, x.size)
    y[[5, 30]] += [9.0, -7.0]

    plain = fit_curve(x, y, "linear")
    robust = fit_curve(x, y, "linear", outlier_threshold=5.0)

    assert np.flatnonzero(robust.outliers).tolist() == [5, 30]
    assert (robust.n_points, robust.n_outliers) == (48, 2)
    assert robust.rmse < plain.rmse / 3.0
    assert robust.values == pytest.approx((2.0, 1.0), abs=0.1)


def test_rejection_leaves_clean_data_alone(rng: np.random.Generator) -> None:
    x = np.linspace(0.0, 10.0, 200)
    y = 2.0 * x + rng.normal(0.0, 0.2, x.size)
    assert fit_curve(x, y, "linear", outlier_threshold=6.0).n_outliers == 0


def test_rejection_stops_at_max_iterations_with_consistent_result() -> None:
    x = np.arange(12.0)
    y = x.copy()
    y[3], y[8] = 40.0, -30.0
    result = fit_curve(x, y, "linear", outlier_threshold=2.0, max_iterations=1)
    assert result.n_points + result.n_outliers == 12
    assert result.n_points == int((~result.outliers).sum())
    refit = fit_curve(x[~result.outliers], y[~result.outliers], "linear")
    assert result.values == pytest.approx(refit.values)


def test_example_data_recovers_the_true_cooling_parameters(cooling_csv: Path) -> None:
    raw = load_measurements(cooling_csv, "time_s", "temp_c", "temp_err_c")
    data, _ = clean_measurements(raw, "time_s", "temp_c", "temp_err_c")
    result = fit_curve(
        data["time_s"],
        data["temp_c"],
        "exponential",
        yerr=data["temp_err_c"],
        outlier_threshold=3.5,
    )
    # True values used by examples/make_cooling_data.py.
    for value, error, expected in zip(
        result.values, result.errors, (68.0, 600.0, 22.0), strict=True
    ):
        assert abs(value - expected) < 3.0 * error
    assert result.n_outliers == 2
    assert data["time_s"][result.outliers].tolist() == [900.0, 2640.0]
    assert 0.5 < result.reduced_chi2 < 1.5


# --- invalid input -----------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "y", "yerr", "message"),
    [
        ([[0.0, 1.0], [2.0, 3.0]], [0.0, 1.0], None, "one-dimensional"),
        ([0.0, 1.0, 2.0], [0.0, 1.0], None, "same length"),
        ([0.0, 1.0, float("nan")], [0.0, 1.0, 2.0], None, "finite"),
        ([0.0, 1.0, 2.0], [0.0, float("inf"), 2.0], None, "finite"),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.1, 0.1], "same length as y"),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.1, 0.0, 0.1], "strictly positive"),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.1, float("nan"), 0.1], "strictly positive"),
        ([0.0, 1.0], [0.0, 1.0], None, "at least 3 points"),
    ],
)
def test_invalid_inputs(x: object, y: object, yerr: object, message: str) -> None:
    with pytest.raises(DataError, match=message):
        fit_curve(x, y, "linear", yerr=yerr)


def test_exponential_needs_four_points() -> None:
    with pytest.raises(DataError, match="at least 4 points"):
        fit_curve([0.0, 1.0, 2.0], [3.0, 2.0, 1.5], "exponential")


def test_unknown_model() -> None:
    with pytest.raises(DataError, match="unknown model"):
        fit_curve([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], "parabola")


def test_invalid_max_iterations() -> None:
    with pytest.raises(DataError, match="max_iterations"):
        fit_curve([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], max_iterations=0)


def test_too_few_points_left_after_rejection() -> None:
    x = np.arange(5.0)
    y = np.array([0.0, 1.0, 2.1, 30.0, -40.0])
    with pytest.raises(DataError, match=r"only \d are usable"):
        fit_curve(x, y, "linear", outlier_threshold=0.05)


def test_failure_of_the_optimiser_is_reported_as_fit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never_converges(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Optimal parameters not found")

    monkeypatch.setattr("labcurve.fitting.curve_fit", never_converges)
    with pytest.raises(FitError, match="did not converge"):
        fit_curve(np.arange(5.0), np.arange(5.0), "linear")


def test_non_finite_parameters_are_reported_as_fit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "labcurve.fitting.curve_fit",
        lambda *args, **kwargs: (np.array([np.nan, 1.0]), np.eye(2)),
    )
    with pytest.raises(FitError, match="non-finite"):
        fit_curve(np.arange(5.0), np.arange(5.0), "linear")


def test_unknown_parameter_errors_become_nan(monkeypatch: pytest.MonkeyPatch) -> None:
    flat = Model(
        name="flat",
        formula="y = a + 0 * b",
        description="Model with a parameter the data cannot determine.",
        parameters=("a", "b"),
        function=lambda x, a, b: a + 0.0 * b + 0.0 * x,
        initial_guess=lambda x, y: (0.0, 1.0),
    )
    monkeypatch.setitem(MODELS, "flat", flat)
    result = fit_curve(np.arange(6.0), np.array([1.0, 1.2, 0.9, 1.1, 1.0, 0.8]), "flat")
    assert result.values[0] == pytest.approx(1.0)
    assert math.isnan(result.errors[1])
