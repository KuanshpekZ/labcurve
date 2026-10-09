from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from labcurve import DataError, fit_curve, plot_fit, save_figure
from labcurve.plotting import format_parameter

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 10.0, 30)
    y = 2.0 * x + 1.0 + np.sin(7.0 * x) * 0.1
    y[4] += 12.0
    return x, y


def test_figure_has_a_data_panel_and_a_residual_panel(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    figure = plot_fit(x, y, fit_curve(x, y), xlabel="time (s)", ylabel="signal (V)", title="Run 1")
    top, bottom = figure.axes
    assert isinstance(figure, Figure)
    assert top.get_ylabel() == "signal (V)"
    assert bottom.get_xlabel() == "time (s)"
    assert bottom.get_ylabel() == "residual"
    assert figure.get_suptitle() == "Run 1"
    assert [text.get_text() for text in top.get_legend().get_texts()] == ["data", "fit"]


def test_title_defaults_to_the_formula_and_reports_the_parameters(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    x, y = data
    result = fit_curve(x, y, yerr=np.full(x.size, 0.1))
    figure = plot_fit(x, y, result, yerr=np.full(x.size, 0.1))
    assert figure.get_suptitle() == "y = a * x + b"
    subtitle = figure.axes[0].get_title(loc="left")
    assert "a = " in subtitle and "b = " in subtitle
    assert "R² = " in subtitle and "χ²/dof = " in subtitle


def test_rejected_points_get_their_own_legend_entry(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    result = fit_curve(x, y, outlier_threshold=3.5)
    assert result.n_outliers == 1
    figure = plot_fit(x, y, result)
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert labels == ["data", "fit", "rejected (1)"]


def test_fitted_curve_spans_the_data_range(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    result = fit_curve(x, y)
    curve = next(line for line in plot_fit(x, y, result).axes[0].lines if line.get_label() == "fit")
    xs, ys = curve.get_data()
    assert (xs[0], xs[-1]) == (0.0, 10.0)
    np.testing.assert_allclose(ys, result.predict(xs))


@pytest.mark.parametrize("suffix", [".png", ".pdf", ".svg"])
def test_save_figure_creates_folders_and_writes_the_format(
    tmp_path: Path, data: tuple[np.ndarray, np.ndarray], suffix: str
) -> None:
    x, y = data
    target = tmp_path / "nested" / "folder" / f"fit{suffix}"
    written = save_figure(plot_fit(x, y, fit_curve(x, y)), target, dpi=60)
    assert written == target
    assert target.stat().st_size > 1000
    if suffix == ".png":
        assert target.read_bytes().startswith(PNG_SIGNATURE)


def test_plotting_does_not_touch_pyplot_state(data: tuple[np.ndarray, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    x, y = data
    before = plt.get_fignums()
    plot_fit(x, y, fit_curve(x, y))
    assert plt.get_fignums() == before


def test_points_must_match_the_fit(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = data
    result = fit_curve(x, y)
    with pytest.raises(DataError, match="same points"):
        plot_fit(x[:-1], y[:-1], result)
    with pytest.raises(DataError, match="same length as y"):
        plot_fit(x, y, result, yerr=np.ones(3))


@pytest.mark.parametrize(
    ("value", "error", "expected"),
    [
        (68.3421, 0.2919, "68.34 ± 0.29"),
        (597.953, 5.274, "598.0 ± 5.3"),
        (615.2, 12.4, "615 ± 12"),
        (12345.6, 250.0, "12346 ± 250"),
        (0.0012345, 0.000021, "0.001234 ± 0.000021"),
        (3.14159, float("nan"), "3.1416"),
        (3.14159, 0.0, "3.1416"),
        (1.0, 1e-15, "1 ± 1e-15"),
    ],
)
def test_format_parameter(value: float, error: float, expected: str) -> None:
    assert format_parameter(value, error) == expected


def test_format_parameter_with_ascii_separator() -> None:
    assert format_parameter(2.5, 0.25, pm="+/-") == "2.50 +/- 0.25"
