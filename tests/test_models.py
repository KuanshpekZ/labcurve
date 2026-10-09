from __future__ import annotations

import numpy as np
import pytest

from labcurve import MODELS, DataError, get_model


def test_registry_is_consistent() -> None:
    for name, model in MODELS.items():
        assert model.name == name
        assert model.formula.startswith("y = ")
        assert model.description
        assert len(set(model.parameters)) == len(model.parameters)


def test_unknown_model_lists_the_alternatives() -> None:
    with pytest.raises(DataError, match=r"unknown model 'cubic'.*exponential, linear, power"):
        get_model("cubic")


def test_linear_values() -> None:
    x = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(get_model("linear").function(x, 2.0, -1.0), [-1.0, 1.0, 3.0])


def test_exponential_values() -> None:
    x = np.array([0.0, 5.0, 1e9])
    expected = [5.0, 3.0 * np.exp(-1.0) + 2.0, 2.0]
    np.testing.assert_allclose(get_model("exponential").function(x, 3.0, 5.0, 2.0), expected)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_initial_guess_has_one_finite_value_per_parameter(name: str) -> None:
    model = get_model(name)
    x = np.linspace(1.0, 10.0, 25)
    for y in (3.0 * x + 1.0, 50.0 * np.exp(-x / 2.0) + 4.0, np.full_like(x, 7.0)):
        guess = model.initial_guess(x, y)
        assert len(guess) == len(model.parameters)
        assert np.all(np.isfinite(guess))


def test_linear_guess_is_exact_for_noise_free_data() -> None:
    x = np.array([3.0, 1.0, 2.0])
    assert get_model("linear").initial_guess(x, 4.0 * x - 2.0) == pytest.approx((4.0, -2.0))


def test_linear_guess_survives_identical_x() -> None:
    guess = get_model("linear").initial_guess(np.array([2.0, 2.0]), np.array([1.0, 3.0]))
    assert guess == (0.0, 2.0)


@pytest.mark.parametrize(("amplitude", "offset"), [(40.0, 20.0), (-40.0, 60.0)])
def test_exponential_guess_is_close_for_decay_and_saturating_growth(
    amplitude: float, offset: float
) -> None:
    x = np.linspace(0.0, 100.0, 201)
    y = amplitude * np.exp(-x / 12.0) + offset
    a, tau, c = get_model("exponential").initial_guess(x[::-1], y[::-1])
    assert a == pytest.approx(amplitude, rel=0.05)
    assert tau == pytest.approx(12.0, rel=0.2)
    assert c == pytest.approx(offset, abs=0.5)
    assert tau > 0


def test_exponential_guess_survives_repeated_first_x() -> None:
    x = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    y = np.array([10.0, 1.0, 1.0, 1.0, 1.0])
    assert get_model("exponential").initial_guess(x, y) == pytest.approx((9.0, 1.0, 1.0))


def test_power_values() -> None:
    x = np.array([1.0, 4.0, 9.0])
    np.testing.assert_allclose(get_model("power").function(x, 2.0, 0.5), [2.0, 4.0, 6.0])


@pytest.mark.parametrize("scale", [3.0, -3.0])
def test_power_guess_is_exact_for_noise_free_data(scale: float) -> None:
    x = np.linspace(0.5, 20.0, 15)
    assert get_model("power").initial_guess(x, scale * x**1.7) == pytest.approx((scale, 1.7))


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([1.0, 2.0, 3.0], [1.0, -2.0, 3.0]),  # y changes sign
        ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]),  # x is not strictly positive
        ([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]),  # no spread in x
    ],
)
def test_power_guess_falls_back_when_logarithms_are_impossible(
    x: list[float], y: list[float]
) -> None:
    assert get_model("power").initial_guess(np.array(x), np.array(y)) == (1.0, 1.0)


def test_only_the_power_model_restricts_x() -> None:
    assert [name for name, model in MODELS.items() if model.requires_positive_x] == ["power"]
