from __future__ import annotations

import math

import numpy as np
import pytest

from labcurve import DataError, summarize


def test_known_values() -> None:
    summary = summarize([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert summary.n == 8
    assert summary.mean == pytest.approx(5.0)
    assert summary.std == pytest.approx(math.sqrt(32.0 / 7.0))
    assert summary.sem == pytest.approx(math.sqrt(32.0 / 7.0) / math.sqrt(8.0))
    assert summary.median == pytest.approx(4.5)
    assert (summary.minimum, summary.maximum) == (2.0, 9.0)


def test_confidence_interval_matches_the_t_table() -> None:
    # For n = 5 the two-sided 95 % quantile of Student's t (4 d.o.f.) is 2.776.
    summary = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    half_width = 2.776 * math.sqrt(2.5) / math.sqrt(5.0)
    assert summary.ci_low == pytest.approx(3.0 - half_width, abs=1e-3)
    assert summary.ci_high == pytest.approx(3.0 + half_width, abs=1e-3)


def test_higher_confidence_gives_a_wider_interval() -> None:
    sample = [1.0, 2.0, 4.0, 8.0]
    narrow, wide = summarize(sample, 0.90), summarize(sample, 0.99)
    assert wide.ci_high - wide.ci_low > narrow.ci_high - narrow.ci_low
    assert wide.confidence == 0.99


def test_interval_covers_the_true_mean_at_the_nominal_rate(rng: np.random.Generator) -> None:
    covered = 0
    for _ in range(2000):
        summary = summarize(rng.normal(10.0, 2.0, size=12))
        covered += summary.ci_low <= 10.0 <= summary.ci_high
    assert covered / 2000 == pytest.approx(0.95, abs=0.02)


def test_single_value_has_no_spread() -> None:
    summary = summarize([3.5])
    assert (summary.n, summary.mean, summary.median) == (1, 3.5, 3.5)
    assert all(math.isnan(v) for v in (summary.std, summary.sem, summary.ci_low, summary.ci_high))


def test_accepts_numpy_arrays_and_returns_plain_numbers() -> None:
    values = summarize(np.array([1, 2, 3])).as_dict()
    assert values["mean"] == 2.0
    assert all(type(value) in (int, float) for value in values.values())


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "empty"),
        ([[1.0, 2.0], [3.0, 4.0]], "one-dimensional"),
        ([1.0, float("nan")], "finite"),
        ([1.0, float("inf")], "finite"),
    ],
)
def test_invalid_samples(values: list[object], message: str) -> None:
    with pytest.raises(DataError, match=message):
        summarize(values)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 95.0])
def test_invalid_confidence(confidence: float) -> None:
    with pytest.raises(DataError, match="confidence"):
        summarize([1.0, 2.0], confidence)
