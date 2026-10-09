from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labcurve import DataError, clean_measurements, load_measurements


def frame(rows: list[tuple[object, ...]], columns: tuple[str, ...] = ("t", "v")) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns))


def test_clean_data_passes_through_unchanged() -> None:
    cleaned, report = clean_measurements(frame([("0", "1.5"), ("1", "2.5")]), "t", "v")
    assert cleaned["t"].tolist() == [0.0, 1.0]
    assert cleaned["v"].tolist() == [1.5, 2.5]
    assert cleaned.dtypes.tolist() == [np.dtype(float), np.dtype(float)]
    assert report.n_removed == 0


@pytest.mark.parametrize("marker", ["", "  ", "NA", "n/a", "NaN", "null", "None", "-", "--"])
def test_missing_markers_are_counted_as_missing(marker: str) -> None:
    cleaned, report = clean_measurements(frame([("0", "1"), ("1", marker), ("2", "3")]), "t", "v")
    assert len(cleaned) == 2
    assert (report.n_missing, report.n_non_numeric) == (1, 0)


def test_real_missing_values_are_counted_as_missing() -> None:
    cleaned, report = clean_measurements(frame([(0.0, 1.0), (1.0, None), (np.nan, 3.0)]), "t", "v")
    assert len(cleaned) == 1
    assert report.n_missing == 2


@pytest.mark.parametrize("text", ["sensor error", "12,5", "1.2.3", "inf", "-inf"])
def test_text_that_is_not_a_finite_number_is_counted_as_non_numeric(text: str) -> None:
    cleaned, report = clean_measurements(frame([("0", "1"), ("1", text)]), "t", "v")
    assert len(cleaned) == 1
    assert (report.n_missing, report.n_non_numeric) == (0, 1)


def test_row_with_missing_and_non_numeric_cells_is_counted_once() -> None:
    _, report = clean_measurements(frame([("0", "1"), ("", "oops")]), "t", "v")
    assert (report.n_missing, report.n_non_numeric) == (1, 0)


def test_non_positive_uncertainties_are_removed() -> None:
    table = frame(
        [("0", "1", "0.1"), ("1", "2", "0"), ("2", "3", "-0.1"), ("3", "4", "0.2")],
        ("t", "v", "err"),
    )
    cleaned, report = clean_measurements(table, "t", "v", "err")
    assert cleaned["t"].tolist() == [0.0, 3.0]
    assert report.n_invalid_uncertainty == 2


def test_duplicates_are_removed_by_default_and_kept_on_request() -> None:
    table = frame([("0", "1"), ("1", "2"), ("1", "2"), ("1", "2.0")])
    cleaned, report = clean_measurements(table, "t", "v")
    assert len(cleaned) == 2
    assert report.n_duplicates == 2

    kept, report_kept = clean_measurements(table, "t", "v", drop_duplicates=False)
    assert len(kept) == 4
    assert report_kept.n_duplicates == 0


def test_repeated_x_with_different_y_is_not_a_duplicate() -> None:
    cleaned, report = clean_measurements(frame([("1", "2"), ("1", "3")]), "t", "v")
    assert len(cleaned) == 2
    assert report.n_duplicates == 0


def test_result_is_sorted_by_x_with_a_fresh_index() -> None:
    cleaned, _ = clean_measurements(frame([("2", "c"), ("3", "30"), ("1", "10")]), "t", "v")
    assert cleaned["t"].tolist() == [1.0, 3.0]
    assert cleaned.index.tolist() == [0, 1]


def test_input_table_is_not_modified() -> None:
    table = frame([("1", "x"), ("0", "1")])
    snapshot = table.copy()
    clean_measurements(table, "t", "v")
    pd.testing.assert_frame_equal(table, snapshot)


def test_counts_add_up_for_the_example_file(cooling_csv: Path) -> None:
    raw = load_measurements(cooling_csv, "time_s", "temp_c", "temp_err_c")
    cleaned, report = clean_measurements(raw, "time_s", "temp_c", "temp_err_c")
    assert report.as_dict() == {
        "n_input": 62,
        "n_missing": 2,
        "n_non_numeric": 1,
        "n_invalid_uncertainty": 0,
        "n_duplicates": 1,
        "n_output": 58,
    }
    assert report.n_removed == 4
    assert len(cleaned) == report.n_output
    assert cleaned["time_s"].is_monotonic_increasing


def test_nothing_left_after_cleaning() -> None:
    with pytest.raises(DataError, match="no usable rows"):
        clean_measurements(frame([("0", ""), ("x", "1")]), "t", "v")


def test_unknown_column() -> None:
    with pytest.raises(DataError, match="not found: err"):
        clean_measurements(frame([("0", "1")]), "t", "v", "err")
