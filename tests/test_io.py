from __future__ import annotations

from pathlib import Path

import pytest

from labcurve import DataError, load_measurements


def write(tmp_path: Path, text: str, name: str = "data.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_reads_requested_columns_as_text(tmp_path: Path) -> None:
    path = write(tmp_path, "t,v,extra\n0,1.5,a\n1,2.5,b\n")
    table = load_measurements(path, "t", "v")
    assert list(table.columns) == ["t", "v"]
    assert table["v"].tolist() == ["1.5", "2.5"]


def test_column_order_follows_the_arguments(tmp_path: Path) -> None:
    path = write(tmp_path, "err,v,t\n0.1,1.5,0\n")
    table = load_measurements(path, "t", "v", "err")
    assert list(table.columns) == ["t", "v", "err"]


def test_comment_lines_and_surrounding_spaces_are_ignored(tmp_path: Path) -> None:
    path = write(tmp_path, "# recorded on bench 2\n t , v \n 0 , 1.5 \n# note\n1,2.5\n")
    table = load_measurements(path, "t", "v")
    assert table["t"].tolist() == ["0", "1"]
    assert table["v"].tolist() == ["1.5", "2.5"]


def test_semicolon_delimiter_and_decimal_comma(tmp_path: Path) -> None:
    path = write(tmp_path, "t;v\n0;1,5\n1;2,5\n")
    table = load_measurements(path, "t", "v", delimiter=";", decimal=",")
    assert table["v"].tolist() == ["1.5", "2.5"]


def test_empty_cells_are_kept_for_the_cleaning_step(tmp_path: Path) -> None:
    path = write(tmp_path, "t,v\n0,\n1,2.5\n")
    table = load_measurements(path, "t", "v")
    assert table["v"].tolist() == ["", "2.5"]


def test_example_file_loads(cooling_csv: Path) -> None:
    table = load_measurements(cooling_csv, "time_s", "temp_c", "temp_err_c")
    assert len(table) == 62


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        load_measurements(tmp_path / "absent.csv", "t", "v")


def test_directory_is_not_a_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        load_measurements(tmp_path, "t", "v")


def test_empty_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="empty"):
        load_measurements(write(tmp_path, ""), "t", "v")


def test_header_without_rows(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="no data rows"):
        load_measurements(write(tmp_path, "t,v\n"), "t", "v")


def test_unknown_column_lists_the_available_ones(tmp_path: Path) -> None:
    path = write(tmp_path, "t,v\n0,1\n")
    with pytest.raises(DataError, match=r"not found: voltage.*Available columns: t, v"):
        load_measurements(path, "t", "voltage")


def test_repeated_column_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "t,v\n0,1\n")
    with pytest.raises(DataError, match="must be different"):
        load_measurements(path, "t", "t")


def test_malformed_file(tmp_path: Path) -> None:
    path = write(tmp_path, 't,v\n0,1\n1,"unterminated\n')
    with pytest.raises(DataError, match="could not parse"):
        load_measurements(path, "t", "v")


def test_non_utf8_file(tmp_path: Path) -> None:
    path = tmp_path / "latin.csv"
    path.write_bytes("t,v\n0,caf\xe9\n".encode("latin-1"))
    with pytest.raises(DataError, match="could not parse"):
        load_measurements(path, "t", "v")
