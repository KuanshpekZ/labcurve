"""Reading measurement tables from CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from labcurve.errors import DataError


def load_measurements(
    path: str | Path,
    x: str,
    y: str,
    yerr: str | None = None,
    *,
    delimiter: str = ",",
    decimal: str = ".",
) -> pd.DataFrame:
    """Read the columns needed for an analysis from a CSV file.

    The values are returned exactly as they appear in the file (as text), so
    that :func:`labcurve.cleaning.clean_measurements` can report how many
    entries were missing or not numeric. Lines starting with ``#`` are treated
    as comments, which makes it possible to keep metadata in the data file.

    Parameters
    ----------
    path
        Location of the CSV file.
    x, y
        Names of the columns holding the independent and dependent variable.
    yerr
        Optional name of the column holding the uncertainty of ``y``.
    delimiter
        Field separator used in the file.
    decimal
        Character used as the decimal point. It is normalised to ``"."``.

    Returns
    -------
    pandas.DataFrame
        A table with the requested columns, in the order ``x``, ``y``, ``yerr``.

    Raises
    ------
    DataError
        If the file does not exist, cannot be parsed, has no rows, or lacks
        one of the requested columns.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise DataError(f"input file not found: {file_path}")

    columns = [x, y] if yerr is None else [x, y, yerr]
    if len(set(columns)) != len(columns):
        raise DataError("the x, y and yerr columns must be different")

    try:
        table = pd.read_csv(
            file_path,
            sep=delimiter,
            comment="#",
            dtype=str,
            keep_default_na=False,
            skipinitialspace=True,
            encoding="utf-8",
        )
    except pd.errors.EmptyDataError as exc:
        raise DataError(f"input file is empty: {file_path}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataError(f"could not parse {file_path}: {exc}") from exc

    table.columns = pd.Index([str(name).strip() for name in table.columns])
    missing = [name for name in columns if name not in table.columns]
    if missing:
        available = ", ".join(table.columns) or "(none)"
        raise DataError(
            f"column(s) not found: {', '.join(missing)}. Available columns: {available}"
        )

    selected = table[columns].copy()
    if selected.empty:
        raise DataError(f"input file has a header but no data rows: {file_path}")

    for name in columns:
        selected[name] = selected[name].str.strip()
        if decimal != ".":
            selected[name] = selected[name].str.replace(decimal, ".", regex=False)
    return selected
