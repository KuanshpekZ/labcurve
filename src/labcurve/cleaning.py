"""Turning a raw measurement table into numeric data that can be analysed."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from labcurve.errors import DataError

#: Text values that mean "no measurement was recorded".
MISSING_MARKERS = frozenset({"", "na", "n/a", "nan", "null", "none", "-", "--"})


@dataclass(frozen=True)
class CleaningReport:
    """Number of rows removed at each cleaning step.

    Every input row is counted exactly once, so
    ``n_input == n_missing + n_non_numeric + n_invalid_uncertainty
    + n_duplicates + n_output``.
    """

    n_input: int
    n_missing: int
    n_non_numeric: int
    n_invalid_uncertainty: int
    n_duplicates: int
    n_output: int

    @property
    def n_removed(self) -> int:
        """Total number of rows that were discarded."""
        return self.n_input - self.n_output

    def as_dict(self) -> dict[str, int]:
        """Return the counts as a plain dictionary."""
        return asdict(self)


def clean_measurements(
    table: pd.DataFrame,
    x: str,
    y: str,
    yerr: str | None = None,
    *,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Convert measurements to numbers and discard unusable rows.

    Rows are removed, in this order, when they

    1. contain a missing value (empty cell or a marker such as ``n/a``);
    2. contain text that is not a finite number;
    3. have an uncertainty that is not strictly positive;
    4. exactly repeat an earlier row (only if ``drop_duplicates`` is true).

    The remaining rows are sorted by ``x``. Nothing is modified in place and
    no value is ever altered or imputed: a row is either kept as measured or
    dropped and counted in the report.

    Parameters
    ----------
    table
        Raw table, typically from :func:`labcurve.io.load_measurements`.
    x, y
        Names of the columns holding the independent and dependent variable.
    yerr
        Optional name of the column holding the uncertainty of ``y``.
    drop_duplicates
        Whether exact duplicate rows are removed.

    Returns
    -------
    cleaned : pandas.DataFrame
        Table of ``float`` columns, sorted by ``x``, with a fresh index.
    report : CleaningReport
        How many rows were removed and why.

    Raises
    ------
    DataError
        If a requested column is absent or no usable row remains.
    """
    columns = [x, y] if yerr is None else [x, y, yerr]
    absent = [name for name in columns if name not in table.columns]
    if absent:
        raise DataError(f"column(s) not found: {', '.join(absent)}")

    raw = table[columns]
    n_input = len(raw)

    as_text = raw.astype("string").apply(lambda column: column.str.strip().str.lower())
    is_missing = (raw.isna() | as_text.isin(MISSING_MARKERS)).to_numpy()

    numeric = raw.apply(pd.to_numeric, errors="coerce").astype(float)
    is_bad_number = ~np.isfinite(numeric.to_numpy()) & ~is_missing

    row_missing = is_missing.any(axis=1)
    row_non_numeric = is_bad_number.any(axis=1) & ~row_missing
    row_ok = ~row_missing & ~row_non_numeric

    row_bad_uncertainty = np.zeros(n_input, dtype=bool)
    if yerr is not None:
        row_bad_uncertainty = row_ok & ~(numeric[yerr].to_numpy() > 0)
        row_ok &= ~row_bad_uncertainty

    cleaned = numeric.loc[row_ok]
    n_before_duplicates = len(cleaned)
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()
    n_duplicates = n_before_duplicates - len(cleaned)

    cleaned = cleaned.sort_values(x, kind="stable").reset_index(drop=True)
    report = CleaningReport(
        n_input=n_input,
        n_missing=int(row_missing.sum()),
        n_non_numeric=int(row_non_numeric.sum()),
        n_invalid_uncertainty=int(row_bad_uncertainty.sum()),
        n_duplicates=n_duplicates,
        n_output=len(cleaned),
    )
    if cleaned.empty:
        raise DataError("no usable rows remain after cleaning")
    return cleaned, report
