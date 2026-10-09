from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from labcurve import __version__, clean_measurements, fit_curve, load_measurements, summarize
from labcurve.report import build_report, environment, file_sha256, write_report


def test_checksum_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 1000
    path.write_bytes(payload)
    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_environment_lists_the_library_versions() -> None:
    versions = environment()
    assert set(versions) == {"python", "labcurve", "numpy", "pandas", "scipy", "matplotlib"}
    assert versions["labcurve"] == __version__
    assert versions["numpy"] == np.__version__


def test_environment_tolerates_a_missing_distribution(monkeypatch) -> None:
    monkeypatch.setattr("labcurve.report._LIBRARIES", ("labcurve", "not-installed-anywhere"))
    assert environment()["not-installed-anywhere"] == "unknown"


def test_report_round_trips_through_json(tmp_path: Path, cooling_csv: Path) -> None:
    raw = load_measurements(cooling_csv, "time_s", "temp_c")
    data, cleaning = clean_measurements(raw, "time_s", "temp_c")
    fit = fit_curve(data["time_s"], data["temp_c"], "exponential", outlier_threshold=3.5)
    report = build_report(
        source=cooling_csv,
        columns={"x": "time_s", "y": "temp_c", "yerr": None},
        cleaning=cleaning,
        statistics={"temp_c": summarize(data["temp_c"])},
        fit=fit,
        outlier_threshold=3.5,
    )
    path = write_report(report, tmp_path / "out" / "report.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["input"] == {
        "file": "cooling.csv",
        "sha256": file_sha256(cooling_csv),
        "columns": {"x": "time_s", "y": "temp_c", "yerr": None},
    }
    assert loaded["options"] == {"model": "exponential", "outlier_threshold": 3.5}
    assert loaded["cleaning"]["n_output"] == len(data)
    assert loaded["statistics"]["temp_c"]["n"] == len(data)
    assert loaded["fit"]["n_outliers"] == 2
    assert loaded["fit"]["reduced_chi2"] is None
    assert loaded["environment"]["labcurve"] == __version__


def test_non_finite_numbers_are_written_as_null(tmp_path: Path) -> None:
    report = {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": (2.0, float("-inf"))}}
    path = write_report(report, tmp_path / "report.json")
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"a": None, "b": [1.0, None], "c": {"d": [2.0, None]}}
    assert text.endswith("}\n")
