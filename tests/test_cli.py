from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from labcurve import __version__
from labcurve.cli import build_parser, main

COOLING = ["-x", "time_s", "-y", "temp_c", "--yerr", "temp_err_c", "-m", "exponential"]


def test_full_analysis_writes_all_outputs(
    tmp_path: Path, cooling_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "results"
    code = main(["analyze", str(cooling_csv), *COOLING, "--reject-outliers", "-o", str(out_dir)])
    output = capsys.readouterr().out

    assert code == 0
    assert (
        "62 read, 58 kept (2 missing, 1 not numeric, 0 invalid uncertainty, 1 duplicate)" in output
    )
    assert "2 rejected (modified z-score > 3.5)" in output
    assert "chi2/dof" in output
    # Everything except the lines that echo user-supplied paths is plain ASCII.
    fixed = [line for line in output.splitlines() if not line.startswith(("input", "written"))]
    assert all(line.isascii() for line in fixed)

    assert (out_dir / "fit.png").stat().st_size > 5000
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["fit"]["parameters"]["tau"]["value"] == pytest.approx(600.0, rel=0.03)
    assert report["options"]["outlier_threshold"] == 3.5

    cleaned = pd.read_csv(out_dir / "cleaned.csv")
    assert list(cleaned.columns) == ["time_s", "temp_c", "temp_err_c", "rejected"]
    assert len(cleaned) == 58
    assert cleaned["rejected"].sum() == 2


def test_outlier_rejection_is_off_by_default(
    tmp_path: Path, cooling_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["analyze", str(cooling_csv), "-x", "time_s", "-y", "temp_c", "-o", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 0
    assert "rejection disabled" in output
    assert "model      linear" in output
    assert "chi2/dof" not in output
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["options"]["outlier_threshold"] is None
    assert pd.read_csv(tmp_path / "cleaned.csv")["rejected"].sum() == 0


def test_custom_threshold_delimiter_and_decimal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "european.csv"
    rows = [f"{i};{str(0.5 * i + 1).replace('.', ',')}" for i in range(12)]
    source.write_text("x;y\n" + "\n".join(rows) + "\n", encoding="utf-8")
    code = main(
        [
            "analyze",
            str(source),
            "-x",
            "x",
            "-y",
            "y",
            "--delimiter",
            ";",
            "--decimal",
            ",",
            "--reject-outliers",
            "5",
            "-o",
            str(tmp_path / "out"),
            "--dpi",
            "50",
            "--xlabel",
            "x (mm)",
            "--ylabel",
            "y (N)",
            "--title",
            "Calibration",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "modified z-score > 5" in output
    assert "a        0.5" in output


def test_models_command_lists_every_model(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models"]) == 0
    output = capsys.readouterr().out
    assert "linear" in output and "y = a * x + b" in output
    assert "exponential" in output
    assert "power" in output and "y = a * x^b" in output


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["-x", "time_s", "-y", "voltage"], "column(s) not found: voltage"),
        (["-x", "time_s", "-y", "temp_c", "--reject-outliers", "-1"], "must be positive"),
    ],
)
def test_data_problems_exit_with_code_1(
    tmp_path: Path,
    cooling_csv: Path,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    code = main(["analyze", str(cooling_csv), *arguments, "-o", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: ")
    assert message in captured.err
    assert not (tmp_path / "out").exists()


def test_missing_input_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["analyze", str(tmp_path / "nope.csv"), "-x", "a", "-y", "b"]) == 1
    assert "input file not found" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["analyze"],
        ["analyze", "data.csv", "-x", "t"],
        ["analyze", "d.csv", "-x", "t", "-y", "v", "-m", "cubic"],
    ],
)
def test_usage_errors_exit_with_code_2(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(arguments)
    assert raised.value.code == 2


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0
    assert capsys.readouterr().out.strip() == f"labcurve {__version__}"


def test_parser_defaults() -> None:
    args = build_parser().parse_args(["analyze", "data.csv", "-x", "t", "-y", "v"])
    assert args.model == "linear"
    assert args.reject_outliers is None
    assert args.out_dir == Path("results")
    assert (args.delimiter, args.decimal, args.dpi) == (",", ".", 150)


def test_module_can_be_run_with_python_m(tmp_path: Path, cooling_csv: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "labcurve",
            "analyze",
            str(cooling_csv),
            *COOLING,
            "-o",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "written" in completed.stdout
    assert (tmp_path / "fit.png").is_file()


def test_dunder_main_delegates_to_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["labcurve", "models"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("labcurve", run_name="__main__")
    assert raised.value.code == 0
    assert "linear" in capsys.readouterr().out


def test_power_model_reports_unsuitable_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "data.csv"
    source.write_text("x,y\n0,0\n1,1\n2,4\n3,9\n", encoding="utf-8")
    code = main(["analyze", str(source), "-x", "x", "-y", "y", "-m", "power", "-o", str(tmp_path)])
    assert code == 1
    assert "only defined for x > 0" in capsys.readouterr().err
