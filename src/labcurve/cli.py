"""Command-line interface: ``labcurve analyze`` and ``labcurve models``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from labcurve import __version__
from labcurve.cleaning import clean_measurements
from labcurve.errors import LabcurveError
from labcurve.fitting import DEFAULT_OUTLIER_THRESHOLD, fit_curve
from labcurve.io import load_measurements
from labcurve.models import MODELS
from labcurve.plotting import format_parameter, plot_fit, save_figure
from labcurve.report import build_report, write_report
from labcurve.stats import summarize


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser of the ``labcurve`` command."""
    parser = argparse.ArgumentParser(
        prog="labcurve",
        description="Clean, summarise, fit and plot experimental measurements from a CSV file.",
    )
    parser.add_argument("--version", action="version", version=f"labcurve {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    analyze = commands.add_parser(
        "analyze",
        help="run the full analysis on a CSV file",
        description="Clean the data, fit a model and write a figure, a report and the clean data.",
    )
    analyze.add_argument("input", type=Path, help="CSV file with the measurements")
    analyze.add_argument("-x", "--x", required=True, metavar="COLUMN", help="independent variable")
    analyze.add_argument("-y", "--y", required=True, metavar="COLUMN", help="dependent variable")
    analyze.add_argument("--yerr", metavar="COLUMN", help="uncertainty of the dependent variable")
    analyze.add_argument(
        "-m",
        "--model",
        choices=sorted(MODELS),
        default="linear",
        help="model to fit (default: %(default)s)",
    )
    analyze.add_argument(
        "--reject-outliers",
        nargs="?",
        type=float,
        const=DEFAULT_OUTLIER_THRESHOLD,
        default=None,
        metavar="THRESHOLD",
        help=(
            "reject points whose residual has a modified z-score above THRESHOLD "
            f"(default threshold: {DEFAULT_OUTLIER_THRESHOLD}); off unless requested"
        ),
    )
    analyze.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="folder for the output files (default: %(default)s)",
    )
    analyze.add_argument("--delimiter", default=",", help="field separator (default: ',')")
    analyze.add_argument("--decimal", default=".", help="decimal separator (default: '.')")
    analyze.add_argument("--xlabel", help="x-axis label (default: name of the x column)")
    analyze.add_argument("--ylabel", help="y-axis label (default: name of the y column)")
    analyze.add_argument("--title", help="figure title (default: formula of the model)")
    analyze.add_argument("--dpi", type=int, default=150, help="figure resolution (default: 150)")

    commands.add_parser("models", help="list the available models")
    return parser


def _run_models() -> int:
    for model in MODELS.values():
        print(f"{model.name:<12} {model.formula}")
        print(f"{'':<12} {model.description}")
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    raw = load_measurements(
        args.input, args.x, args.y, args.yerr, delimiter=args.delimiter, decimal=args.decimal
    )
    data, cleaning = clean_measurements(raw, args.x, args.y, args.yerr)
    x = data[args.x].to_numpy()
    y = data[args.y].to_numpy()
    yerr = None if args.yerr is None else data[args.yerr].to_numpy()

    fit = fit_curve(x, y, args.model, yerr=yerr, outlier_threshold=args.reject_outliers)
    statistics = {args.x: summarize(x), args.y: summarize(y)}
    report = build_report(
        source=args.input,
        columns={"x": args.x, "y": args.y, "yerr": args.yerr},
        cleaning=cleaning,
        statistics=statistics,
        fit=fit,
        outlier_threshold=args.reject_outliers,
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figure = plot_fit(
        x,
        y,
        fit,
        yerr=yerr,
        xlabel=args.xlabel or args.x,
        ylabel=args.ylabel or args.y,
        title=args.title,
    )
    figure_path = save_figure(figure, out_dir / "fit.png", dpi=args.dpi)
    report_path = write_report(report, out_dir / "report.json")
    data_path = out_dir / "cleaned.csv"
    data.assign(rejected=fit.outliers.astype(int)).to_csv(data_path, index=False)

    # Plain ASCII output, so that it prints on every console encoding.
    print(f"input      {args.input}")
    print(
        f"rows       {cleaning.n_input} read, {cleaning.n_output} kept "
        f"({cleaning.n_missing} missing, {cleaning.n_non_numeric} not numeric, "
        f"{cleaning.n_invalid_uncertainty} invalid uncertainty, "
        f"{cleaning.n_duplicates} duplicate)"
    )
    print(f"model      {fit.model}: {fit.formula}")
    if args.reject_outliers is None:
        print("outliers   rejection disabled")
    else:
        print(f"outliers   {fit.n_outliers} rejected (modified z-score > {args.reject_outliers:g})")
    for name, value, error in zip(fit.parameters, fit.values, fit.errors, strict=True):
        print(f"  {name:<8} {format_parameter(value, error, pm='+/-')}")
    print(f"R^2        {fit.r_squared:.5f}")
    print(f"RMSE       {fit.rmse:.4g}")
    if fit.reduced_chi2 is not None:
        print(f"chi2/dof   {fit.reduced_chi2:.3f}")
    print(f"written    {figure_path}, {report_path}, {data_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return the process exit code.

    Problems with the input data or the fit are reported on standard error
    with exit code 1; invalid command-line usage exits with code 2.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "models":
            return _run_models()
        return _run_analyze(args)
    except LabcurveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
