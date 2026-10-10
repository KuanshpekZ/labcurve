# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-09

First public release.

### Added

- `load_measurements`: read selected columns of a CSV file, with support for
  comment lines, custom delimiters and decimal commas.
- `clean_measurements`: convert to numbers, drop missing, non-numeric,
  invalid-uncertainty and duplicate rows, and report how many rows each rule
  removed.
- `summarize`: descriptive statistics with a Student's t confidence interval
  for the mean.
- `fit_curve`: weighted non-linear least-squares fit of the `linear` and
  `exponential` models with parameter uncertainties, R², RMSE and reduced χ².
- `power` model (`y = a * x^b`) for power laws such as the period of a
  pendulum; data with non-positive `x` are rejected with a clear message.
- Optional iterative outlier rejection based on the modified z-score.
- `plot_fit`: figure of the data, the fitted curve and the residuals.
- `labcurve analyze` and `labcurve models` commands; the analysis writes a
  figure, the cleaned data and a JSON report that records the input checksum
  and the library versions.
- Synthetic, reproducible example data set (`examples/cooling.csv`).
- Test suite with 100 % line and branch coverage.
- Continuous integration (formatting, linting, type-checking, tests on
  Python 3.11 to 3.14 on Linux and Windows, oldest supported dependencies,
  package build) and automated GitHub releases from version tags.

[Unreleased]: https://github.com/KuanshpekZ/labcurve/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/KuanshpekZ/labcurve/releases/tag/v0.1.0
