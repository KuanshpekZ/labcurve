# Contributing to labcurve

Thank you for considering a contribution. This document describes how the
project is developed, so that changes stay easy to review and `main` stays
releasable.

## Setting up

```bash
git clone https://github.com/KuanshpekZ/labcurve.git
cd labcurve
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Workflow

1. **Open or pick an issue.** Describe the problem or the feature before
   writing code; use the issue templates.
2. **Create a branch from `main`:**

   | Prefix     | Use for                         | Example                     |
   |------------|---------------------------------|-----------------------------|
   | `feature/` | new functionality               | `feature/power-law-model`   |
   | `fix/`     | bug fixes                       | `fix/decimal-comma-parsing` |
   | `docs/`    | documentation only              | `docs/api-examples`         |
   | `ci/`      | pipeline and tooling            | `ci/add-macos-runner`       |

3. **Commit in small, self-contained steps** using
   [Conventional Commits](https://www.conventionalcommits.org/):

   ```text
   feat(fitting): add iterative outlier rejection
   fix(io): report a clear error for an empty file
   docs: explain the modified z-score
   test(stats): cover the single-value case
   ci: test on Python 3.14
   ```

4. **Run the checks locally** — the same ones CI runs:

   ```bash
   ruff format .
   ruff check .
   mypy
   python -m pytest --cov
   ```

5. **Open a pull request** against `main`. Fill in the template, link the
   issue with `Closes #N`, and wait for CI. A pull request is merged only when
   every check is green.

## Coding standards

- Public functions have type hints and NumPy-style docstrings that state
  units, assumptions and the errors raised.
- Problems caused by user data raise `DataError` or `FitError` with a message
  that says what to fix; they never surface as a bare traceback from the
  command line.
- No measured value is ever modified silently. Cleaning steps drop rows and
  count them.
- Numerical results are tested against known answers with explicit tolerances
  (`pytest.approx`, `numpy.testing.assert_allclose`), never with `==` on
  computed floats.
- Tests that need random numbers use the seeded `rng` fixture.
- New behaviour comes with tests; total coverage must stay above 90 %.
- User-visible changes are added to `CHANGELOG.md`.

## Adding a model

1. Write the function and its starting-value rule in `src/labcurve/models.py`
   and register a `Model` in `MODELS`.
2. Add tests in `tests/test_models.py` (values, starting guess) and
   `tests/test_fitting.py` (recovery of known parameters from noisy data).
3. Add a row to the model table in `README.md` and an entry to the changelog.

## Releasing

1. Set the new version in `src/labcurve/__init__.py` and `CITATION.cff`, and
   move the entries of the changelog under a heading for that version
   (`tests/test_metadata.py` checks that the three agree).
2. Merge to `main` through a pull request.
3. Tag and push:

   ```bash
   git tag -a v0.2.0 -m "labcurve 0.2.0"
   git push origin v0.2.0
   ```

The release workflow tests, builds and publishes the GitHub release.
