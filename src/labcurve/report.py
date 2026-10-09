"""Machine-readable record of an analysis.

Besides the numbers, the report stores what is needed to reproduce them: a
checksum of the input file, the options that were used and the versions of the
libraries that did the computation.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

from labcurve.cleaning import CleaningReport
from labcurve.fitting import FitResult
from labcurve.stats import Summary

_LIBRARIES = ("labcurve", "numpy", "pandas", "scipy", "matplotlib")


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 checksum of a file as a hexadecimal string."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict[str, str]:
    """Return the versions of Python and of the libraries used."""
    versions = {"python": platform.python_version()}
    for name in _LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def build_report(
    *,
    source: str | Path,
    columns: dict[str, str | None],
    cleaning: CleaningReport,
    statistics: dict[str, Summary],
    fit: FitResult,
    outlier_threshold: float | None,
) -> dict[str, Any]:
    """Collect all results of one analysis in a JSON-serialisable dictionary."""
    return {
        "input": {
            "file": Path(source).name,
            "sha256": file_sha256(source),
            "columns": columns,
        },
        "options": {"model": fit.model, "outlier_threshold": outlier_threshold},
        "cleaning": cleaning.as_dict(),
        "statistics": {name: summary.as_dict() for name, summary in statistics.items()},
        "fit": fit.as_dict(),
        "environment": environment(),
    }


def _json_safe(value: Any) -> Any:
    """Replace ``nan`` and infinities, which JSON cannot represent, by ``None``."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write a report as strict, human-readable JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(report), indent=2, allow_nan=False)
    target.write_text(text + "\n", encoding="utf-8")
    return target
