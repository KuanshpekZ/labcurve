"""Print the section of CHANGELOG.md that describes one version.

Used by the release workflow to fill in the notes of a GitHub release:

    python tools/release_notes.py 0.1.0
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"


def extract(text: str, version: str) -> str:
    """Return the body of the ``## [version]`` section of a changelog.

    Raises
    ------
    ValueError
        If the changelog has no section for this version or it is empty.
    """
    heading = re.compile(rf"^## \[{re.escape(version)}\].*$", re.MULTILINE)
    match = heading.search(text)
    if match is None:
        raise ValueError(f"CHANGELOG.md has no section for version {version}")
    rest = text[match.end() :]
    # The section ends at the next version heading or at the link definitions.
    end = re.search(r"^(## \[|\[[^\]]+\]: )", rest, re.MULTILINE)
    body = (rest[: end.start()] if end else rest).strip()
    if not body:
        raise ValueError(f"the changelog section for version {version} is empty")
    return body


def main(argv: list[str]) -> int:
    """Print the release notes for the version given on the command line."""
    if len(argv) != 2:
        print("usage: release_notes.py VERSION", file=sys.stderr)
        return 2
    try:
        print(extract(CHANGELOG.read_text(encoding="utf-8"), argv[1]))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
