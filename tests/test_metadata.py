"""The version number is written in several files; they must agree."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from labcurve import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def load_release_notes_tool():
    spec = importlib.util.spec_from_file_location(
        "release_notes", REPO_ROOT / "tools" / "release_notes.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_is_a_semantic_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_citation_file_has_the_same_version() -> None:
    match = re.search(r"^version: (\S+)$", read("CITATION.cff"), re.MULTILINE)
    assert match is not None
    assert match.group(1) == __version__


def test_changelog_has_a_dated_section_for_the_version() -> None:
    pattern = rf"^## \[{re.escape(__version__)}\] - \d{{4}}-\d{{2}}-\d{{2}}$"
    assert re.search(pattern, read("CHANGELOG.md"), re.MULTILINE)


def test_release_notes_can_be_extracted_for_the_version() -> None:
    tool = load_release_notes_tool()
    notes = tool.extract(read("CHANGELOG.md"), __version__)
    assert notes.startswith("First public release.") or "###" in notes
    assert "## [" not in notes
    assert not re.search(r"^\[[^\]]+\]: ", notes, re.MULTILINE)


def test_release_notes_tool_rejects_unknown_and_empty_versions() -> None:
    tool = load_release_notes_tool()
    text = "# Changelog\n\n## [Unreleased]\n\n## [1.0.0] - 2026-01-01\n\n- Something.\n"
    assert tool.extract(text, "1.0.0") == "- Something."
    with pytest.raises(ValueError, match="no section"):
        tool.extract(text, "2.0.0")
    with pytest.raises(ValueError, match="empty"):
        tool.extract(text, "Unreleased")


def test_release_notes_tool_command_line(capsys: pytest.CaptureFixture[str]) -> None:
    tool = load_release_notes_tool()
    assert tool.main(["release_notes.py", __version__]) == 0
    assert capsys.readouterr().out.strip()
    assert tool.main(["release_notes.py", "99.0.0"]) == 1
    assert "no section" in capsys.readouterr().err
    assert tool.main(["release_notes.py"]) == 2
