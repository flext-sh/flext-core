"""Workspace/parser helper tests for deps modernizer."""

from __future__ import annotations

from pathlib import Path

import tomlkit

from flext_infra.deps.modernizer import parser, read_doc, workspace_root
from flext_tests import tm


class TestReadDoc:
    """Tests TOML document reading helper."""

    def testread_doc_valid_file(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('key = "value"\n')
        result = read_doc(toml_file)
        tm.that(result is None, eq=False)
        if result is not None:
            tm.that(result["key"], eq="value")

    def testread_doc_nonexistent_file(self, tmp_path: Path) -> None:
        tm.that(read_doc(tmp_path / "nonexistent.toml"), eq=None)

    def testread_doc_invalid_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "invalid.toml"
        toml_file.write_text("invalid toml content [[[")
        tm.that(read_doc(toml_file), eq=None)

    def testread_doc_permission_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[project]\nname = 'test'")
        toml_file.chmod(0)
        try:
            tm.that(read_doc(toml_file), eq=None)
        finally:
            toml_file.chmod(420)


class TestWorkspaceRoot:
    """Tests workspace root detection helper."""

    def testworkspace_root_with_gitmodules(self, tmp_path: Path) -> None:
        (tmp_path / ".gitmodules").touch()
        (tmp_path / "pyproject.toml").touch()
        result = workspace_root(tmp_path / "subdir")
        tm.that(str(result), eq=str(tmp_path))

    def testworkspace_root_with_git(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / "pyproject.toml").touch()
        result = workspace_root(tmp_path / "subdir")
        tm.that(str(result), eq=str(tmp_path))

    def testworkspace_root_fallback(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep_path.mkdir(parents=True, exist_ok=True)
        tm.that(str(workspace_root(deep_path)) != "", eq=True)


class TestParser:
    """Tests CLI parser helper."""

    def testparser_args(self) -> None:
        parser = parser()
        tm.that(parser.parse_args(["--audit"]).audit, eq=True)
        tm.that(parser.parse_args(["--dry-run"]).dry_run, eq=True)
        tm.that(parser.parse_args(["--skip-comments"]).skip_comments, eq=True)
        tm.that(parser.parse_args(["--skip-check"]).skip_check, eq=True)


def testworkspace_root_doc_construction() -> None:
    doc = tomlkit.document()
    doc["project"] = {"name": "test"}
    tm.that("project" in doc, eq=True)
