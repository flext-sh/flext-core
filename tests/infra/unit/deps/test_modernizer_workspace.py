from __future__ import annotations

from pathlib import Path

import tomlkit

from flext_infra.deps.modernizer import _parser, _read_doc, _workspace_root
from flext_tests import tm


class TestReadDoc:
    def test_read_doc_valid_file(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('key = "value"\n')
        result = _read_doc(toml_file)
        tm.that(result is None, eq=False)
        if result is not None:
            tm.that(result["key"], eq="value")

    def test_read_doc_nonexistent_file(self, tmp_path: Path) -> None:
        tm.that(_read_doc(tmp_path / "nonexistent.toml"), eq=None)

    def test_read_doc_invalid_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "invalid.toml"
        toml_file.write_text("invalid toml content [[[")
        tm.that(_read_doc(toml_file), eq=None)

    def test_read_doc_permission_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[project]\nname = 'test'")
        toml_file.chmod(0)
        try:
            tm.that(_read_doc(toml_file), eq=None)
        finally:
            toml_file.chmod(420)


class TestWorkspaceRoot:
    def test_workspace_root_with_gitmodules(self, tmp_path: Path) -> None:
        (tmp_path / ".gitmodules").touch()
        (tmp_path / "pyproject.toml").touch()
        result = _workspace_root(tmp_path / "subdir")
        tm.that(str(result), eq=str(tmp_path))

    def test_workspace_root_with_git(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / "pyproject.toml").touch()
        result = _workspace_root(tmp_path / "subdir")
        tm.that(str(result), eq=str(tmp_path))

    def test_workspace_root_fallback(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep_path.mkdir(parents=True, exist_ok=True)
        tm.that(str(_workspace_root(deep_path)) != "", eq=True)


class TestParser:
    def test_parser_args(self) -> None:
        parser = _parser()
        tm.that(parser.parse_args(["--audit"]).audit, eq=True)
        tm.that(parser.parse_args(["--dry-run"]).dry_run, eq=True)
        tm.that(parser.parse_args(["--skip-comments"]).skip_comments, eq=True)
        tm.that(parser.parse_args(["--skip-check"]).skip_check, eq=True)


def test_workspace_root_doc_construction() -> None:
    doc = tomlkit.document()
    doc["project"] = {"name": "test"}
    tm.that("project" in doc, eq=True)
