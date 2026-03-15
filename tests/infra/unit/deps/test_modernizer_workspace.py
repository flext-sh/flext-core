"""Workspace/parser helper tests for deps modernizer."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import patch

import tomlkit

from flext_infra import u
from flext_infra.deps.modernizer import FlextInfraPyprojectModernizer, main
from flext_tests import t, tm


class TestReadDoc:
    """Tests TOML document reading helper."""

    def testread_doc_valid_file(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('key = "value"\n')
        result = u.Infra.read(toml_file)
        tm.that(result is None, eq=False)
        if result is not None:
            tm.that(
                cast("t.Tests.Matcher.MatcherKwargValue", result["key"]), eq="value"
            )

    def testread_doc_nonexistent_file(self, tmp_path: Path) -> None:
        tm.that(u.Infra.read(tmp_path / "nonexistent.toml"), eq=None)

    def testread_doc_invalid_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "invalid.toml"
        toml_file.write_text("invalid toml content [[[")
        tm.that(u.Infra.read(toml_file), eq=None)

    def testread_doc_permission_error(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "test.toml"
        toml_file.write_text("[project]\nname = 'test'")
        toml_file.chmod(0)
        try:
            tm.that(u.Infra.read(toml_file), eq=None)
        finally:
            toml_file.chmod(420)


class TestWorkspaceRoot:
    """Tests workspace root detection helper."""

    def testworkspace_root_with_gitmodules(self, tmp_path: Path) -> None:
        (tmp_path / ".gitmodules").touch()
        (tmp_path / "pyproject.toml").touch()
        result = u.Infra.workspace_root(tmp_path / "subdir")
        tm.ok(result)
        tm.that(str(result.value), eq=str(tmp_path / "subdir"))

    def testworkspace_root_with_git(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        (tmp_path / "pyproject.toml").touch()
        result = u.Infra.workspace_root(tmp_path / "subdir")
        tm.ok(result)
        tm.that(str(result.value), eq=str(tmp_path / "subdir"))

    def testworkspace_root_fallback(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep_path.mkdir(parents=True, exist_ok=True)
        result = u.Infra.workspace_root(deep_path)
        tm.ok(result)
        tm.that(str(result.value) != "", eq=True)


class TestParser:
    """Tests CLI parser helper."""

    def test_parser_args(self) -> None:
        class _ModernizerAdapter(FlextInfraPyprojectModernizer):
            def __init__(
                self,
                root: Path | None = None,
                workspace_root: Path | None = None,
            ) -> None:
                super().__init__(workspace_root=workspace_root or root)

        with (
            patch(
                "flext_infra.deps.modernizer.FlextInfraPyprojectModernizer",
                _ModernizerAdapter,
            ),
            patch(
                "flext_infra.deps.modernizer.FlextInfraPyprojectModernizer.run",
                return_value=0,
            ) as run_mock,
        ):
            exit_code = main([
                "--audit",
                "--dry-run",
                "--skip-comments",
                "--skip-check",
            ])
        tm.that(exit_code, eq=0)
        tm.that(run_mock.called, eq=True)
        call_args = run_mock.call_args
        tm.that(call_args is not None, eq=True)
        if call_args is None:
            return
        args = call_args.args[0]
        tm.that(args.audit, eq=True)
        tm.that(args.dry_run, eq=True)
        tm.that(args.skip_comments, eq=True)
        tm.that(args.skip_check, eq=True)


def test_workspace_root_doc_construction() -> None:
    doc = tomlkit.document()
    doc["project"] = {"name": "test"}
    tm.that("project" in doc, eq=True)
