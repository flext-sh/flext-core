"""Tests for FlextInfraConfigFixer error handling and edge cases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path

import pytest

from flext_core import r
from flext_infra.check.services import FlextInfraConfigFixer
from flext_tests import tm


def _fake_process(
    _s: FlextInfraConfigFixer,
    _p: Path,
    *,
    dry_run: bool = False,
) -> r[list[str]]:
    del dry_run
    return r[list[str]].ok(["fix1"])


class TestConfigFixerRunMethods:
    """Test FlextInfraConfigFixer.run method with monkeypatch."""

    def test_run_with_verbose_and_fixes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.pyrefly]\nsearch-path = []\n")

        def _find(
            _s: FlextInfraConfigFixer,
            _p: list[Path] | None = None,
        ) -> r[list[Path]]:
            return r[list[Path]].ok([pyproject])

        def _proc(
            _s: FlextInfraConfigFixer,
            _p: Path,
            *,
            dry_run: bool = False,
        ) -> r[list[str]]:
            del dry_run
            return r[list[str]].ok(["fix1", "fix2"])

        monkeypatch.setattr(FlextInfraConfigFixer, "find_pyproject_files", _find)
        monkeypatch.setattr(FlextInfraConfigFixer, "process_file", _proc)
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.run(["project1"], verbose=True)
        tm.ok(result)
        tm.that(len(result.value) > 0, eq=True)

    def test_run_with_dry_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.pyrefly]\nsearch-path = []\n")

        def _find(
            _s: FlextInfraConfigFixer,
            _p: list[Path] | None = None,
        ) -> r[list[Path]]:
            return r[list[Path]].ok([pyproject])

        monkeypatch.setattr(FlextInfraConfigFixer, "find_pyproject_files", _find)
        monkeypatch.setattr(FlextInfraConfigFixer, "process_file", _fake_process)
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        tm.ok(fixer.run(["project1"], dry_run=True))


class TestProcessFileReadError:
    """Test process_file read error handling."""

    def test_read_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)

        def _raise(_self: Path, encoding: str = "utf-8") -> str:
            del encoding
            msg = "read error"
            raise OSError(msg)

        monkeypatch.setattr(Path, "read_text", _raise)
        tm.fail(fixer.process_file(tmp_path / "pyproject.toml"), has="read error")

    def test_parse_error(self, tmp_path: Path) -> None:
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("invalid toml {")
        tm.fail(fixer.process_file(pyproject), has="parse")

    def test_no_tool_section(self, tmp_path: Path) -> None:
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[build-system]\n")
        result = fixer.process_file(pyproject)
        tm.ok(result)
        tm.that(len(result.value), eq=0)

    def test_write_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.pyrefly]\nsearch-path = []\n")

        def _fake_fix(
            _s: FlextInfraConfigFixer,
            _d: MutableMapping[str, object],
            _r: Path,
        ) -> list[str]:
            return ["fix1"]

        original_write = Path.write_text

        def _raise_on_write(self_path: Path, data: str, **kw: str) -> None:
            if self_path == pyproject:
                msg = "write error"
                raise OSError(msg)
            original_write(self_path, data, **kw)

        monkeypatch.setattr(FlextInfraConfigFixer, "_fix_search_paths_tk", _fake_fix)
        monkeypatch.setattr(Path, "write_text", _raise_on_write)
        tm.fail(fixer.process_file(pyproject), has="write error")


class TestConfigFixerPathResolution:
    """Test config fixer path resolution."""

    def test_non_mutable_pyrefly_returns_empty(self, tmp_path: Path) -> None:
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[tool]\npyrefly = "string"\n')
        result = fixer.process_file(pyproject)
        tm.ok(result)
        tm.that(result.value, eq=[])

    def test_empty_projects(self, tmp_path: Path) -> None:
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        result = fixer.run(projects=[], dry_run=False, verbose=False)
        tm.ok(result)
        tm.that(result.value, eq=[])


class TestConfigFixerRunWithVerbose:
    """Test run method with verbose flag."""

    def test_verbose_with_fixes(self, tmp_path: Path) -> None:
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[tool.pyrefly]\nsearch_paths = ["src"]\nignore = true\n',
        )
        tm.ok(fixer.run(projects=[], dry_run=False, verbose=True))

    def test_empty_fixes_skips_logging(self, tmp_path: Path) -> None:
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool]\n")
        result = fixer.run(projects=[], dry_run=False, verbose=True)
        tm.ok(result)
        tm.that(result.value, eq=[])

    def test_relative_to_error(self, tmp_path: Path) -> None:
        fixer = FlextInfraConfigFixer(workspace_root=tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[tool.pyrefly]\nsearch_paths = ["src"]\nignore = true\n',
        )
        tm.ok(fixer.run(projects=[], dry_run=False, verbose=True))
