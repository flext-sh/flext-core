"""Main modernizer flow tests."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomlkit

from flext_infra import FlextInfraPyprojectModernizer, u
from flext_infra.deps.modernizer import main
from flext_tests import tm


class TestFlextInfraPyprojectModernizer:
    """Tests modernizer class behavior."""

    def test_modernizer_initialization(self) -> None:
        modernizer = FlextInfraPyprojectModernizer()
        tm.that(str(modernizer.root) != "", eq=True)

    def test_modernizer_with_custom_root(self, tmp_path: Path) -> None:
        modernizer = FlextInfraPyprojectModernizer(workspace_root=tmp_path)
        tm.that(str(modernizer.root), eq=str(tmp_path))

    def test_find_pyproject_files(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "pyproject.toml").touch()
        files = FlextInfraPyprojectModernizer(
            workspace_root=tmp_path
        ).find_pyproject_files()
        tm.that(len(files) >= 2, eq=True)

    def test_find_pyproject_files_skips_directories(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "pyproject.toml").touch()
        files = FlextInfraPyprojectModernizer(
            workspace_root=tmp_path
        ).find_pyproject_files()
        tm.that(all(".venv" not in str(path) for path in files), eq=True)

    def test_process_file_paths(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"\n')
        modernizer = FlextInfraPyprojectModernizer(workspace_root=tmp_path)
        changes = modernizer.process_file(
            pyproject, canonical_dev=[], dry_run=True, skip_comments=False
        )
        tm.that(len(changes) >= 0, eq=True)
        pyproject.write_text("invalid [[[")
        invalid = modernizer.process_file(
            pyproject, canonical_dev=[], dry_run=True, skip_comments=False
        )
        tm.that("invalid TOML" in invalid, eq=True)

    def test_process_file_dry_run_and_skip_comments(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        original = '[project]\nname = "test"\n'
        pyproject.write_text(original)
        modernizer = FlextInfraPyprojectModernizer(workspace_root=tmp_path)
        _ = modernizer.process_file(
            pyproject, canonical_dev=["pytest"], dry_run=True, skip_comments=False
        )
        tm.that(pyproject.read_text(), eq=original)
        changes = modernizer.process_file(
            pyproject, canonical_dev=[], dry_run=True, skip_comments=True
        )
        tm.that(any("banner" in item for item in changes), eq=False)

    def test_process_file_removes_empty_poetry_groups(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "test"\n[tool.poetry.group.empty.dependencies]\n'
        )
        modernizer = FlextInfraPyprojectModernizer(workspace_root=tmp_path)
        changes = modernizer.process_file(
            pyproject, canonical_dev=[], dry_run=True, skip_comments=False
        )
        tm.that(any("empty" in item for item in changes), eq=True)


class TestModernizerRunAndMain:
    """Tests run and CLI entrypoint behavior."""

    def test_run_with_audit_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"\n')
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}
        args = argparse.Namespace(
            dry_run=False, audit=True, skip_comments=False, skip_check=True
        )
        modernizer = FlextInfraPyprojectModernizer(workspace_root=tmp_path)

        def _find_files() -> list[Path]:
            return [pyproject]

        def _read_doc(_path: Path) -> tomlkit.TOMLDocument:
            return doc

        monkeypatch.setattr(modernizer, "find_pyproject_files", _find_files)
        monkeypatch.setattr(u.Infra, "read", _read_doc)
        tm.that(
            modernizer.run(args, u.Infra.CliArgs(workspace=tmp_path)) in {0, 1},
            eq=True,
        )

    def test_run_with_poetry_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"\n')
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}
        args = argparse.Namespace(
            dry_run=False, audit=False, skip_comments=False, skip_check=False
        )
        modernizer = FlextInfraPyprojectModernizer(workspace_root=tmp_path)

        def _find_files() -> list[Path]:
            return [pyproject]

        def _read_doc(_path: Path) -> tomlkit.TOMLDocument:
            return doc

        def _check(_files: list[Path]) -> int:
            return 0

        monkeypatch.setattr(modernizer, "find_pyproject_files", _find_files)
        monkeypatch.setattr(u.Infra, "read", _read_doc)
        monkeypatch.setattr(modernizer, "_run_poetry_check", _check)
        tm.that(
            modernizer.run(
                args,
                u.Infra.CliArgs(workspace=tmp_path, apply=True),
            ),
            eq=0,
        )

    def test_run_poetry_check_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'")
        modernizer = FlextInfraPyprojectModernizer(workspace_root=tmp_path)

        def _run_ok(
            _cmd: Sequence[str],
            cwd: Path | None = None,
            timeout: int | None = None,
            env: dict[str, str] | None = None,
        ) -> SimpleNamespace:
            _ = (cwd, timeout, env)
            return SimpleNamespace(is_failure=False, value=SimpleNamespace(exit_code=0))

        def _run_fail(
            _cmd: Sequence[str],
            cwd: Path | None = None,
            timeout: int | None = None,
            env: dict[str, str] | None = None,
        ) -> SimpleNamespace:
            _ = (cwd, timeout, env)
            return SimpleNamespace(is_failure=True)

        def _run_non_zero(
            _cmd: Sequence[str],
            cwd: Path | None = None,
            timeout: int | None = None,
            env: dict[str, str] | None = None,
        ) -> SimpleNamespace:
            _ = (cwd, timeout, env)
            return SimpleNamespace(is_failure=False, value=SimpleNamespace(exit_code=1))

        monkeypatch.setattr(u.Infra, "run_raw", _run_ok)
        tm.that(modernizer._run_poetry_check([pyproject]), eq=0)
        monkeypatch.setattr(u.Infra, "run_raw", _run_fail)
        tm.that(modernizer._run_poetry_check([pyproject]), eq=1)
        monkeypatch.setattr(u.Infra, "run_raw", _run_non_zero)
        tm.that(modernizer._run_poetry_check([pyproject]), eq=1)

    def test_main_cli_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _ModernizerAdapter(FlextInfraPyprojectModernizer):
            def __init__(
                self,
                root: Path | None = None,
                workspace_root: Path | None = None,
            ) -> None:
                super().__init__(workspace_root=workspace_root or root)

        def _run_zero(
            _self: FlextInfraPyprojectModernizer,
            _args: argparse.Namespace,
            _cli: u.Infra.CliArgs,
        ) -> int:
            return 0

        def _run_forty_two(
            _self: FlextInfraPyprojectModernizer,
            _args: argparse.Namespace,
            _cli: u.Infra.CliArgs,
        ) -> int:
            return 42

        monkeypatch.setattr(
            "flext_infra.deps.modernizer.FlextInfraPyprojectModernizer",
            _ModernizerAdapter,
        )
        monkeypatch.setattr("sys.argv", ["modernizer", "--dry-run"])
        monkeypatch.setattr(_ModernizerAdapter, "run", _run_zero)
        tm.that(main(), eq=0)
        monkeypatch.setattr("sys.argv", ["modernizer", "--audit"])
        tm.that(main(), eq=0)
        monkeypatch.setattr("sys.argv", ["modernizer"])
        monkeypatch.setattr(_ModernizerAdapter, "run", _run_forty_two)
        tm.that(main(), eq=42)
