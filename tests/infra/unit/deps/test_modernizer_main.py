from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import tomlkit

import flext_infra.deps.modernizer as modernizer_module
from flext_infra.deps.modernizer import FlextInfraPyprojectModernizer, main
from flext_tests import tm
from tests.infra import h


class _StubRunner:
    def __init__(self, is_failure: bool, exit_code: int = 0) -> None:
        self._is_failure = is_failure
        self._exit_code = exit_code

    def run_raw(self, _cmd: list[str], cwd: Path) -> SimpleNamespace:
        h.assert_dir_exists(cwd)
        if self._is_failure:
            return SimpleNamespace(is_failure=True)
        return SimpleNamespace(
            is_failure=False,
            value=SimpleNamespace(exit_code=self._exit_code),
        )


class TestFlextInfraPyprojectModernizer:
    def test_modernizer_initialization(self) -> None:
        modernizer = FlextInfraPyprojectModernizer()
        tm.that(modernizer is not None, eq=True)
        tm.that(modernizer.root is not None, eq=True)

    def test_modernizer_with_custom_root(self, tmp_path: Path) -> None:
        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        tm.that(modernizer.root, eq=tmp_path)

    def test_find_pyproject_files(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "pyproject.toml").touch()
        files = FlextInfraPyprojectModernizer(root=tmp_path).find_pyproject_files()
        tm.that(len(files) >= 2, eq=True)

    def test_find_pyproject_files_skips_directories(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "pyproject.toml").touch()
        files = FlextInfraPyprojectModernizer(root=tmp_path).find_pyproject_files()
        tm.that(all(".venv" not in str(path) for path in files), eq=True)

    def test_process_file_paths(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"\n')
        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        changes = modernizer.process_file(
            pyproject, canonical_dev=[], dry_run=True, skip_comments=False
        )
        tm.that(isinstance(changes, list), eq=True)
        pyproject.write_text("invalid [[[")
        invalid = modernizer.process_file(
            pyproject, canonical_dev=[], dry_run=True, skip_comments=False
        )
        tm.that("invalid TOML" in invalid, eq=True)

    def test_process_file_dry_run_and_skip_comments(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        original = '[project]\nname = "test"\n'
        pyproject.write_text(original)
        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        _ = modernizer.process_file(
            pyproject, canonical_dev=["pytest"], dry_run=True, skip_comments=False
        )
        tm.that(pyproject.read_text(), eq=original)
        changes = modernizer.process_file(
            pyproject, canonical_dev=[], dry_run=True, skip_comments=True
        )
        tm.that(any("banner" in c for c in changes), eq=False)

    def test_process_file_removes_empty_poetry_groups(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "test"\n[tool.poetry.group.empty.dependencies]\n'
        )
        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        changes = modernizer.process_file(
            pyproject, canonical_dev=[], dry_run=True, skip_comments=False
        )
        tm.that(any("empty" in c for c in changes), eq=True)


class TestModernizerRunAndMain:
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
        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        monkeypatch.setattr(modernizer, "find_pyproject_files", lambda: [pyproject])
        monkeypatch.setattr(modernizer_module, "_read_doc", lambda _path: doc)
        tm.that(modernizer.run(args) in {0, 1}, eq=True)

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
        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        monkeypatch.setattr(modernizer, "find_pyproject_files", lambda: [pyproject])
        monkeypatch.setattr(modernizer_module, "_read_doc", lambda _path: doc)
        monkeypatch.setattr(modernizer, "_run_poetry_check", lambda _files: 0)
        tm.that(modernizer.run(args), eq=0)

    def test_run_poetry_check_paths(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'")
        modernizer = FlextInfraPyprojectModernizer(root=tmp_path)
        modernizer._runner = _StubRunner(is_failure=False, exit_code=0)
        tm.that(modernizer._run_poetry_check([pyproject]), eq=0)
        modernizer._runner = _StubRunner(is_failure=True)
        tm.that(modernizer._run_poetry_check([pyproject]), eq=1)
        modernizer._runner = _StubRunner(is_failure=False, exit_code=1)
        tm.that(modernizer._run_poetry_check([pyproject]), eq=1)

    def test_main_cli_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["modernizer", "--dry-run"])
        monkeypatch.setattr(
            FlextInfraPyprojectModernizer, "run", lambda _self, _args: 0
        )
        tm.that(main(), eq=0)
        monkeypatch.setattr("sys.argv", ["modernizer", "--audit"])
        tm.that(main(), eq=0)
        monkeypatch.setattr("sys.argv", ["modernizer"])
        monkeypatch.setattr(
            FlextInfraPyprojectModernizer, "run", lambda _self, _args: 42
        )
        tm.that(main(), eq=42)
