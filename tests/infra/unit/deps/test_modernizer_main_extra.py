from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import tomlkit

import flext_infra.deps.modernizer as modernizer_module
from flext_infra.deps.modernizer import FlextInfraPyprojectModernizer
from flext_tests import tm
from tests.infra import h


class TestModernizerEdgeCases:
    def test_modernizer_with_empty_pyproject(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("")
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        args = argparse.Namespace(
            project=None,
            dry_run=True,
            verbose=False,
            audit=False,
            skip_comments=False,
            skip_check=True,
        )
        tm.that(isinstance(modernizer.run(args), int), eq=True)

    def test_modernizer_with_invalid_toml(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[invalid toml {")
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        args = argparse.Namespace(
            project=None,
            dry_run=True,
            verbose=False,
            audit=False,
            skip_comments=False,
            skip_check=True,
        )
        tm.that(isinstance(modernizer.run(args), int), eq=True)

    def test_modernizer_with_missing_pyproject(self, tmp_path: Path) -> None:
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        args = argparse.Namespace(
            project=None,
            dry_run=True,
            verbose=False,
            audit=False,
            skip_comments=False,
            skip_check=True,
        )
        tm.that(isinstance(modernizer.run(args), int), eq=True)


class TestModernizerUncoveredLines:
    def test_run_with_missing_root_pyproject(self, tmp_path: Path) -> None:
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        args = argparse.Namespace(
            project=None,
            dry_run=True,
            verbose=False,
            audit=False,
            skip_comments=False,
            skip_check=True,
        )
        tm.that(modernizer.run(args), eq=2)

    def test_run_with_no_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"\n')
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        args = argparse.Namespace(
            project=None,
            dry_run=True,
            verbose=False,
            audit=False,
            skip_comments=False,
            skip_check=True,
        )
        monkeypatch.setattr(modernizer, "find_pyproject_files", lambda: [pyproject])
        monkeypatch.setattr(modernizer_module, "_read_doc", lambda _path: doc)
        monkeypatch.setattr(modernizer, "process_file", lambda *_args, **_kwargs: [])
        tm.that(modernizer.run(args), eq=0)


def test_flext_infra_pyproject_modernizer_process_file_invalid_toml(
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("invalid toml {", encoding="utf-8")
    modernizer = FlextInfraPyprojectModernizer(tmp_path)
    changes = modernizer.process_file(
        pyproject,
        canonical_dev=[],
        dry_run=True,
        skip_comments=False,
    )
    tm.that("invalid TOML" in changes, eq=True)


def test_flext_infra_pyproject_modernizer_find_pyproject_files(tmp_path: Path) -> None:
    (tmp_path / "project1").mkdir()
    (tmp_path / "project1" / "pyproject.toml").write_text(
        "[project]\n", encoding="utf-8"
    )
    (tmp_path / "project2").mkdir()
    (tmp_path / "project2" / "pyproject.toml").write_text(
        "[project]\n", encoding="utf-8"
    )
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    files = FlextInfraPyprojectModernizer(tmp_path).find_pyproject_files()
    tm.that(files, length=2)
    tm.that(all("project" in str(path) for path in files), eq=True)
    tm.that(hasattr(h, "assert_ok"), eq=True)
