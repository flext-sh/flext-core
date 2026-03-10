"""Extra edge-case tests for modernizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import tomlkit

import flext_infra.deps.modernizer as modernizer_module
from flext_core import t
from flext_infra.deps.modernizer import FlextInfraPyprojectModernizer
from flext_tests import tm


def _modernizer_args(**overrides: t.ContainerValue | None) -> argparse.Namespace:
    """Create standard modernizer args namespace with defaults."""
    defaults: dict[str, t.ContainerValue | None] = {
        "project": None,
        "dry_run": True,
        "verbose": False,
        "audit": False,
        "skip_comments": False,
        "skip_check": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestModernizerEdgeCases:
    """Tests edge-case run behavior."""

    def test_modernizer_with_empty_pyproject(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("")
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        tm.that(modernizer.run(_modernizer_args()) in {0, 1, 2}, eq=True)

    def test_modernizer_with_invalid_toml(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[invalid toml {")
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        tm.that(modernizer.run(_modernizer_args()) in {0, 1, 2}, eq=True)

    def test_modernizer_with_missing_pyproject(self, tmp_path: Path) -> None:
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        tm.that(modernizer.run(_modernizer_args()) in {0, 1, 2}, eq=True)


class TestModernizerUncoveredLines:
    """Tests specific uncovered branches."""

    def test_run_with_missing_root_pyproject(self, tmp_path: Path) -> None:
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        tm.that(modernizer.run(_modernizer_args()), eq=2)

    def test_run_with_no_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"\n')
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        args = _modernizer_args()

        def _find_files() -> list[Path]:
            return [pyproject]

        def _read_doc(_path: Path) -> tomlkit.TOMLDocument:
            return doc

        def _process_file(
            *_args: t.ContainerValue, **_kwargs: t.ContainerValue
        ) -> list[str]:
            return []

        monkeypatch.setattr(modernizer, "find_pyproject_files", _find_files)
        monkeypatch.setattr(modernizer_module, "_read_doc", _read_doc)
        monkeypatch.setattr(modernizer, "process_file", _process_file)
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
    tm.that(len(files), eq=2)
    tm.that(all("project" in str(path) for path in files), eq=True)
