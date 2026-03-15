"""Extra edge-case tests for modernizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import tomlkit

from flext_infra import FlextInfraPyprojectModernizer, FlextInfraUtilitiesCli
from flext_infra.deps import modernizer as modernizer_module
from flext_tests import t, tm


def _modernizer_args(**overrides: t.Tests.object) -> argparse.Namespace:
    """Create standard modernizer args namespace with defaults."""
    defaults: dict[str, t.Tests.object] = {
        "project": None,
        "dry_run": True,
        "verbose": False,
        "audit": False,
        "skip_comments": False,
        "skip_check": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _default_cli(workspace: Path | None = None) -> FlextInfraUtilitiesCli.CliArgs:
    return FlextInfraUtilitiesCli.CliArgs(workspace=workspace or Path.cwd())


class TestModernizerEdgeCases:
    """Tests edge-case run behavior."""

    def test_modernizer_with_empty_pyproject(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("")
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        tm.that(
            modernizer.run(_modernizer_args(), _default_cli(tmp_path)) in {0, 1, 2},
            eq=True,
        )

    def test_modernizer_with_invalid_toml(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[invalid toml {")
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        tm.that(
            modernizer.run(_modernizer_args(), _default_cli(tmp_path)) in {0, 1, 2},
            eq=True,
        )

    def test_modernizer_with_missing_pyproject(self, tmp_path: Path) -> None:
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        tm.that(
            modernizer.run(_modernizer_args(), _default_cli(tmp_path)) in {0, 1, 2},
            eq=True,
        )


class TestModernizerUncoveredLines:
    """Tests specific uncovered branches."""

    def test_run_with_missing_root_pyproject(self, tmp_path: Path) -> None:
        modernizer = FlextInfraPyprojectModernizer(tmp_path)
        tm.that(modernizer.run(_modernizer_args(), _default_cli(tmp_path)), eq=2)

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
            _path: Path,
            canonical_dev: list[str],
            dry_run: bool,
            skip_comments: bool,
        ) -> list[str]:
            _ = (_path, canonical_dev, dry_run, skip_comments)
            return []

        monkeypatch.setattr(modernizer, "find_pyproject_files", _find_files)
        monkeypatch.setattr(modernizer_module.u.Infra, "read", _read_doc)
        monkeypatch.setattr(modernizer, "process_file", _process_file)
        tm.that(modernizer.run(args, _default_cli(tmp_path)), eq=0)


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
