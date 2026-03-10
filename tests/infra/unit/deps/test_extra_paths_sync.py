from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import tomlkit

from flext_core import r
from flext_infra.deps import extra_paths
from flext_infra.deps.extra_paths import main, sync_extra_paths, sync_one
from flext_tests import tm
from tests.infra.typings import t

from ...helpers import h


class TestSyncExtraPaths:
    def test_sync_extra_paths_with_project_dirs(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        pyproject = project / "pyproject.toml"
        pyproject.write_text("[tool.pyright]\nextraPaths = []\n", encoding="utf-8")
        tm.ok(sync_extra_paths(project_dirs=[project]))

    def test_sync_extra_paths_dry_run(self, tmp_path: Path) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        pyproject = project / "pyproject.toml"
        pyproject.write_text("[tool.pyright]\nextraPaths = ['old']\n", encoding="utf-8")
        tm.ok(sync_extra_paths(dry_run=True, project_dirs=[project]))
        tm.that(pyproject.read_text(encoding="utf-8"), contains="old")

    def test_sync_extra_paths_no_project_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(extra_paths, "ROOT", tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.pyright]\nextraPaths = []\n", encoding="utf-8")
        tm.ok(sync_extra_paths())

    def test_sync_extra_paths_missing_root_pyproject(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(extra_paths, "ROOT", tmp_path)
        tm.fail(sync_extra_paths(), has="Missing")

    def test_sync_extra_paths_sync_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()

        def _fail_sync(
            *_args: t.ContainerValue,
            **_kwargs: t.ContainerValue,
        ) -> r[bool]:
            return r[bool].fail("sync error")

        monkeypatch.setattr(extra_paths, "sync_one", _fail_sync)
        tm.fail(sync_extra_paths(project_dirs=[project]), has="sync error")

    def test_sync_extra_paths_with_empty_project_list(self) -> None:
        tm.ok(sync_extra_paths(dry_run=True, project_dirs=[]))

    def test_sync_extra_paths_with_no_args(self) -> None:
        tm.ok(sync_extra_paths(dry_run=True))


class TestMain:
    def test_main_no_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(extra_paths, "ROOT", tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.pyright]\nextraPaths = []\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["extra_paths.py"])
        tm.that(main(), eq=0)

    def test_main_with_dry_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(extra_paths, "ROOT", tmp_path)
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.pyright]\nextraPaths = []\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["extra_paths.py", "--dry-run"])
        tm.that(main(), eq=0)

    def test_main_with_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        pyproject = project / "pyproject.toml"
        pyproject.write_text("[tool.pyright]\nextraPaths = []\n", encoding="utf-8")
        monkeypatch.setattr(extra_paths, "ROOT", tmp_path)
        monkeypatch.setattr(sys, "argv", ["extra_paths.py", "--project", "proj"])
        tm.that(main(), eq=0)

    def test_main_with_multiple_projects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ["proj-a", "proj-b"]:
            project = tmp_path / name
            project.mkdir()
            pyproject = project / "pyproject.toml"
            pyproject.write_text("[tool.pyright]\nextraPaths = []\n", encoding="utf-8")
        monkeypatch.setattr(extra_paths, "ROOT", tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            ["extra_paths.py", "--project", "proj-a", "--project", "proj-b"],
        )
        tm.that(main(), eq=0)

    def test_main_sync_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["extra_paths.py"])

        def _fail_sync(
            *_args: t.ContainerValue,
            **_kwargs: t.ContainerValue,
        ) -> r[int]:
            return r[int].fail("sync error")

        monkeypatch.setattr(extra_paths, "sync_extra_paths", _fail_sync)
        tm.that(main(), eq=1)

    def test_main_cli_with_project_arg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "project"
        project.mkdir()
        pyproject = project / "pyproject.toml"
        pyproject.write_text("[tool.pyright]\nextraPaths = []\n", encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv", ["prog", "--project", str(project), "--dry-run"]
        )
        tm.that(main(), eq=0)


class TestSyncOneEdgeCases:
    def test_sync_one_with_nonexistent_file(self) -> None:
        tm.that(
            h.assert_ok(sync_one(Path("/nonexistent/pyproject.toml"), dry_run=True)),
            eq=False,
        )

    def test_sync_one_with_invalid_toml(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("invalid toml {", encoding="utf-8")
        tm.fail(sync_one(pyproject, dry_run=True))

    def test_sync_one_stubbed_toml_service(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.pyright]\nextraPaths = []\n", encoding="utf-8")
        doc = tomlkit.document()
        doc["tool"] = {"pyright": {"extraPaths": []}}

        def _read_document(_path: Path) -> r[tomlkit.TOMLDocument]:
            return r[tomlkit.TOMLDocument].ok(doc)

        def _write_document(
            _path: Path,
            _doc: tomlkit.TOMLDocument,
        ) -> r[bool]:
            return r[bool].ok(True)

        stub = types.SimpleNamespace(
            read_document=_read_document,
            write_document=_write_document,
        )
        monkeypatch.setattr(extra_paths, "FlextInfraUtilitiesToml", lambda: stub)
        tm.ok(sync_one(pyproject, is_root=True))
