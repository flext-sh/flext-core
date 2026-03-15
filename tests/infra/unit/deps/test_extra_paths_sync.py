from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import tomlkit
from tomlkit.toml_document import TOMLDocument

from flext_core import r
from flext_infra import FlextInfraExtraPathsManager
from flext_infra.deps import extra_paths
from flext_infra.deps.extra_paths import main
from flext_tests import tf, tm


@pytest.fixture
def pyright_content() -> str:
    return "[tool.pyright]\nextraPaths = []\n"


def _create_pyproject(directory: Path, content: str) -> Path:
    return tf.create_in(content=content, name="pyproject.toml", directory=directory)


def _manager() -> FlextInfraExtraPathsManager:
    return FlextInfraExtraPathsManager()


class _ManagerAdapter(FlextInfraExtraPathsManager):
    def __init__(
        self,
        root: Path | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        super().__init__(workspace_root=workspace_root or root)


@pytest.mark.parametrize(
    ("mode", "dry_run", "project_dirs", "expect_fail", "expect_has"),
    [
        ("project", False, ["proj"], False, None),
        ("project", True, ["proj"], False, "old"),
        ("root", False, None, False, None),
        ("none", True, [], False, None),
        ("none", True, None, False, None),
    ],
)
def test_sync_extra_paths_success_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pyright_content: str,
    mode: str,
    dry_run: bool,
    project_dirs: list[str] | None,
    expect_fail: bool,
    expect_has: str | None,
) -> None:
    project_dirs_arg: list[Path] | None = None
    if mode == "project":
        project = tmp_path / "proj"
        project.mkdir()
        content = (
            "[tool.pyright]\nextraPaths = ['old']\n" if dry_run else pyright_content
        )
        pyproject = _create_pyproject(project, content)
        project_dirs_arg = [project] if project_dirs else []
        result = _manager().sync_extra_paths(
            dry_run=dry_run, project_dirs=project_dirs_arg
        )
        tm.ok(result)
        if expect_has:
            tm.that(pyproject.read_text(encoding="utf-8"), has=expect_has)
        return
    if mode == "root":
        monkeypatch.setattr(FlextInfraExtraPathsManager, "ROOT", tmp_path)
        _ = _create_pyproject(tmp_path, pyright_content)
        tm.ok(_manager().sync_extra_paths())
        return
    result = _manager().sync_extra_paths(
        dry_run=dry_run, project_dirs=[] if project_dirs == [] else None
    )
    if expect_fail:
        tm.fail(result)
        return
    tm.ok(result)


def test_sync_extra_paths_missing_root_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FlextInfraExtraPathsManager, "ROOT", tmp_path)
    tm.fail(_manager().sync_extra_paths(), has="Missing")


def test_sync_extra_paths_sync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()

    def _fail_sync(
        _self: FlextInfraExtraPathsManager,
        _pyproject_path: Path,
        *,
        dry_run: bool = False,
        is_root: bool = False,
    ) -> r[bool]:
        _ = _self, _pyproject_path, dry_run, is_root
        return r[bool].fail("sync error")

    monkeypatch.setattr(FlextInfraExtraPathsManager, "sync_one", _fail_sync)
    tm.fail(_manager().sync_extra_paths(project_dirs=[project]), has="sync error")


@pytest.mark.parametrize(
    ("mode", "argv", "expected_exit"),
    [
        ("root", ["extra_paths.py"], 0),
        ("root", ["extra_paths.py", "--dry-run"], 0),
        ("project", ["extra_paths.py", "--project", "proj"], 0),
        (
            "multi",
            ["extra_paths.py", "--projects", "proj-a,proj-b"],
            0,
        ),
        ("abs-project", ["prog", "--project", "project", "--dry-run"], 0),
    ],
)
def test_main_success_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pyright_content: str,
    mode: str,
    argv: list[str],
    expected_exit: int,
) -> None:
    monkeypatch.setattr(extra_paths, "FlextInfraExtraPathsManager", _ManagerAdapter)
    monkeypatch.setattr(FlextInfraExtraPathsManager, "ROOT", tmp_path)
    if mode == "root":
        _ = _create_pyproject(tmp_path, pyright_content)
    if mode == "project":
        project = tmp_path / "proj"
        project.mkdir()
        _ = _create_pyproject(project, pyright_content)
    if mode == "multi":
        for name in ["proj-a", "proj-b"]:
            project = tmp_path / name
            project.mkdir()
            _ = _create_pyproject(project, pyright_content)
    if mode == "abs-project":
        project = tmp_path / "project"
        project.mkdir()
        _ = _create_pyproject(project, pyright_content)
        argv = ["prog", "--project", str(project), "--dry-run"]
    argv = [argv[0], "--workspace", str(tmp_path), *argv[1:]]
    monkeypatch.setattr(sys, "argv", argv)
    tm.that(main(), eq=expected_exit)


def test_main_sync_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["extra_paths.py"])
    monkeypatch.setattr(extra_paths, "FlextInfraExtraPathsManager", _ManagerAdapter)

    def _fail_sync(
        _self: FlextInfraExtraPathsManager,
        *,
        dry_run: bool = False,
        project_dirs: list[Path] | None = None,
    ) -> r[int]:
        _ = _self, dry_run, project_dirs
        return r[int].fail("sync error")

    monkeypatch.setattr(FlextInfraExtraPathsManager, "sync_extra_paths", _fail_sync)
    tm.that(main(), eq=1)


@pytest.mark.parametrize(
    ("mode", "dry_run", "expected_ok", "expect_fail"),
    [
        ("nonexistent", True, False, False),
        ("invalid", True, False, True),
        ("stubbed", False, True, False),
    ],
)
def test_sync_one_edge_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pyright_content: str,
    mode: str,
    dry_run: bool,
    expected_ok: bool,
    expect_fail: bool,
) -> None:
    if mode == "nonexistent":
        tm.that(
            _manager()
            .sync_one(Path("/nonexistent/pyproject.toml"), dry_run=dry_run)
            .is_success,
            eq=False,
        )
        return
    if mode == "invalid":
        pyproject = _create_pyproject(tmp_path, "invalid toml {")
        result = _manager().sync_one(pyproject, dry_run=dry_run)
        if expect_fail:
            tm.fail(result)
            return
        tm.that(result.is_success, eq=expected_ok)
        return
    pyproject = _create_pyproject(tmp_path, pyright_content)
    doc = tomlkit.document()
    tool = tomlkit.table()
    pyright = tomlkit.table()
    pyright["extraPaths"] = tomlkit.array()
    tool["pyright"] = pyright
    doc["tool"] = tool

    def _read_document(_path: Path) -> r[TOMLDocument]:
        return r[TOMLDocument].ok(doc)

    def _write_document(_path: Path, _doc: TOMLDocument) -> r[bool]:
        return r[bool].ok(True)

    stub = types.SimpleNamespace(
        read_document=_read_document,
        write_document=_write_document,
    )
    monkeypatch.setattr(extra_paths, "FlextInfraUtilitiesToml", lambda: stub)
    tm.ok(_manager().sync_one(pyproject, is_root=True, dry_run=dry_run))
