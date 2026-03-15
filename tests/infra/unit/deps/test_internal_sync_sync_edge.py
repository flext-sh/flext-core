from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from flext_core import r
from flext_infra import FlextInfraInternalDependencySyncService
from flext_tests import tm
from tests.infra import t


def _set_toml_stub(
    service: FlextInfraInternalDependencySyncService,
    values: list[r[t.Infra.TomlConfig]],
) -> None:
    state = {"index": 0}

    def _read(_path: Path) -> r[t.Infra.TomlConfig]:
        item = values[state["index"]]
        state["index"] += 1
        return item

    class _TomlReaderStub:
        def __init__(self, fn: Callable[[Path], r[t.Infra.TomlConfig]]) -> None:
            self._fn = fn

        def read_plain(self, path: Path) -> r[t.Infra.TomlConfig]:
            return self._fn(path)

    service.toml = _TomlReaderStub(_read)


class TestSyncMethodEdgeCases:
    def test_sync_with_parsed_repo_map_failure(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[tool.poetry.dependencies]\nflext-core = { path = "../flext-core" }\n'
        )
        (tmp_path / "flext-repo-map.toml").write_text("invalid toml {")
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            [
                r[t.Infra.TomlConfig].ok({
                    "tool": {
                        "poetry": {
                            "dependencies": {"flext-core": {"path": "../flext-core"}}
                        }
                    },
                    "project": {},
                }),
                r[t.Infra.TomlConfig].ok({
                    "tool": {
                        "poetry": {
                            "dependencies": {"flext-core": {"path": "../flext-core"}}
                        }
                    },
                    "project": {},
                }),
            ],
        )
        tm.fail(service.sync(tmp_path))

    def test_sync_with_workspace_mode_and_gitmodules(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".gitmodules").write_text(
            '[submodule "flext-core"]\n\turl = git@github.com:flext-sh/flext-core.git\n'
        )
        project = workspace / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[tool.poetry.dependencies]\nflext-core = { path = "../flext-core" }\n'
        )
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            [
                r[t.Infra.TomlConfig].ok({
                    "tool": {
                        "poetry": {
                            "dependencies": {"flext-core": {"path": "../flext-core"}}
                        }
                    },
                    "project": {},
                })
            ],
        )
        monkeypatch.setenv("FLEXT_WORKSPACE_ROOT", str(workspace))

        def _resolve_ref(_root: Path) -> str:
            return "main"

        def _ensure_checkout(_dep: Path, _url: str, _ref: str) -> r[bool]:
            return r[bool].ok(True)

        monkeypatch.setattr(service, "resolve_ref", _resolve_ref)
        monkeypatch.setattr(
            service,
            "ensure_checkout",
            _ensure_checkout,
        )
        tm.that(service.sync(project).is_success, eq=True)

    def test_sync_with_synthesized_repo_map(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[tool.poetry.dependencies]\nflext-core = { path = "../flext-core" }\n'
        )
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            [
                r[t.Infra.TomlConfig].ok({
                    "tool": {
                        "poetry": {
                            "dependencies": {"flext-core": {"path": "../flext-core"}}
                        }
                    },
                    "project": {},
                })
            ],
        )

        def _infer_owner(_root: Path) -> str:
            return "flext-sh"

        def _resolve_ref(_root: Path) -> str:
            return "main"

        def _ensure_checkout(_dep: Path, _url: str, _ref: str) -> r[bool]:
            return r[bool].ok(True)

        monkeypatch.setattr(
            service,
            "infer_owner_from_origin",
            _infer_owner,
        )
        monkeypatch.setattr(service, "resolve_ref", _resolve_ref)
        monkeypatch.setattr(service, "ensure_checkout", _ensure_checkout)
        tm.that(service.sync(tmp_path).is_success, eq=True)

    def test_sync_missing_repo_mapping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[tool.poetry.dependencies]\nflext-core = { path = "../flext-core" }\n'
        )
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            [
                r[t.Infra.TomlConfig].ok({
                    "tool": {
                        "poetry": {
                            "dependencies": {"flext-core": {"path": "../flext-core"}}
                        }
                    },
                    "project": {},
                })
            ],
        )

        def _infer_owner(_root: Path) -> None:
            return None

        monkeypatch.setattr(service, "infer_owner_from_origin", _infer_owner)
        tm.fail(service.sync(tmp_path))

    def test_sync_symlink_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".gitmodules").write_text(
            '[submodule "flext-core"]\n\turl = git@github.com:flext-sh/flext-core.git\n'
        )
        (workspace / "flext-core").mkdir()
        project = workspace / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text(
            '[tool.poetry.dependencies]\nflext-core = { path = "../flext-core" }\n'
        )
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            [
                r[t.Infra.TomlConfig].ok({
                    "tool": {
                        "poetry": {
                            "dependencies": {"flext-core": {"path": "../flext-core"}}
                        }
                    },
                    "project": {},
                })
            ],
        )
        monkeypatch.setenv("FLEXT_WORKSPACE_ROOT", str(workspace))

        def _ensure_symlink_fail(_dep: Path, _sib: Path) -> r[bool]:
            return r[bool].fail("symlink failed")

        monkeypatch.setattr(
            service,
            "ensure_symlink",
            _ensure_symlink_fail,
        )
        tm.fail(service.sync(project))
