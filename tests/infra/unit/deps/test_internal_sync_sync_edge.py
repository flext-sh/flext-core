from __future__ import annotations

import types
from pathlib import Path

from flext_core import r
from flext_infra.deps.internal_sync import FlextInfraInternalDependencySyncService
from flext_tests import tm


def _set_toml_stub(
    service: FlextInfraInternalDependencySyncService,
    values: list[object],
) -> None:
    state = {"index": 0}

    def _read(_path: Path) -> object:
        item = values[state["index"]]
        state["index"] += 1
        return item

    service.toml = types.SimpleNamespace(read_plain=_read)


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
                r[dict[str, object]].ok({
                    "tool": {
                        "poetry": {
                            "dependencies": {"flext-core": {"path": "../flext-core"}}
                        }
                    },
                    "project": {},
                }),
                r[dict[str, object]].ok({
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
        self, tmp_path: Path, monkeypatch
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
                r[dict[str, object]].ok({
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
        monkeypatch.setattr(service, "resolve_ref", lambda _root: "main")
        monkeypatch.setattr(
            service, "ensure_checkout", lambda _dep, _url, _ref: r[bool].ok(True)
        )
        tm.that(service.sync(project).is_success, eq=True)

    def test_sync_with_synthesized_repo_map(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[tool.poetry.dependencies]\nflext-core = { path = "../flext-core" }\n'
        )
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            [
                r[dict[str, object]].ok({
                    "tool": {
                        "poetry": {
                            "dependencies": {"flext-core": {"path": "../flext-core"}}
                        }
                    },
                    "project": {},
                })
            ],
        )
        monkeypatch.setattr(
            service, "infer_owner_from_origin", lambda _root: "flext-sh"
        )
        monkeypatch.setattr(service, "resolve_ref", lambda _root: "main")
        monkeypatch.setattr(
            service, "ensure_checkout", lambda _dep, _url, _ref: r[bool].ok(True)
        )
        tm.that(service.sync(tmp_path).is_success, eq=True)

    def test_sync_missing_repo_mapping(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[tool.poetry.dependencies]\nflext-core = { path = "../flext-core" }\n'
        )
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            [
                r[dict[str, object]].ok({
                    "tool": {
                        "poetry": {
                            "dependencies": {"flext-core": {"path": "../flext-core"}}
                        }
                    },
                    "project": {},
                })
            ],
        )
        monkeypatch.setattr(service, "infer_owner_from_origin", lambda _root: None)
        tm.fail(service.sync(tmp_path))

    def test_sync_symlink_failure(self, tmp_path: Path, monkeypatch) -> None:
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
                r[dict[str, object]].ok({
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
        monkeypatch.setattr(
            service, "ensure_symlink", lambda _dep, _sib: r[bool].fail("symlink failed")
        )
        tm.fail(service.sync(project))
