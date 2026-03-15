from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from flext_core import r
from flext_infra import FlextInfraInternalDependencySyncService
from flext_infra.deps import internal_sync
from flext_tests import tm
from tests.infra import h, t


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


class TestSync:
    def test_sync_no_deps(self, tmp_path: Path) -> None:
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(service, [r[t.Infra.TomlConfig].ok({"tool": {}, "project": {}})])
        (tmp_path / "pyproject.toml").write_text("")
        tm.ok(service.sync(tmp_path), eq=0)

    def test_sync_collect_failure(self, tmp_path: Path) -> None:
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(service, [r[t.Infra.TomlConfig].fail("read error")])
        (tmp_path / "pyproject.toml").write_text("")
        tm.fail(service.sync(tmp_path))

    def test_sync_workspace_mode_symlink(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".gitmodules").write_text(
            '[submodule "flext-api"]\n\tpath = flext-api\n\turl = git@github.com:flext-sh/flext-api.git\n'
        )
        (workspace / "flext-api").mkdir()
        project = workspace / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            [
                r[t.Infra.TomlConfig].ok({
                    "tool": {
                        "poetry": {
                            "dependencies": {
                                "flext-api": {"path": ".flext-deps/flext-api"}
                            }
                        }
                    },
                    "project": {},
                })
            ],
        )
        monkeypatch.setenv("FLEXT_STANDALONE", "")
        monkeypatch.setenv("FLEXT_WORKSPACE_ROOT", "")

        def _git_run(_cmd: list[str], cwd: Path) -> r[str]:
            _ = cwd
            return r[str].ok("")

        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_run",
            _git_run,
        )
        tm.ok(service.sync(project))

    def test_sync_missing_repo_mapping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")
        (project / "flext-repo-map.toml").write_text("")
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            [
                r[t.Infra.TomlConfig].ok({
                    "tool": {
                        "poetry": {
                            "dependencies": {
                                "flext-api": {"path": ".flext-deps/flext-api"}
                            }
                        }
                    },
                    "project": {},
                }),
                r[t.Infra.TomlConfig].ok({"repo": {}}),
            ],
        )
        monkeypatch.setenv("FLEXT_STANDALONE", "")
        monkeypatch.setenv("FLEXT_WORKSPACE_ROOT", "")

        def _git_run(_cmd: list[str], cwd: Path) -> r[str]:
            _ = cwd
            return r[str].ok("")

        monkeypatch.setattr(
            internal_sync.u.Infra,
            "git_run",
            _git_run,
        )
        error = tm.fail(service.sync(project))
        tm.that(error, contains="missing repo mapping")
        tm.that(hasattr(h, "assert_ok"), eq=True)
