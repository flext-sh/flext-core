from __future__ import annotations

import types
from pathlib import Path

from flext_core import r
from flext_infra.deps.internal_sync import FlextInfraInternalDependencySyncService
from flext_tests import tm
from tests.infra import h
from tests.infra.typings import t


def _set_toml_stub(
    service: FlextInfraInternalDependencySyncService,
    values: list[t.ContainerValue],
) -> None:
    state = {"index": 0}

    def _read(_path: Path) -> t.ContainerValue:
        item = values[state["index"]]
        state["index"] += 1
        return item

    service.toml = types.SimpleNamespace(read_plain=_read)


class TestSyncMethodEdgeCasesMore:
    def test_sync_checkout_failure(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[tool.poetry.dependencies]\nflext-core = { path = "../flext-core" }\n'
        )
        (tmp_path / "flext-repo-map.toml").write_text(
            '[repo.flext-core]\nssh_url = "git@github.com:flext-sh/flext-core.git"\nhttps_url = "https://github.com/flext-sh/flext-core.git"\n'
        )
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            [
                r[dict[str, t.ContainerValue]].ok({
                    "tool": {
                        "poetry": {
                            "dependencies": {"flext-core": {"path": "../flext-core"}}
                        }
                    },
                    "project": {},
                }),
                r[dict[str, t.ContainerValue]].ok({
                    "repo": {
                        "flext-core": {
                            "ssh_url": "git@github.com:flext-sh/flext-core.git",
                            "https_url": "https://github.com/flext-sh/flext-core.git",
                        }
                    }
                }),
            ],
        )
        monkeypatch.setattr(
            service,
            "ensure_checkout",
            lambda _dep, _url, _ref: r[bool].fail("checkout failed"),
        )
        tm.fail(service.sync(tmp_path))

    def test_sync_no_dependencies(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            [r[dict[str, t.ContainerValue]].ok({"project": {"name": "test"}})],
        )
        tm.ok(service.sync(tmp_path), eq=0)
        tm.that(h is not None, eq=True)
