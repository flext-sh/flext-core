from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra import FlextInfraInternalDependencySyncService
from flext_tests import tm
from tests.infra import h, t


class _TomlStub:
    """Typed stub implementing TomlReader protocol for testing."""

    def __init__(self, values: list[r[t.Infra.TomlConfig]]) -> None:
        self._values = values
        self._index = 0

    def read_plain(self, path: Path) -> r[t.Infra.TomlConfig]:
        """Return next pre-configured result."""
        _ = path
        item = self._values[self._index]
        self._index += 1
        return item


def _set_toml_stub(
    service: FlextInfraInternalDependencySyncService,
    values: list[r[t.Infra.TomlConfig]],
) -> None:
    service.toml = _TomlStub(values)


class TestSyncMethodEdgeCasesMore:
    def test_sync_checkout_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
                r[t.Infra.TomlConfig].ok({
                    "tool": {
                        "poetry": {
                            "dependencies": {"flext-core": {"path": "../flext-core"}}
                        }
                    },
                    "project": {},
                }),
                r[t.Infra.TomlConfig].ok({
                    "repo": {
                        "flext-core": {
                            "ssh_url": "git@github.com:flext-sh/flext-core.git",
                            "https_url": "https://github.com/flext-sh/flext-core.git",
                        }
                    }
                }),
            ],
        )

        def _ensure_checkout_fail(_dep: Path, _url: str, _ref: str) -> r[bool]:
            return r[bool].fail("checkout failed")

        monkeypatch.setattr(
            service,
            "ensure_checkout",
            _ensure_checkout_fail,
        )
        tm.fail(service.sync(tmp_path))

    def test_sync_no_dependencies(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            [r[t.Infra.TomlConfig].ok({"project": {"name": "test"}})],
        )
        tm.ok(service.sync(tmp_path), eq=0)
        tm.that(hasattr(h, "assert_ok"), eq=True)
