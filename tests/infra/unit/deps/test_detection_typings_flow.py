from __future__ import annotations

from pathlib import Path


import pytest

from flext_core import r
from flext_infra.deps.detection import FlextInfraDependencyDetectionService
from flext_tests import tm

from ...models import m
from ...typings import t


class _StubToml:
    def __init__(self, values: list[r[dict[str, t.Infra.TomlValue]]]) -> None:
        self._values = values
        self._idx = 0

    def read_plain(self, path: Path) -> r[dict[str, t.Infra.TomlValue]]:
        _ = path
        value = self._values[self._idx]
        if self._idx < len(self._values) - 1:
            self._idx += 1
        return value


class _StubRunner:
    def __init__(self, result: r[m.Infra.Core.CommandOutput]) -> None:
        self._result = result

    def run_raw(
        self, *args: t.Infra.TomlValue, **kwargs: t.Infra.TomlValue
    ) -> r[m.Infra.Core.CommandOutput]:
        _ = args
        _ = kwargs
        return self._result


class TestModuleAndTypingsFlow:
    def test_module_to_types_package(self) -> None:
        service = FlextInfraDependencyDetectionService()
        tm.that(service.module_to_types_package("yaml", {}), eq="types-pyyaml")
        tm.that(service.module_to_types_package("flext_core", {}), eq=None)
        limits = {
            "typing_libraries": {"module_to_package": {"yaml": "custom-types-yaml"}}
        }
        tm.that(service.module_to_types_package("yaml", limits), eq="custom-types-yaml")
        tm.that(service.module_to_types_package("unknown_module", {}), eq=None)
        tm.that(service.module_to_types_package("yaml.parser", {}), eq="types-pyyaml")

    def test_get_current_typings_from_pyproject(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        payload: dict[str, t.Infra.TomlValue] = {
            "tool": {
                "poetry": {
                    "group": {
                        "typings": {
                            "dependencies": {
                                "types-pyyaml": "^6.0",
                                "types-requests": "^2.28",
                            }
                        }
                    }
                }
            }
        }
        monkeypatch.setattr(
            service,
            "toml",
            _StubToml([r[t.Infra.TomlConfig].ok(payload)]),
        )
        got = service.get_current_typings_from_pyproject(tmp_path)
        tm.that("types-pyyaml" in got, eq=True)
        tm.that("types-requests" in got, eq=True)

    def test_get_current_typings_from_pyproject_variants(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        values: list[r[dict[str, t.Infra.TomlValue]]] = [
            r[t.Infra.TomlConfig].ok({
                "project": {
                    "optional-dependencies": {
                        "typings": ["types-pyyaml>=6.0", "types-requests[extra]==2.28"]
                    }
                }
            }),
            r[t.Infra.TomlConfig].ok({
                "project": {
                    "optional-dependencies": {"typings": {"types-pyyaml": ">=6.0"}}
                }
            }),
            r[t.Infra.TomlConfig].fail("not found"),
            r[t.Infra.TomlConfig].ok({}),
        ]
        monkeypatch.setattr(service, "toml", _StubToml(values))
        tm.that(
            "types-pyyaml" in service.get_current_typings_from_pyproject(tmp_path),
            eq=True,
        )
        tm.that(
            "types-pyyaml" in service.get_current_typings_from_pyproject(tmp_path),
            eq=True,
        )
        tm.that(service.get_current_typings_from_pyproject(tmp_path), eq=[])
        tm.that(service.get_current_typings_from_pyproject(tmp_path), eq=[])

    def test_get_required_typings_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")
        out = m.Infra.Core.CommandOutput(exit_code=0, stdout="", stderr="")
        service = FlextInfraDependencyDetectionService()
        monkeypatch.setattr(
            service, "runner", _StubRunner(r[m.Infra.Core.CommandOutput].ok(out))
        )
        monkeypatch.setattr(
            service,
            "toml",
            _StubToml([
                r[t.Infra.TomlConfig].ok({}),
                r[t.Infra.TomlConfig].ok({
                    "project": {"optional-dependencies": {"typings": []}}
                }),
            ]),
        )
        tm.ok(service.get_required_typings(tmp_path, venv_bin))
        monkeypatch.setattr(
            service,
            "toml",
            _StubToml([
                r[t.Infra.TomlConfig].ok({}),
                r[t.Infra.TomlConfig].ok({}),
            ]),
        )
        tm.ok(service.get_required_typings(tmp_path, venv_bin, include_mypy=False))
        monkeypatch.setattr(
            service,
            "runner",
            _StubRunner(r[m.Infra.Core.CommandOutput].fail("mypy crash")),
        )
        monkeypatch.setattr(
            service,
            "toml",
            _StubToml([r[t.Infra.TomlConfig].ok({})]),
        )
        tm.fail(service.get_required_typings(tmp_path, venv_bin))
