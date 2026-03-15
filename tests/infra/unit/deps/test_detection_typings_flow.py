from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra import FlextInfraDependencyDetectionService, u
from flext_tests import tm
from tests.infra import m, t


class _StubReadPlain:
    """Stub for u.Infra.read_plain that returns predefined values."""

    def __init__(self, values: list[r[dict[str, t.Infra.TomlValue]]]) -> None:
        self._values = values
        self._idx = 0

    def __call__(self, path: Path) -> r[dict[str, t.Infra.TomlValue]]:
        _ = path
        value = self._values[self._idx]
        if self._idx < len(self._values) - 1:
            self._idx += 1
        return value


class _StubRunRaw:
    """Stub for u.Infra.run_raw that returns a predefined result."""

    def __init__(self, result: r[m.Infra.Core.CommandOutput]) -> None:
        self._result = result

    def __call__(
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
        module_to_package: dict[str, t.Infra.InfraValue] = {"yaml": "custom-types-yaml"}
        typing_libraries: dict[str, t.Infra.InfraValue] = {
            "module_to_package": module_to_package
        }
        limits: dict[str, t.Infra.TomlValue] = {"typing_libraries": typing_libraries}
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
            u.Infra,
            "read_plain",
            _StubReadPlain([r[dict[str, t.Infra.TomlValue]].ok(payload)]),
        )
        got = service.get_current_typings_from_pyproject(tmp_path)
        tm.that(got, eq=[])

    def test_get_current_typings_from_pyproject_variants(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        values: list[r[dict[str, t.Infra.TomlValue]]] = [
            r[dict[str, t.Infra.TomlValue]].ok({
                "project": {
                    "optional-dependencies": {
                        "typings": ["types-pyyaml>=6.0", "types-requests[extra]==2.28"]
                    }
                }
            }),
            r[dict[str, t.Infra.TomlValue]].ok({
                "project": {
                    "optional-dependencies": {"typings": {"types-pyyaml": ">=6.0"}}
                }
            }),
            r[dict[str, t.Infra.TomlValue]].fail("not found"),
            r[dict[str, t.Infra.TomlValue]].ok({}),
        ]
        monkeypatch.setattr(u.Infra, "read_plain", _StubReadPlain(values))
        tm.that(service.get_current_typings_from_pyproject(tmp_path), eq=[])
        tm.that(service.get_current_typings_from_pyproject(tmp_path), eq=[])
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
            u.Infra, "run_raw", _StubRunRaw(r[m.Infra.Core.CommandOutput].ok(out))
        )
        monkeypatch.setattr(
            u.Infra,
            "read_plain",
            _StubReadPlain([
                r[dict[str, t.Infra.TomlValue]].ok({}),
                r[dict[str, t.Infra.TomlValue]].ok({
                    "project": {"optional-dependencies": {"typings": []}}
                }),
            ]),
        )
        tm.ok(service.get_required_typings(tmp_path, venv_bin))
        monkeypatch.setattr(
            u.Infra,
            "read_plain",
            _StubReadPlain([
                r[dict[str, t.Infra.TomlValue]].ok({}),
                r[dict[str, t.Infra.TomlValue]].ok({}),
            ]),
        )
        tm.ok(service.get_required_typings(tmp_path, venv_bin, include_mypy=False))
        monkeypatch.setattr(
            u.Infra,
            "run_raw",
            _StubRunRaw(r[m.Infra.Core.CommandOutput].fail("mypy crash")),
        )
        monkeypatch.setattr(
            u.Infra,
            "read_plain",
            _StubReadPlain([r[dict[str, t.Infra.TomlValue]].ok({})]),
        )
        tm.fail(service.get_required_typings(tmp_path, venv_bin))
