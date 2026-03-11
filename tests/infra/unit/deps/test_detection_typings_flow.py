from __future__ import annotations

from pathlib import Path
import pytest

from flext_core import r
from flext_infra.deps.detection import FlextInfraDependencyDetectionService
from flext_tests import tm
from tests.infra.models import m
from tests.infra.typings import t


class _StubToml:
    def __init__(self, values: list[r[dict[str, t.ContainerValue]]]) -> None:
        self._values = values
        self._idx = 0

    def read_plain(self, path: Path) -> r[dict[str, t.ContainerValue]]:
        _ = path
        value = self._values[self._idx]
        if self._idx < len(self._values) - 1:
            self._idx += 1
        return value


class _StubRunner:
    def __init__(self, result: r[m.Infra.Core.CommandOutput]) -> None:
        self._result = result

    def run_raw(
        self, *args: t.ContainerValue, **kwargs: t.ContainerValue
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
        payload = {
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
            _StubToml([
                r[dict[str, dict[str, dict[str, dict[str, dict[str, str]]]]]].ok(
                    payload
                )
            ]),
        )
        got = service.get_current_typings_from_pyproject(tmp_path)
        tm.that("types-pyyaml" in got, eq=True)
        tm.that("types-requests" in got, eq=True)

    def test_get_current_typings_from_pyproject_variants(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        values: list[r[dict[str, t.ContainerValue]]] = [
            r[dict[str, t.ContainerValue]].ok({
                "project": {
                    "optional-dependencies": {
                        "typings": ["types-pyyaml>=6.0", "types-requests[extra]==2.28"]
                    }
                }
            }),
            r[dict[str, t.ContainerValue]].ok({
                "project": {
                    "optional-dependencies": {"typings": {"types-pyyaml": ">=6.0"}}
                }
            }),
            r[dict[str, t.ContainerValue]].fail("not found"),
            r[dict[str, t.ContainerValue]].ok({}),
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
                r[dict[str, t.ContainerValue]].ok({}),
                r[dict[str, t.ContainerValue]].ok({
                    "project": {"optional-dependencies": {"typings": []}}
                }),
            ]),
        )
        tm.ok(service.get_required_typings(tmp_path, venv_bin))
        monkeypatch.setattr(
            service,
            "toml",
            _StubToml([
                r[dict[str, t.ContainerValue]].ok({}),
                r[dict[str, t.ContainerValue]].ok({}),
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
            _StubToml([r[dict[str, t.ContainerValue]].ok({})]),
        )
        tm.fail(service.get_required_typings(tmp_path, venv_bin))
