from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from flext_core import r
from flext_infra.deps.detection import FlextInfraDependencyDetectionService
from flext_tests import tm
from tests.infra import h


class _StubToml:
    def __init__(self, values) -> None:
        self._values = values
        self._idx = 0

    def read_plain(self, path: Path):
        _ = path
        value = self._values[self._idx]
        if self._idx < len(self._values) - 1:
            self._idx += 1
        return value


class _StubRunner:
    def __init__(self, result) -> None:
        self._result = result
        self.last_kwargs: dict[str, str | int | Path | dict[str, str]] = {}

    def run_raw(self, *args, **kwargs):
        _ = args
        self.last_kwargs = kwargs
        return self._result


class _StubMypyService(FlextInfraDependencyDetectionService):
    def __init__(self, hints_result, package) -> None:
        super().__init__()
        self._hints_result = hints_result
        self._package = package
        self.calls = 0

    def run_mypy_stub_hints(
        self, project_path: Path, venv_bin: Path, *, timeout: int = 300
    ):
        _ = project_path
        _ = venv_bin
        _ = timeout
        return self._hints_result

    def module_to_types_package(self, module_name: str, limits):
        _ = limits
        self.calls += 1
        _ = module_name
        return self._package


class TestLoadDependencyLimits:
    def test_success(self, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        monkeypatch.setattr(
            service,
            "toml",
            _StubToml([r[dict[str, str | int]].ok({"key": "value", "num": 42})]),
        )
        result = service.load_dependency_limits(Path("/fake/limits.toml"))
        tm.that(result["key"], eq="value")
        tm.that(result["num"], eq=42)

    def test_failure_returns_empty(self, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        monkeypatch.setattr(
            service, "toml", _StubToml([r[dict[str, str]].fail("not found")])
        )
        tm.that(service.load_dependency_limits(Path("/fake/limits.toml")), eq={})

    def test_unconvertible_values_skipped(self, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        monkeypatch.setattr(
            service,
            "toml",
            _StubToml([r[dict[str, str | set[str]]].ok({"good": "val", "bad": {"x"}})]),
        )
        result = service.load_dependency_limits(Path("/fake/limits.toml"))
        tm.that("good" in result, eq=True)
        tm.that("bad" in result, eq=False)

    def test_none_value_preserved(self, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        monkeypatch.setattr(
            service, "toml", _StubToml([r[dict[str, None]].ok({"key": None})])
        )
        result = service.load_dependency_limits(Path("/fake/limits.toml"))
        tm.that("key" in result, eq=True)
        tm.that(result["key"], eq=None)


class TestRunMypyStubHints:
    def test_mypy_not_found(self, tmp_path: Path) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        tm.that(
            h.assert_ok(service.run_mypy_stub_hints(tmp_path, venv_bin)), eq=([], [])
        )

    def test_runner_failure(self, tmp_path: Path, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")
        monkeypatch.setattr(
            service, "runner", _StubRunner(r[SimpleNamespace].fail("mypy crash"))
        )
        h.assert_fail(service.run_mypy_stub_hints(tmp_path, venv_bin))

    def test_parses_hints(self, tmp_path: Path, monkeypatch) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")
        out = SimpleNamespace(
            exit_code=0,
            stdout='note: hint: "pip install types-pyyaml"',
            stderr='error: Library stubs not installed for "requests"',
        )
        monkeypatch.setattr(service, "runner", _StubRunner(r[type(out)].ok(out)))
        h.assert_ok(service.run_mypy_stub_hints(tmp_path, venv_bin))

    def test_run_mypy_stub_hints_with_timeout(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")
        out = SimpleNamespace(exit_code=0, stdout="", stderr="")
        runner = _StubRunner(r[type(out)].ok(out))
        monkeypatch.setattr(service, "runner", runner)
        h.assert_ok(service.run_mypy_stub_hints(tmp_path, venv_bin, timeout=600))
        tm.that(runner.last_kwargs["timeout"], eq=600)


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
        self, tmp_path: Path, monkeypatch
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
        self, tmp_path: Path, monkeypatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        values = [
            r[dict[str, dict[str, dict[str, list[str]]]]].ok({
                "project": {
                    "optional-dependencies": {
                        "typings": ["types-pyyaml>=6.0", "types-requests[extra]==2.28"]
                    }
                }
            }),
            r[dict[str, dict[str, dict[str, dict[str, str]]]]].ok({
                "project": {
                    "optional-dependencies": {"typings": {"types-pyyaml": ">=6.0"}}
                }
            }),
            r[dict[str, str]].fail("not found"),
            r[dict[str, str]].ok({}),
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

    def test_get_required_typings_paths(self, tmp_path: Path, monkeypatch) -> None:
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")
        out = SimpleNamespace(exit_code=0, stdout="", stderr="")
        service = FlextInfraDependencyDetectionService()
        monkeypatch.setattr(service, "runner", _StubRunner(r[type(out)].ok(out)))
        monkeypatch.setattr(
            service,
            "toml",
            _StubToml([
                r[dict[str, str]].ok({}),
                r[dict[str, dict[str, dict[str, list[str]]]]].ok({
                    "project": {"optional-dependencies": {"typings": []}}
                }),
            ]),
        )
        h.assert_ok(service.get_required_typings(tmp_path, venv_bin))
        monkeypatch.setattr(
            service,
            "toml",
            _StubToml([r[dict[str, str]].ok({}), r[dict[str, str]].ok({})]),
        )
        h.assert_ok(
            service.get_required_typings(tmp_path, venv_bin, include_mypy=False)
        )
        monkeypatch.setattr(
            service, "runner", _StubRunner(r[SimpleNamespace].fail("mypy crash"))
        )
        monkeypatch.setattr(service, "toml", _StubToml([r[dict[str, str]].ok({})]))
        h.assert_fail(service.get_required_typings(tmp_path, venv_bin))
