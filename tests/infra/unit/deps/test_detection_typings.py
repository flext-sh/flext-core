from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra import FlextInfraDependencyDetectionService
from flext_tests import tm
from tests.infra import m, t


class _StubToml:
    def __init__(self, values: list[r[t.Infra.TomlConfig]]) -> None:
        self._values = values
        self._idx = 0

    def read_plain(self, path: Path) -> r[t.Infra.TomlConfig]:
        _ = path
        value = self._values[self._idx]
        if self._idx < len(self._values) - 1:
            self._idx += 1
        return value


class _StubRunner:
    def __init__(self, result: r[m.Infra.Core.CommandOutput]) -> None:
        self._result = result
        self.last_kwargs: dict[str, str | int | Path | dict[str, str]] = {}

    def run_raw(
        self, *args: t.Infra.TomlValue, **kwargs: str | int | Path | dict[str, str]
    ) -> r[m.Infra.Core.CommandOutput]:
        _ = args
        self.last_kwargs = kwargs
        return self._result


class TestLoadDependencyLimits:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = FlextInfraDependencyDetectionService()
        monkeypatch.setattr(
            service,
            "toml",
            _StubToml([r[t.Infra.TomlConfig].ok({"key": "value", "num": 42})]),
        )
        result = service.load_dependency_limits(Path("/fake/limits.toml"))
        assert result.get("key") == "value"
        assert result.get("num") == 42

    def test_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = FlextInfraDependencyDetectionService()
        monkeypatch.setattr(
            service,
            "toml",
            _StubToml([r[t.Infra.TomlConfig].fail("not found")]),
        )
        tm.that(service.load_dependency_limits(Path("/fake/limits.toml")), eq={})

    def test_unconvertible_values_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        monkeypatch.setattr(
            service,
            "toml",
            _StubToml([r[t.Infra.TomlConfig].ok({"good": "val", "bad": ["x"]})]),
        )
        result = service.load_dependency_limits(Path("/fake/limits.toml"))
        assert "good" in result
        assert "bad" in result

    def test_none_value_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = FlextInfraDependencyDetectionService()
        monkeypatch.setattr(
            service,
            "toml",
            _StubToml([r[t.Infra.TomlConfig].ok({"key": None})]),
        )
        result = service.load_dependency_limits(Path("/fake/limits.toml"))
        assert "key" in result
        tm.that(result["key"], eq=None)


class TestRunMypyStubHints:
    def test_mypy_not_found(self, tmp_path: Path) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        tm.that(tm.ok(service.run_mypy_stub_hints(tmp_path, venv_bin)), eq=([], []))

    def test_runner_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")
        monkeypatch.setattr(
            service,
            "runner",
            _StubRunner(r[m.Infra.Core.CommandOutput].fail("mypy crash")),
        )
        tm.fail(service.run_mypy_stub_hints(tmp_path, venv_bin))

    def test_parses_hints(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")
        out = m.Infra.Core.CommandOutput(
            exit_code=0,
            stdout='note: hint: "pip install types-pyyaml"',
            stderr='error: Library stubs not installed for "requests"',
        )
        monkeypatch.setattr(
            service, "runner", _StubRunner(r[m.Infra.Core.CommandOutput].ok(out))
        )
        tm.ok(service.run_mypy_stub_hints(tmp_path, venv_bin))

    def test_run_mypy_stub_hints_with_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")
        out = m.Infra.Core.CommandOutput(exit_code=0, stdout="", stderr="")
        runner = _StubRunner(r[m.Infra.Core.CommandOutput].ok(out))
        monkeypatch.setattr(service, "runner", runner)
        tm.ok(service.run_mypy_stub_hints(tmp_path, venv_bin, timeout=600))
        tm.that(runner.last_kwargs["timeout"], eq=600)
