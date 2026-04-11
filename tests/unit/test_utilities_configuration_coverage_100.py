from __future__ import annotations

import logging
from pathlib import Path

import pytest

from flext_core import FlextContainer
from tests import c, u

pytestmark = [pytest.mark.unit, pytest.mark.coverage]


class TestFlextUtilitiesConfiguration:
    def test_get_log_level_from_config_uses_default_constant(self) -> None:
        expected = getattr(logging, c.DEFAULT_LEVEL.upper(), logging.INFO)
        assert u.get_log_level_from_config() == expected

    def test_resolve_env_file_prefers_existing_env_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        env_file = tmp_path / "custom.env"
        env_file.write_text("APP_NAME=test\n", encoding="utf-8")
        monkeypatch.setenv(c.ENV_FILE_ENV_VAR, str(env_file))
        assert u.resolve_env_file() == str(env_file.resolve())

    def test_resolve_env_file_returns_override_when_target_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        missing = tmp_path / "missing.env"
        monkeypatch.setenv(c.ENV_FILE_ENV_VAR, str(missing))
        assert u.resolve_env_file() == str(missing)

    def test_resolve_env_file_uses_default_file_in_cwd(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        default_env = tmp_path / c.ENV_FILE_DEFAULT
        default_env.write_text("APP_NAME=test\n", encoding="utf-8")
        monkeypatch.delenv(c.ENV_FILE_ENV_VAR, raising=False)
        monkeypatch.chdir(tmp_path)
        assert u.resolve_env_file() == str(default_env.resolve())

    def test_resolve_env_file_returns_default_name_when_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv(c.ENV_FILE_ENV_VAR, raising=False)
        monkeypatch.chdir(tmp_path)
        assert u.resolve_env_file() == c.ENV_FILE_DEFAULT

    def test_register_factory_registers_real_container_service(self) -> None:
        container = FlextContainer()
        register_result = u.register_factory(
            container,
            "unit_test_factory",
            lambda: "factory-value",
        )
        assert register_result.success
        resolved = container.get("unit_test_factory")
        assert resolved.success
        assert resolved.value == "factory-value"

    def test_resolve_effective_log_level_prioritizes_trace(self) -> None:
        assert (
            u.resolve_effective_log_level(
                trace=True,
                debug=False,
                log_level=c.LogLevel.WARNING,
            )
            == c.LogLevel.DEBUG
        )

    def test_resolve_effective_log_level_promotes_debug_to_info(self) -> None:
        assert (
            u.resolve_effective_log_level(
                trace=False,
                debug=True,
                log_level=c.LogLevel.ERROR,
            )
            == c.LogLevel.INFO
        )

    def test_resolve_effective_log_level_keeps_explicit_level(self) -> None:
        assert (
            u.resolve_effective_log_level(
                trace=False,
                debug=False,
                log_level=c.LogLevel.ERROR,
            )
            == c.LogLevel.ERROR
        )

    @pytest.mark.parametrize(
        ("url"),
        [
            pytest.param("postgresql://localhost/db", id="postgresql"),
            pytest.param("mysql://localhost/db", id="mysql"),
            pytest.param("sqlite:///tmp/test.db", id="sqlite"),
            pytest.param("", id="empty"),
        ],
    )
    def test_validate_database_url_scheme_accepts_supported_urls(
        self,
        url: str,
    ) -> None:
        u.validate_database_url_scheme(url)

    def test_validate_database_url_scheme_rejects_invalid_scheme(self) -> None:
        with pytest.raises(ValueError, match="Invalid database URL scheme"):
            u.validate_database_url_scheme("oracle://localhost/db")

    def test_validate_trace_requires_debug_accepts_valid_combinations(self) -> None:
        u.validate_trace_requires_debug(trace=False, debug=False)
        u.validate_trace_requires_debug(trace=False, debug=True)
        u.validate_trace_requires_debug(trace=True, debug=True)

    def test_validate_trace_requires_debug_rejects_trace_without_debug(self) -> None:
        with pytest.raises(ValueError, match="Trace mode requires debug mode"):
            u.validate_trace_requires_debug(trace=True, debug=False)
