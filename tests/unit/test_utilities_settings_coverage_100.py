"""Utilities settings smoke tests for stable public helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from flext_core import FlextContainer
from tests import c, u


class TestsFlextUtilitiesSettings:
    def test_resolve_effective_log_level_prioritizes_trace(self) -> None:
        assert (
            u.resolve_effective_log_level(
                trace=True,
                debug=False,
                log_level=c.LogLevel.WARNING,
            )
            == c.LogLevel.DEBUG
        )

    def test_resolve_effective_log_level_promotes_debug(self) -> None:
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


class TestsFlextUtilitiesSettingsEnvFile:
    _orig: str | None

    def setup_method(self) -> None:
        self._orig = os.environ.pop(c.ENV_FILE_ENV_VAR, None)

    def teardown_method(self) -> None:
        if self._orig is not None:
            os.environ[c.ENV_FILE_ENV_VAR] = self._orig
        else:
            os.environ.pop(c.ENV_FILE_ENV_VAR, None)

    def test_resolve_env_file_default_when_no_env_var_and_no_dotenv(self) -> None:
        assert c.ENV_FILE_ENV_VAR not in os.environ
        result = u.resolve_env_file()
        assert result == c.ENV_FILE_DEFAULT

    def test_resolve_env_file_returns_nonexistent_path_as_is(self) -> None:
        os.environ[c.ENV_FILE_ENV_VAR] = "/tmp/flext_test_nonexistent.env"
        result = u.resolve_env_file()
        assert result == "/tmp/flext_test_nonexistent.env"

    def test_resolve_env_file_resolves_existing_path(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".env", delete=False) as f:
            tmp_path = f.name
        try:
            os.environ[c.ENV_FILE_ENV_VAR] = tmp_path
            result = u.resolve_env_file()
            assert result == str(Path(tmp_path).resolve())
        finally:
            Path(tmp_path).unlink()


class TestsFlextUtilitiesSettingsRegisterFactory:
    def setup_method(self) -> None:
        FlextContainer.reset_for_testing()

    def teardown_method(self) -> None:
        FlextContainer.reset_for_testing()

    def test_register_factory_returns_ok_true_on_success(self) -> None:
        container = FlextContainer()
        container.clear()
        result = u.register_factory(container, "test_svc", lambda: "service_value")
        assert not result.failure
        assert result.value is True
