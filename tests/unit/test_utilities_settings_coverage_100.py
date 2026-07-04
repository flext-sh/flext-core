"""Behavior contract for public settings helpers in bootstrap workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from flext_tests import tm

from flext_core import FlextContainer, u
from tests.constants import c
from tests.models import m

if TYPE_CHECKING:
    from tests.typings import t


class TestsFlextUtilitiesSettings:
    _original_env_file: str | None
    _original_cwd: Path

    def setup_method(self) -> None:
        self._original_env_file = os.environ.pop(c.ENV_FILE_ENV_VAR, None)
        self._original_cwd = Path.cwd()
        FlextContainer.reset_for_testing()

    def teardown_method(self) -> None:
        if self._original_env_file is not None:
            os.environ[c.ENV_FILE_ENV_VAR] = self._original_env_file
        else:
            os.environ.pop(c.ENV_FILE_ENV_VAR, None)
        os.chdir(self._original_cwd)
        FlextContainer.reset_for_testing()

    def test_public_settings_helpers_resolve_env_override_and_bootstrap_context(
        self,
        tmp_path: Path,
    ) -> None:
        env_file = tmp_path / c.ENV_FILE_DEFAULT
        env_file.write_text("FLEXT_APP_NAME=test-app\n", encoding="utf-8")
        probe_env_var = "FLEXT_TEST_BOOTSTRAP_MODE"
        os.environ[c.ENV_FILE_ENV_VAR] = str(env_file)
        os.environ[probe_env_var] = "integration"
        try:
            snapshot = m.Tests.BootstrapSnapshot(
                env_file=u.resolve_env_file(),
                process_environment=u.resolve_process_environment(),
                log_level=str(
                    u.resolve_effective_log_level(
                        trace=True,
                        debug=False,
                        log_level=c.LogLevel.ERROR,
                    ),
                ),
            )
        finally:
            os.environ.pop(probe_env_var, None)

        assert snapshot.env_file == str(env_file.resolve())
        assert snapshot.process_environment[c.ENV_FILE_ENV_VAR] == str(env_file)
        assert snapshot.process_environment[probe_env_var] == "integration"
        assert snapshot.log_level == c.LogLevel.DEBUG


class TestsFlextUtilitiesSettingsEnvFile(TestsFlextUtilitiesSettings):
    def test_public_settings_helpers_resolve_cwd_env_and_fallback_paths(
        self,
        tmp_path: Path,
    ) -> None:
        os.chdir(tmp_path)
        default_env_file = tmp_path / c.ENV_FILE_DEFAULT
        default_env_file.write_text("FLEXT_DEBUG=true\n", encoding="utf-8")

        cwd_resolved = u.resolve_env_file()
        explicit_level = u.resolve_effective_log_level(
            trace=False,
            debug=False,
            log_level=c.LogLevel.ERROR,
        )
        debug_level = u.resolve_effective_log_level(
            trace=False,
            debug=True,
            log_level=c.LogLevel.WARNING,
        )

        default_env_file.unlink()
        missing_override = str(tmp_path / "missing.env")
        os.environ[c.ENV_FILE_ENV_VAR] = missing_override

        assert cwd_resolved == str(default_env_file.resolve())
        assert explicit_level == c.LogLevel.ERROR
        assert debug_level == c.LogLevel.INFO
        assert u.resolve_env_file() == missing_override

        os.environ.pop(c.ENV_FILE_ENV_VAR, None)
        assert u.resolve_env_file() == c.ENV_FILE_DEFAULT


class TestsFlextUtilitiesSettingsRegisterFactory(TestsFlextUtilitiesSettings):
    def test_public_settings_helpers_register_factory_success_and_failure(self) -> None:
        container = FlextContainer()
        container.clear()

        def build_settings_summary() -> t.RegisterableService:
            return {
                "env_file": u.resolve_env_file(),
                "log_level": str(
                    u.resolve_effective_log_level(
                        trace=False,
                        debug=True,
                        log_level=c.LogLevel.WARNING,
                    ),
                ),
            }

        success_result = u.register_factory(
            container,
            "settings_summary",
            build_settings_summary,
        )
        resolved_summary = container.resolve("settings_summary")

        error_message = "factory exploded"

        def failing_factory() -> t.RegisterableService:
            raise RuntimeError(error_message)

        failure_result = u.register_factory(
            container,
            "broken_settings_summary",
            failing_factory,
        )

        tm.ok(success_result)
        assert success_result.value is True
        tm.ok(resolved_summary)
        assert resolved_summary.value == {
            "env_file": c.ENV_FILE_DEFAULT,
            "log_level": c.LogLevel.INFO,
        }
        tm.fail(failure_result)
        assert failure_result.error is not None
        assert "factory exploded" in failure_result.error
