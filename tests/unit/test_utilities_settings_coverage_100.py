"""Behavior contract for public settings helpers in bootstrap workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from flext_tests import tm

from flext_core import FlextContainer, u
from tests.constants import c
from tests.models import m

if TYPE_CHECKING:
    from tests.typings import t


class TestsFlextCoreUtilitiesSettings:
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

    @pytest.mark.parametrize(
        ("trace", "debug", "requested", "expected"),
        [
            (True, False, c.LogLevel.ERROR, c.LogLevel.DEBUG),
            (True, True, c.LogLevel.WARNING, c.LogLevel.DEBUG),
            (False, True, c.LogLevel.WARNING, c.LogLevel.INFO),
            (False, True, c.LogLevel.ERROR, c.LogLevel.INFO),
            (False, False, c.LogLevel.ERROR, c.LogLevel.ERROR),
            (False, False, c.LogLevel.WARNING, c.LogLevel.WARNING),
        ],
    )
    def test_effective_log_level_prioritises_trace_then_debug_then_request(
        self,
        *,
        trace: bool,
        debug: bool,
        requested: c.LogLevel,
        expected: c.LogLevel,
    ) -> None:
        resolved = u.resolve_effective_log_level(
            trace=trace,
            debug=debug,
            log_level=requested,
        )

        tm.that(resolved, eq=expected)

    def test_env_override_and_process_environment_are_observable(
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

        tm.that(snapshot.env_file, eq=str(env_file.resolve()))
        tm.that(snapshot.process_environment[c.ENV_FILE_ENV_VAR], eq=str(env_file))
        tm.that(snapshot.process_environment[probe_env_var], eq="integration")
        tm.that(snapshot.log_level, eq=c.LogLevel.DEBUG)

    def test_env_file_resolves_cwd_default_then_override_then_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        os.chdir(tmp_path)
        default_env_file = tmp_path / c.ENV_FILE_DEFAULT
        default_env_file.write_text("FLEXT_DEBUG=true\n", encoding="utf-8")

        cwd_resolved = u.resolve_env_file()

        default_env_file.unlink()
        missing_override = str(tmp_path / "missing.env")
        os.environ[c.ENV_FILE_ENV_VAR] = missing_override

        tm.that(cwd_resolved, eq=str(default_env_file.resolve()))
        tm.that(u.resolve_env_file(), eq=missing_override)

        os.environ.pop(c.ENV_FILE_ENV_VAR, None)
        tm.that(u.resolve_env_file(), eq=c.ENV_FILE_DEFAULT)

    def test_register_factory_reports_success_and_resolvable_service(self) -> None:
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

        tm.ok(success_result)
        assert success_result.value is True
        tm.ok(resolved_summary)
        tm.that(resolved_summary.value, eq={
            "env_file": c.ENV_FILE_DEFAULT,
            "log_level": c.LogLevel.INFO,
        })

    def test_register_factory_surfaces_factory_failure_as_result(self) -> None:
        container = FlextContainer()
        container.clear()
        error_message = "factory exploded"

        def failing_factory() -> t.RegisterableService:
            raise RuntimeError(error_message)

        failure_result = u.register_factory(
            container,
            "broken_settings_summary",
            failing_factory,
        )

        tm.fail(failure_result)
        assert failure_result.error is not None
        assert error_message in failure_result.error
