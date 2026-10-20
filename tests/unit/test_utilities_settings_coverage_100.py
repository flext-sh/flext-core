"""Behavior contract for public settings helpers in bootstrap workflows.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Never

import pytest

from flext_core import FlextContainer, FlextSettings
from tests import c, m, tm, u


class TestsFlextCoreUtilitiesSettings:
    """Tests for ``FlextCoreUtilitiesSettings``."""

    _original_cwd: Path

    def setup_method(self) -> None:
        """Provide ``setup_method``."""
        self._original_cwd = Path.cwd()
        FlextContainer.reset_for_testing()

    def teardown_method(self) -> None:
        """Provide ``teardown_method``."""
        os.chdir(self._original_cwd)
        FlextContainer.reset_for_testing()

    @pytest.mark.parametrize(
        ("trace", "debug", "requested", "expected"),
        [
            (True, False, c.LogLevel.ERROR, c.LogLevel.ERROR),
            (True, True, c.LogLevel.WARNING, c.LogLevel.WARNING),
            (False, True, c.LogLevel.WARNING, c.LogLevel.WARNING),
            (False, True, c.LogLevel.ERROR, c.LogLevel.ERROR),
            (False, False, c.LogLevel.ERROR, c.LogLevel.ERROR),
            (False, False, c.LogLevel.WARNING, c.LogLevel.WARNING),
        ],
    )
    @staticmethod
    def test_effective_log_level_preserves_explicit_severity(
        *,
        trace: bool,
        debug: bool,
        requested: c.LogLevel,
        expected: c.LogLevel,
    ) -> None:
        """Explicit severity wins debug; trace never changes the threshold."""
        resolved = u.resolve_effective_log_level(
            trace=trace,
            debug=debug,
            log_level=requested,
        )

        tm.that(resolved, eq=expected)

    @staticmethod
    def test_env_override_and_process_environment_are_observable(
        tmp_path: Path,
    ) -> None:
        """Test env override and process environment are observable."""
        env_file = tmp_path / c.ENV_FILE_DEFAULT
        env_file.write_text("FLEXT_APP_NAME=test-app\n", encoding="utf-8")
        probe_env_var = "FLEXT_TEST_BOOTSTRAP_MODE"
        with u.Tests.env_vars_context(
            env_vars={c.ENV_FILE_ENV_VAR: str(env_file), probe_env_var: "integration"},
        ):
            snapshot = m.Tests.BootstrapSnapshot(
                env_file=FlextSettings.resolve_env_file(),
                process_environment=u.resolve_process_environment(),
                log_level=str(
                    u.resolve_effective_log_level(
                        trace=True,
                        debug=False,
                        log_level=c.LogLevel.ERROR,
                    ),
                ),
            )

        tm.that(snapshot.env_file, eq=str(env_file.resolve()))
        tm.that(snapshot.process_environment[c.ENV_FILE_ENV_VAR], eq=str(env_file))
        tm.that(snapshot.process_environment[probe_env_var], eq="integration")
        tm.that(snapshot.log_level, eq=c.LogLevel.ERROR)

    @staticmethod
    def test_env_file_resolves_cwd_default_then_override_then_fallback(
        tmp_path: Path,
    ) -> None:
        """Test env file resolves cwd default then override then fallback."""
        os.chdir(tmp_path)
        default_env_file = tmp_path / c.ENV_FILE_DEFAULT
        default_env_file.write_text("FLEXT_DEBUG=true\n", encoding="utf-8")
        missing_override = str(tmp_path / "missing.env")

        with u.Tests.env_vars_context(vars_to_clear=[c.ENV_FILE_ENV_VAR]):
            cwd_resolved = FlextSettings.resolve_env_file()
            default_env_file.unlink()
            with u.Tests.env_vars_context(
                env_vars={c.ENV_FILE_ENV_VAR: missing_override},
            ):
                override_resolved = FlextSettings.resolve_env_file()
            fallback_resolved = FlextSettings.resolve_env_file()

        tm.that(cwd_resolved, eq=str(default_env_file.resolve()))
        tm.that(override_resolved, eq=missing_override)
        tm.that(fallback_resolved, eq=c.ENV_FILE_DEFAULT)

    @staticmethod
    def test_register_factory_reports_success_and_resolvable_service() -> None:
        """Test register factory reports success and resolvable service."""
        container = FlextContainer()
        container.clear()
        expected_env_file = FlextSettings.resolve_env_file()
        expected_summary = m.ConfigMap(
            root={
                "env_file": expected_env_file,
                "log_level": u.resolve_effective_log_level(
                    trace=False,
                    debug=True,
                    log_level=c.LogLevel.WARNING,
                ),
            }
        )

        success_result = u.register_factory(
            container,
            "settings_summary",
            lambda: expected_summary,
        )
        resolved_summary = container.resolve("settings_summary", type_cls=m.ConfigMap)

        tm.ok(success_result)
        tm.that(success_result.value, eq=True)
        tm.ok(resolved_summary)
        summary = resolved_summary.value
        tm.that(summary is expected_summary, eq=True)
        tm.that(
            summary.root,
            eq={
                "env_file": expected_env_file,
                "log_level": c.LogLevel.WARNING,
            },
        )

    @staticmethod
    def test_register_factory_surfaces_factory_failure_as_result() -> None:
        """Test register factory surfaces factory failure as result."""
        container = FlextContainer()
        container.clear()
        error_message = "factory exploded"

        def failing_factory() -> Never:
            raise RuntimeError(error_message)

        failure_result = u.register_factory(
            container,
            "broken_settings_summary",
            failing_factory,
        )

        tm.fail(failure_result)
        tm.that(failure_result.error, none=False)
        tm.that(error_message in str(failure_result.error), truth=True)
