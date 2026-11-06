"""Behavior contract for public settings helpers in bootstrap workflows.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import ValidationError

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

    @staticmethod
    @c.Tests.MARK.parametrize(
        ("debug", "requested", "log_level_explicit", "expected"),
        [
            (True, c.LogLevel.WARNING, True, c.LogLevel.WARNING),
            (True, c.LogLevel.ERROR, True, c.LogLevel.ERROR),
            (False, c.LogLevel.ERROR, True, c.LogLevel.ERROR),
            (False, c.LogLevel.WARNING, True, c.LogLevel.WARNING),
            (True, c.LogLevel.WARNING, False, c.LogLevel.DEBUG),
        ],
    )
    def test_effective_log_level_preserves_explicit_severity(
        *,
        debug: bool,
        requested: c.LogLevel,
        log_level_explicit: bool,
        expected: c.LogLevel,
    ) -> None:
        """Explicit severity wins debug; only a non-explicit one yields to debug."""
        resolved = u.resolve_effective_log_level(
            debug=debug,
            log_level=requested,
            log_level_explicit=log_level_explicit,
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
    def test_env_file_resolves_only_explicit_and_scoped_sources(
        tmp_path: Path,
    ) -> None:
        """No implicit cwd dotenv: only ``FLEXT_ENV_FILE`` or a scoped file."""
        os.chdir(tmp_path)
        explicit_env_file = tmp_path / "explicit.env"
        explicit_env_file.write_text("FLEXT_DEBUG=true\n", encoding="utf-8")
        missing_override = str(tmp_path / "missing.env")
        scoped_name = ".env.flext-probe"
        scoped_env_file = tmp_path / scoped_name
        scoped_env_file.write_text("FLEXT_TRACE=true\n", encoding="utf-8")

        with u.Tests.env_vars_context(vars_to_clear=[c.ENV_FILE_ENV_VAR]):
            unset_resolved = FlextSettings.resolve_env_file()
            scoped_resolved = FlextSettings.resolve_env_file("probe")
            with u.Tests.env_vars_context(
                env_vars={c.ENV_FILE_ENV_VAR: str(explicit_env_file)},
            ):
                explicit_resolved = FlextSettings.resolve_env_file()
            with u.Tests.env_vars_context(
                env_vars={c.ENV_FILE_ENV_VAR: missing_override},
            ):
                override_resolved = FlextSettings.resolve_env_file()

        tm.that(unset_resolved, eq="")
        tm.that(scoped_resolved, eq=str(scoped_env_file.resolve()))
        tm.that(explicit_resolved, eq=str(explicit_env_file.resolve()))
        tm.that(override_resolved, eq=missing_override)

    @staticmethod
    def test_validation_error_does_not_echo_input_values() -> None:
        """A malformed setting fails typed without rendering the input value."""
        sentinel = "sentinel-bool-violation-secret"
        with (
            u.Tests.env_vars_context(env_vars={"FLEXT_DEBUG": sentinel}),
            u.Tests.raises(ValidationError) as caught,
        ):
            FlextSettings()

        tm.that(sentinel not in str(caught.value), eq=True)
