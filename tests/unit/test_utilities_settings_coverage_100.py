"""Behavior contract for public settings helpers in bootstrap workflows.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

import os
from pathlib import Path

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

    @staticmethod
    @pytest.mark.parametrize(
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
