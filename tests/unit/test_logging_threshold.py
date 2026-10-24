"""Applied log levels govern real emission, including already-cached loggers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr
from typing import TYPE_CHECKING

import pytest

from flext_core import FlextSettings
from tests import c, m, tm, u

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

from tests import p


class TestsFlextCoreLoggingThreshold:
    """Tests for ``FlextCoreLoggingThreshold``."""

    pytestmark = pytest.mark.usefixtures("restore_default_level")

    @staticmethod
    @pytest.fixture
    def restore_default_level() -> Generator[None]:
        """Restore the captured application policy, never today's defaults."""
        original_settings, original_options = u.logging_settings(), u.logging_options()
        try:
            u.configure_structlog(settings=m.StructlogOptions(async_logging=False))
            yield
        finally:
            u.configure_structlog(
                settings=original_options, runtime_settings=original_settings
            )

    @staticmethod
    def captured(emit: Callable[[], p.Result[bool]]) -> str:
        """Capture real acknowledged emission without sleep or polling.

        Returns:
            The resulting ``str``.

        """
        stream = io.StringIO()
        with redirect_stderr(stream):
            _ = emit()
        return stream.getvalue()

    def test_cached_logger_honours_level_applied_after_creation(self) -> None:
        """Test cached logger honours level applied after creation."""
        logger = u.create_module_logger("threshold.cached")
        _ = logger.info("warm the logger cache")

        u.apply_log_level(log_level=c.LogLevel.WARNING, debug=False, trace=False)

        info_out = self.captured(
            lambda: logger.info("info-under-warning"),
        )
        warning_out = self.captured(
            lambda: logger.warning("warning-at-warning"),
        )
        tm.fail("info-under-warning" in info_out)
        tm.ok("warning-at-warning" in warning_out)

    def test_lowering_the_level_reenables_debug_output(self) -> None:
        """Test lowering the level reenables debug output."""
        logger = u.create_module_logger("threshold.debug")

        u.apply_log_level(log_level=c.LogLevel.DEBUG, debug=False, trace=False)

        debug_out = self.captured(
            lambda: logger.debug("debug-at-debug"),
        )
        tm.ok("debug-at-debug" in debug_out)

    @staticmethod
    @pytest.mark.parametrize(
        ("debug", "trace"),
        [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    def test_debug_and_trace_resolve_through_the_runtime_owner(
        *,
        debug: bool,
        trace: bool,
    ) -> None:
        """Explicit ERROR always wins shorthand; trace is independent detail."""
        u.apply_log_level(log_level=c.LogLevel.ERROR, debug=debug, trace=trace)
        output = io.StringIO()
        with redirect_stderr(output):
            tm.ok(u.debug("filtered debug"))
            tm.ok(u.failure("retained error"))
        tm.that(output.getvalue(), lacks="filtered debug", has="retained error")

    @staticmethod
    def test_debug_shorthand_selects_debug_and_omissions_preserve_it() -> None:
        """The real public utility observes the elected application settings."""
        current = FlextSettings.fetch_global().clone(log_level=c.LogLevel.ERROR)
        selected = u.resolve_logging_settings(runtime_settings=current, debug=True)
        tm.that(selected.log_level, eq=c.LogLevel.DEBUG)
        tm.that(current.log_level, eq=c.LogLevel.ERROR)
        u.configure_structlog(runtime_settings=selected)
        u.apply_log_level()
        output = io.StringIO()
        with redirect_stderr(output):
            tm.ok(u.debug("actual debug %s", "value"))
        tm.that(output.getvalue(), has="actual debug value")
        tm.that(u.logging_settings().log_level, eq=selected.log_level)

    @staticmethod
    def test_trace_alone_does_not_lower_the_elected_threshold() -> None:
        """Trace adds stack detail without permitting otherwise filtered events."""
        current = FlextSettings.fetch_global().clone(
            log_level=c.LogLevel.ERROR, debug=False
        )
        selected = u.resolve_logging_settings(runtime_settings=current, trace=True)
        tm.that(selected.log_level, eq=current.log_level)
        tm.ok(selected.trace)

    @staticmethod
    def test_unknown_level_name_fails_loud() -> None:
        """Test unknown level name fails loud."""
        with pytest.raises(ValueError, match="LOUD"):
            u.apply_log_level(log_level="LOUD", debug=False, trace=False)
