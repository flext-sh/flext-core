"""Applied log levels govern real emission, including already-cached loggers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import pytest
import structlog
from flext_tests import tm

from tests import c, u

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from tests import p


class TestsFlextCoreLoggingThreshold:
    """Tests for ``FlextCoreLoggingThreshold``."""

    @staticmethod
    @pytest.fixture(autouse=True)
    def restore_default_level() -> Generator[None]:
        """Leave the process-wide threshold at the settings default afterwards."""
        yield
        u.apply_log_level(log_level=c.LogLevel.INFO, debug=False, trace=False)

    @staticmethod
    def captured(emit: Callable[[], p.Result[bool]], token: str) -> str:
        """Run ``emit`` and return stdout once ``token`` shows up or time runs out.

        Returns:
            The resulting ``str``.

        """
        stream = io.StringIO()
        with redirect_stdout(stream):
            _ = emit()
            deadline = time.monotonic() + 0.25
            while time.monotonic() < deadline and token not in stream.getvalue():
                time.sleep(0.01)
        return stream.getvalue()

    def test_cached_logger_honours_level_applied_after_creation(self) -> None:
        """Test cached logger honours level applied after creation."""
        logger = u.create_module_logger("threshold.cached")
        _ = logger.info("warm the logger cache")

        u.apply_log_level(log_level=c.LogLevel.WARNING, debug=False, trace=False)

        info_out = self.captured(
            lambda: logger.info("info-under-warning"),
            "info-under-warning",
        )
        warning_out = self.captured(
            lambda: logger.warning("warning-at-warning"),
            "warning-at-warning",
        )
        tm.that("info-under-warning" in info_out, eq=False)
        tm.that("warning-at-warning" in warning_out, eq=True)

    def test_lowering_the_level_reenables_debug_output(self) -> None:
        """Test lowering the level reenables debug output."""
        logger = u.create_module_logger("threshold.debug")

        u.apply_log_level(log_level=c.LogLevel.DEBUG, debug=False, trace=False)

        debug_out = self.captured(
            lambda: logger.debug("debug-at-debug"),
            "debug-at-debug",
        )
        tm.that("debug-at-debug" in debug_out, eq=True)

    @staticmethod
    @pytest.mark.parametrize(
        ("debug", "trace", "event_level", "dropped"),
        [
            (False, False, "info", True),
            (True, False, "info", False),
            (True, False, "debug", True),
            (True, True, "debug", False),
        ],
    )
    def test_debug_and_trace_resolve_through_the_runtime_owner(
        *,
        debug: bool,
        trace: bool,
        event_level: str,
        dropped: bool,
    ) -> None:
        """Test debug and trace resolve through the runtime owner."""
        u.apply_log_level(log_level=c.LogLevel.ERROR, debug=debug, trace=trace)
        event = {"level": event_level, "event": "probe"}

        if dropped:
            with pytest.raises(structlog.DropEvent):
                _ = u.drop_below_threshold(None, event_level, event)
        else:
            tm.that(u.drop_below_threshold(None, event_level, event), eq=event)

    @staticmethod
    def test_unknown_level_name_fails_loud() -> None:
        """Test unknown level name fails loud."""
        with pytest.raises(ValueError, match="LOUD"):
            u.apply_log_level(log_level="LOUD", debug=False, trace=False)
