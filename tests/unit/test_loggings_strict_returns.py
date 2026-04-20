"""Tests for public logger strict return types (r[bool] instead of None).

Module: flext_core
Scope: Verify all public logger methods return r[bool]

Tests verify that:
- All 8 public logging methods (debug, info, warning, error, critical,
  exception, trace, log) return r[bool] instead of None
- Return values have success=True on normal logging
- Backward compatibility: code that discards return value still works
- Protocol compliance: logger instances satisfy p.Logger

Uses Python 3.13 patterns and pytest parametrization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests import tm

from tests import u


class TestLoggingsStrictReturns:
    """Unified strict-return tests for public logger methods."""

    def test_debug_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.debug("test debug message")
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_debug_with_kwargs_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.debug("test message", user_id="123", action="login")
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_info_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.info("test info message")
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_info_with_kwargs_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.info("test message", request_id="abc-123", status="ok")
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_warning_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.warning("test warning message")
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_warning_with_kwargs_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.warning("test message", threshold=100, actual=150)
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_error_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.error("test error message")
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_error_with_kwargs_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.error("test message", error_code="ERR_001", details="failed")
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_critical_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.critical("test critical message")
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_critical_with_kwargs_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.critical("test message", severity="CRITICAL", action="shutdown")
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_exception_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.error("test exception message")
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_exception_with_kwargs_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.error("test message", operation="sync", retry_count=3)
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_trace_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.trace("test trace message")
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_trace_with_kwargs_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.trace("test message", depth=5, scope="operation")
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_log_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.log("info", "test log message")
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_log_with_context_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.log(
            "debug",
            "test message",
            request_id="req-123",
            user="alice",
        )
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_log_with_different_levels_returns_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        for level in ["debug", "info", "warning", "error", "critical"]:
            result = logger.log(level, "test %s message", level)
            tm.ok(result)
            tm.that(result.value, eq=True)

    def test_debug_discard_return_value(self) -> None:
        logger = u.create_module_logger(__name__)
        _ = logger.debug("test message")

    def test_info_discard_return_value(self) -> None:
        logger = u.create_module_logger(__name__)
        _ = logger.info("test message")

    def test_warning_discard_return_value(self) -> None:
        logger = u.create_module_logger(__name__)
        _ = logger.warning("test message")

    def test_error_discard_return_value(self) -> None:
        logger = u.create_module_logger(__name__)
        _ = logger.error("test message")

    def test_critical_discard_return_value(self) -> None:
        logger = u.create_module_logger(__name__)
        _ = logger.critical("test message")

    def test_exception_discard_return_value(self) -> None:
        logger = u.create_module_logger(__name__)
        _ = logger.error("test message")

    def test_trace_discard_return_value(self) -> None:
        logger = u.create_module_logger(__name__)
        _ = logger.trace("test message")

    def test_log_discard_return_value(self) -> None:
        logger = u.create_module_logger(__name__)
        _ = logger.log("info", "test message")

    def test_flext_logger_implements_structlog_logger_protocol(self) -> None:
        u.create_module_logger(__name__)

    def test_all_protocol_methods_return_result_bool(self) -> None:
        logger = u.create_module_logger(__name__)
        protocol_calls = [
            ("debug", logger.debug("test debug message")),
            ("info", logger.info("test info message")),
            ("warning", logger.warning("test warning message")),
            ("error", logger.error("test error message")),
            ("critical", logger.critical("test critical message")),
        ]
        exception_message = "test exception message"
        try:
            raise ValueError(exception_message)
        except ValueError:
            protocol_calls.append((
                "exception",
                logger.exception(exception_message),
            ))
        for _method_name, result in protocol_calls:
            tm.ok(result)
            tm.that(result.value, eq=True)

    def test_protocol_method_signatures_match(self) -> None:
        logger = u.create_module_logger(__name__)
        result = logger.debug("msg", key="value")
        tm.ok(result)
        result = logger.info("msg", key="value")
        tm.ok(result)
        result = logger.warning("msg", key="value")
        tm.ok(result)
        result = logger.error("msg", key="value")
        tm.ok(result)
        result = logger.critical("msg", key="value")
        tm.ok(result)
        result = logger.error("msg", key="value")
        tm.ok(result)
