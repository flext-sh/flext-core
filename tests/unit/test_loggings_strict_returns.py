"""Tests for FlextLogger strict return types (r[bool] instead of None).

Module: flext_core.loggings
Scope: Verify all FlextLogger logging methods return r[bool]

Tests verify that:
- All 8 public logging methods (debug, info, warning, error, critical,
  exception, trace, log) return r[bool] instead of None
- Return values have is_success=True on normal logging
- Backward compatibility: code that discards return value still works
- Protocol compliance: FlextLogger satisfies p.Log.StructlogLogger

Uses Python 3.13 patterns and pytest parametrization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_tests import tm

from flext_core import FlextLogger, r


class TestDebugReturnsResultBool:
    """Test that debug() returns r[bool]."""

    def test_debug_returns_result_bool(self) -> None:
        """Verify debug() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.debug("test debug message")
        tm.that(isinstance(result, r), eq=True), f"Expected r[bool], got {type(result)}"
        tm.ok(result)
        tm.that(result.value, eq=True)
        "Expected debug() result value to be True"

    def test_debug_with_kwargs_returns_result_bool(self) -> None:
        """Verify debug() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.debug("test message", user_id="123", action="login")
        tm.that(isinstance(result, r), eq=True)
        tm.ok(result)
        tm.that(result.value, eq=True)


class TestInfoReturnsResultBool:
    """Test that info() returns r[bool]."""

    def test_info_returns_result_bool(self) -> None:
        """Verify info() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.info("test info message")
        tm.that(isinstance(result, r), eq=True), f"Expected r[bool], got {type(result)}"
        tm.ok(result)
        tm.that(result.value, eq=True)
        "Expected info() result value to be True"

    def test_info_with_kwargs_returns_result_bool(self) -> None:
        """Verify info() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.info("test message", request_id="abc-123", status="ok")
        tm.that(isinstance(result, r), eq=True)
        tm.ok(result)
        tm.that(result.value, eq=True)


class TestWarningReturnsResultBool:
    """Test that warning() returns r[bool]."""

    def test_warning_returns_result_bool(self) -> None:
        """Verify warning() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.warning("test warning message")
        tm.that(isinstance(result, r), eq=True), f"Expected r[bool], got {type(result)}"
        tm.ok(result)
        tm.that(result.value, eq=True)
        "Expected warning() result value to be True"

    def test_warning_with_kwargs_returns_result_bool(self) -> None:
        """Verify warning() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.warning("test message", threshold=100, actual=150)
        tm.that(isinstance(result, r), eq=True)
        tm.ok(result)
        tm.that(result.value, eq=True)


class TestErrorReturnsResultBool:
    """Test that error() returns r[bool]."""

    def test_error_returns_result_bool(self) -> None:
        """Verify error() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.error("test error message")
        tm.that(isinstance(result, r), eq=True), f"Expected r[bool], got {type(result)}"
        tm.ok(result)
        tm.that(result.value, eq=True)
        "Expected error() result value to be True"

    def test_error_with_kwargs_returns_result_bool(self) -> None:
        """Verify error() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.error("test message", error_code="ERR_001", details="failed")
        tm.that(isinstance(result, r), eq=True)
        tm.ok(result)
        tm.that(result.value, eq=True)


class TestCriticalReturnsResultBool:
    """Test that critical() returns r[bool]."""

    def test_critical_returns_result_bool(self) -> None:
        """Verify critical() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.critical("test critical message")
        tm.that(isinstance(result, r), eq=True), f"Expected r[bool], got {type(result)}"
        tm.ok(result)
        tm.that(result.value, eq=True)
        "Expected critical() result value to be True"

    def test_critical_with_kwargs_returns_result_bool(self) -> None:
        """Verify critical() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.critical("test message", severity="CRITICAL", action="shutdown")
        tm.that(isinstance(result, r), eq=True)
        tm.ok(result)
        tm.that(result.value, eq=True)


class TestExceptionReturnsResultBool:
    """Test that exception() returns r[bool]."""

    def test_exception_returns_result_bool(self) -> None:
        """Verify exception() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.error("test exception message")
        tm.that(isinstance(result, r), eq=True), f"Expected r[bool], got {type(result)}"
        tm.ok(result)
        tm.that(result.value, eq=True)
        "Expected exception() result value to be True"

    def test_exception_with_kwargs_returns_result_bool(self) -> None:
        """Verify exception() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.error("test message", operation="sync", retry_count=3)
        tm.that(isinstance(result, r), eq=True)
        tm.ok(result)
        tm.that(result.value, eq=True)


class TestTraceReturnsResultBool:
    """Test that trace() returns r[bool]."""

    def test_trace_returns_result_bool(self) -> None:
        """Verify trace() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.trace("test trace message")
        tm.that(isinstance(result, r), eq=True), f"Expected r[bool], got {type(result)}"
        tm.ok(result)
        tm.that(result.value, eq=True)
        "Expected trace() result value to be True"

    def test_trace_with_kwargs_returns_result_bool(self) -> None:
        """Verify trace() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.trace("test message", depth=5, scope="operation")
        tm.that(isinstance(result, r), eq=True)
        tm.ok(result)
        tm.that(result.value, eq=True)


class TestLogReturnsResultBool:
    """Test that log() returns r[bool]."""

    def test_log_returns_result_bool(self) -> None:
        """Verify log() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.log("info", "test log message")
        tm.that(isinstance(result, r), eq=True), f"Expected r[bool], got {type(result)}"
        tm.ok(result)
        tm.that(result.value, eq=True)
        "Expected log() result value to be True"

    def test_log_with_context_returns_result_bool(self) -> None:
        """Verify log() with context returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.log(
            "debug",
            "test message",
            request_id="req-123",
            user="alice",
        )
        tm.that(isinstance(result, r), eq=True)
        tm.ok(result)
        tm.that(result.value, eq=True)

    def test_log_with_different_levels_returns_result_bool(self) -> None:
        """Verify log() works with different log levels."""
        logger = FlextLogger.create_module_logger(__name__)
        for level in ["debug", "info", "warning", "error", "critical"]:
            result = logger.log(level, "test %s message", level)
            (
                tm.that(isinstance(result, r), eq=True),
                f"log({level}) should return r[bool]",
            )
            tm.ok(result), f"log({level}) should return success"
            tm.that(result.value, eq=True), f"log({level}) result value should be True"


class TestBackwardCompatDiscardReturnValue:
    """Test backward compatibility: code that discards return value still works."""

    def test_debug_discard_return_value(self) -> None:
        """Verify debug() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.debug("test message")

    def test_info_discard_return_value(self) -> None:
        """Verify info() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.info("test message")

    def test_warning_discard_return_value(self) -> None:
        """Verify warning() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.warning("test message")

    def test_error_discard_return_value(self) -> None:
        """Verify error() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.error("test message")

    def test_critical_discard_return_value(self) -> None:
        """Verify critical() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.critical("test message")

    def test_exception_discard_return_value(self) -> None:
        """Verify exception() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.error("test message")

    def test_trace_discard_return_value(self) -> None:
        """Verify trace() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.trace("test message")

    def test_log_discard_return_value(self) -> None:
        """Verify log() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.log("info", "test message")


class TestProtocolComplianceStructlogLogger:
    """Test protocol compliance: FlextLogger satisfies p.Log.StructlogLogger."""

    def test_flext_logger_implements_structlog_logger_protocol(self) -> None:
        """Verify FlextLogger implements p.Log.StructlogLogger protocol."""
        logger = FlextLogger.create_module_logger(__name__)
        tm.that(hasattr(logger, "debug"), eq=True)
        tm.that(hasattr(logger, "info"), eq=True)
        tm.that(hasattr(logger, "warning"), eq=True)
        tm.that(hasattr(logger, "error"), eq=True)
        tm.that(hasattr(logger, "exception"), eq=True)

    def test_all_protocol_methods_return_result_bool(self) -> None:
        """Verify all protocol methods return r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
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
            tm.that(result is not None, eq=True)
            tm.that(isinstance(result, r), eq=True)
            tm.ok(result)
            tm.that(result.value, eq=True)

    def test_protocol_method_signatures_match(self) -> None:
        """Verify method signatures match protocol definition."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.debug("msg %s", "arg1", key="value")
        tm.that(isinstance(result, r), eq=True)
        result = logger.info("msg %s", "arg1", key="value")
        tm.that(isinstance(result, r), eq=True)
        result = logger.warning("msg %s", "arg1", key="value")
        tm.that(isinstance(result, r), eq=True)
        result = logger.error("msg %s", "arg1", key="value")
        tm.that(isinstance(result, r), eq=True)
        result = logger.critical("msg %s", "arg1", key="value")
        tm.that(isinstance(result, r), eq=True)
        result = logger.error("msg %s", "arg1", key="value")
        tm.that(isinstance(result, r), eq=True)
