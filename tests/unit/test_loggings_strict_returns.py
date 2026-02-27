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

from flext_core import p, r
from flext_core.loggings import FlextLogger


class TestDebugReturnsResultBool:
    """Test that debug() returns r[bool]."""

    def test_debug_returns_result_bool(self) -> None:
        """Verify debug() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.debug("test debug message")

        assert isinstance(result, r), f"Expected r[bool], got {type(result)}"
        assert result.is_success, "Expected debug() to return success result"
        assert result.value is True, "Expected debug() result value to be True"

    def test_debug_with_kwargs_returns_result_bool(self) -> None:
        """Verify debug() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.debug("test message", user_id="123", action="login")

        assert isinstance(result, r)
        assert result.is_success
        assert result.value is True


class TestInfoReturnsResultBool:
    """Test that info() returns r[bool]."""

    def test_info_returns_result_bool(self) -> None:
        """Verify info() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.info("test info message")

        assert isinstance(result, r), f"Expected r[bool], got {type(result)}"
        assert result.is_success, "Expected info() to return success result"
        assert result.value is True, "Expected info() result value to be True"

    def test_info_with_kwargs_returns_result_bool(self) -> None:
        """Verify info() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.info("test message", request_id="abc-123", status="ok")

        assert isinstance(result, r)
        assert result.is_success
        assert result.value is True


class TestWarningReturnsResultBool:
    """Test that warning() returns r[bool]."""

    def test_warning_returns_result_bool(self) -> None:
        """Verify warning() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.warning("test warning message")

        assert isinstance(result, r), f"Expected r[bool], got {type(result)}"
        assert result.is_success, "Expected warning() to return success result"
        assert result.value is True, "Expected warning() result value to be True"

    def test_warning_with_kwargs_returns_result_bool(self) -> None:
        """Verify warning() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.warning("test message", threshold=100, actual=150)

        assert isinstance(result, r)
        assert result.is_success
        assert result.value is True


class TestErrorReturnsResultBool:
    """Test that error() returns r[bool]."""

    def test_error_returns_result_bool(self) -> None:
        """Verify error() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.error("test error message")

        assert isinstance(result, r), f"Expected r[bool], got {type(result)}"
        assert result.is_success, "Expected error() to return success result"
        assert result.value is True, "Expected error() result value to be True"

    def test_error_with_kwargs_returns_result_bool(self) -> None:
        """Verify error() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.error("test message", error_code="ERR_001", details="failed")

        assert isinstance(result, r)
        assert result.is_success
        assert result.value is True


class TestCriticalReturnsResultBool:
    """Test that critical() returns r[bool]."""

    def test_critical_returns_result_bool(self) -> None:
        """Verify critical() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.critical("test critical message")

        assert isinstance(result, r), f"Expected r[bool], got {type(result)}"
        assert result.is_success, "Expected critical() to return success result"
        assert result.value is True, "Expected critical() result value to be True"

    def test_critical_with_kwargs_returns_result_bool(self) -> None:
        """Verify critical() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.critical("test message", severity="CRITICAL", action="shutdown")

        assert isinstance(result, r)
        assert result.is_success
        assert result.value is True


class TestExceptionReturnsResultBool:
    """Test that exception() returns r[bool]."""

    def test_exception_returns_result_bool(self) -> None:
        """Verify exception() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.exception("test exception message")  # noqa: LOG004

        assert isinstance(result, r), f"Expected r[bool], got {type(result)}"
        assert result.is_success, "Expected exception() to return success result"
        assert result.value is True, "Expected exception() result value to be True"

    def test_exception_with_kwargs_returns_result_bool(self) -> None:
        """Verify exception() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.exception("test message", operation="sync", retry_count=3)  # noqa: LOG004

        assert isinstance(result, r)
        assert result.is_success
        assert result.value is True


class TestTraceReturnsResultBool:
    """Test that trace() returns r[bool]."""

    def test_trace_returns_result_bool(self) -> None:
        """Verify trace() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.trace("test trace message")

        assert isinstance(result, r), f"Expected r[bool], got {type(result)}"
        assert result.is_success, "Expected trace() to return success result"
        assert result.value is True, "Expected trace() result value to be True"

    def test_trace_with_kwargs_returns_result_bool(self) -> None:
        """Verify trace() with kwargs returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.trace("test message", depth=5, scope="operation")

        assert isinstance(result, r)
        assert result.is_success
        assert result.value is True


class TestLogReturnsResultBool:
    """Test that log() returns r[bool]."""

    def test_log_returns_result_bool(self) -> None:
        """Verify log() returns r[bool] with is_success=True."""
        logger = FlextLogger.create_module_logger(__name__)
        result = logger.log("info", "test log message")

        assert isinstance(result, r), f"Expected r[bool], got {type(result)}"
        assert result.is_success, "Expected log() to return success result"
        assert result.value is True, "Expected log() result value to be True"

    def test_log_with_context_returns_result_bool(self) -> None:
        """Verify log() with context returns r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)
        context = {"request_id": "req-123", "user": "alice"}
        result = logger.log("debug", "test message", _context=context)

        assert isinstance(result, r)
        assert result.is_success
        assert result.value is True

    def test_log_with_different_levels_returns_result_bool(self) -> None:
        """Verify log() works with different log levels."""
        logger = FlextLogger.create_module_logger(__name__)

        for level in ["debug", "info", "warning", "error", "critical"]:
            result = logger.log(level, f"test {level} message")
            assert isinstance(result, r), f"log({level}) should return r[bool]"
            assert result.is_success, f"log({level}) should return success"
            assert result.value is True, f"log({level}) result value should be True"


class TestBackwardCompatDiscardReturnValue:
    """Test backward compatibility: code that discards return value still works."""

    def test_debug_discard_return_value(self) -> None:
        """Verify debug() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        # This should not raise an exception
        logger.debug("test message")  # type: ignore[func-returns-value]

    def test_info_discard_return_value(self) -> None:
        """Verify info() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.info("test message")  # type: ignore[func-returns-value]

    def test_warning_discard_return_value(self) -> None:
        """Verify warning() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.warning("test message")  # type: ignore[func-returns-value]

    def test_error_discard_return_value(self) -> None:
        """Verify error() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.error("test message")  # type: ignore[func-returns-value]

    def test_critical_discard_return_value(self) -> None:
        """Verify critical() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.critical("test message")  # type: ignore[func-returns-value]

    def test_exception_discard_return_value(self) -> None:
        """Verify exception() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.exception("test message")  # type: ignore[func-returns-value]  # noqa: LOG004

    def test_trace_discard_return_value(self) -> None:
        """Verify trace() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.trace("test message")  # type: ignore[func-returns-value]

    def test_log_discard_return_value(self) -> None:
        """Verify log() works when return value is discarded."""
        logger = FlextLogger.create_module_logger(__name__)
        logger.log("info", "test message")  # type: ignore[func-returns-value]


class TestProtocolComplianceStructlogLogger:
    """Test protocol compliance: FlextLogger satisfies p.Log.StructlogLogger."""

    def test_flext_logger_implements_structlog_logger_protocol(self) -> None:
        """Verify FlextLogger implements p.Log.StructlogLogger protocol."""
        logger = FlextLogger.create_module_logger(__name__)

        # Check that logger is an instance of the protocol
        assert isinstance(logger, p.Log.StructlogLogger), (
            f"FlextLogger should implement p.Log.StructlogLogger protocol, "
            f"got {type(logger)}"
        )

    def test_all_protocol_methods_return_result_bool(self) -> None:
        """Verify all protocol methods return r[bool]."""
        logger = FlextLogger.create_module_logger(__name__)

        # Test all methods defined in p.Log.StructlogLogger
        protocol_methods = [
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "exception",
        ]

        for method_name in protocol_methods:
            method = getattr(logger, method_name, None)
            assert method is not None, f"Logger should have {method_name} method"
            assert callable(method), f"{method_name} should be callable"

            # Call method and verify return type
            result = method(f"test {method_name} message")
            assert isinstance(result, r), (
                f"{method_name}() should return r[bool], got {type(result)}"
            )
            assert result.is_success, f"{method_name}() should return success result"
            assert result.value is True, f"{method_name}() result value should be True"

    def test_protocol_method_signatures_match(self) -> None:
        """Verify method signatures match protocol definition."""
        logger = FlextLogger.create_module_logger(__name__)

        # Verify debug signature: msg, *args, **kw -> r[bool]
        result = logger.debug("msg", "arg1", key="value")  # noqa: PLE1205
        assert isinstance(result, r)

        # Verify info signature: msg, *args, **kw -> r[bool]
        result = logger.info("msg", "arg1", key="value")  # noqa: PLE1205
        assert isinstance(result, r)

        # Verify warning signature: msg, *args, **kw -> r[bool]
        result = logger.warning("msg", "arg1", key="value")  # noqa: PLE1205
        assert isinstance(result, r)

        # Verify error signature: msg, *args, **kw -> r[bool]
        result = logger.error("msg", "arg1", key="value")  # noqa: PLE1205
        assert isinstance(result, r)

        # Verify critical signature: msg, *args, **kw -> r[bool]
        result = logger.critical("msg", "arg1", key="value")  # noqa: PLE1205
        assert isinstance(result, r)

        # Verify exception signature: msg, *args, **kw -> r[bool]
        result = logger.exception("msg", "arg1", key="value")  # noqa: LOG004,PLE1205
        assert isinstance(result, r)
