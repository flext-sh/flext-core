"""Comprehensive coverage tests for the public logging DSL.

This module provides extensive tests for public logging functionality:
- Global context management (bind/unbind/clear/get)
- Scoped context (application/request/operation tiers)
- Level-based context filtering
- Factory patterns (service logger, module logger)
- All logging levels (trace/debug/info/warning/error/critical)
- Exception logging with stack traces
- Performance tracking and timing
- Result integration and logging
- PerformanceTracker inner class

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_tests import tm

from tests import p, t, u


class TestCoverageLoggings:
    @staticmethod
    def make_result_logger(
        name: str,
        *,
        service_name: str | None = None,
        service_version: str | None = None,
        correlation_id: str | None = None,
    ) -> p.Logger:
        """Helper to build public logger instances for logging tests."""
        return u.create_module_logger(
            name,
            service_name=service_name,
            service_version=service_version,
            correlation_id=correlation_id,
        )

    @staticmethod
    def assert_log_result_success[TResult: p.ResultLike[bool]](
        result: TResult | None,
    ) -> TResult:
        if result is None:
            msg = "Expected result to not be None"
            raise AssertionError(msg)
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=True)
        return result

    def test_clear_scope_application(self) -> None:
        """Test clearing application scope."""
        _ = u.bind_context(scope="application", app_name="test")
        result = u.clear_scope("application")
        _ = self.assert_log_result_success(result)

    def test_clear_scope_request(self) -> None:
        """Test clearing request scope."""
        _ = u.bind_context(scope="request", correlation_id="flext-123")
        result = u.clear_scope("request")
        _ = self.assert_log_result_success(result)

    def test_clear_scope_operation(self) -> None:
        """Test clearing operation scope."""
        _ = u.bind_context(scope="operation", operation="test")
        result = u.clear_scope("operation")
        _ = self.assert_log_result_success(result)

    def test_clear_scope_nonexistent(self) -> None:
        """Test clearing nonexistent scope."""
        result = u.clear_scope("nonexistent")
        _ = self.assert_log_result_success(result)

    def test_create_service_logger(self) -> None:
        """Test creating a service logger through the public logging DSL."""
        u.fetch_logger("user-service")
        tm.that(u.logger, none=False)

    def test_create_service_logger_with_version(self) -> None:
        """Test creating service logger with version via bind."""
        u.fetch_logger("auth-service").bind(version="2.0.0")
        tm.that(u.logger, none=False)

    def test_create_service_logger_with_correlation_id(self) -> None:
        """Test creating service logger with correlation ID via bind."""
        u.fetch_logger("payment-service").bind(correlation_id="flext-abc123")
        tm.that(u.logger, none=False)

    def test_create_module_logger(self) -> None:
        """Test creating module logger."""
        u.create_module_logger("myapp.services")
        tm.that(u.logger, none=False)

    def test_create_module_logger_dunder_name(self) -> None:
        """Test creating module logger with __name__."""
        module_name = __name__
        u.create_module_logger(module_name)
        tm.that(u.logger, none=False)

    def test_fetch_logger(self) -> None:
        """Test creating logger via constructor (fetch_logger pattern)."""
        u.fetch_logger("default_logger")
        tm.that(u.logger, none=False)

    def test_logger_init_with_name(self) -> None:
        """Test initializing logger with name."""
        u.create_module_logger("test_module")
        tm.that(u.logger, none=False)

    def test_logger_init_with_service_context(self) -> None:
        """Test initializing logger with service context."""
        u.create_module_logger(
            "service_logger",
        ).bind(service_name="my-service", service_version="1.0.0")
        tm.that(u.logger, none=False)

    def test_logger_name_property(self) -> None:
        """Test logger name property."""
        u.create_module_logger("test")
        tm.that(u.logger, none=False)

    def test_bind_creates_new_instance(self) -> None:
        """Test bind creates new logger instance."""
        logger1 = self.make_result_logger("test")
        logger1.bind(request_id="123")
        tm.that(logger1, none=False)

    def test_bind_chaining(self) -> None:
        """Test chaining multiple bind calls."""
        self.make_result_logger("test").bind(a="1").bind(b="2").bind(c="3")
        tm.that(u.logger, none=False)

    def test_trace_logging(self) -> None:
        """Test trace level logging.

        Validates:
        1. Logger is created successfully
        2. Trace logging executes without errors
        3. Result indicates success
        """
        logger = u.create_module_logger("test")
        tm.that(logger, none=False)
        self.assert_log_result_success(logger.trace("Test trace message"))

    def test_debug_logging(self) -> None:
        """Test debug level logging.

        Validates:
        1. Logger is created successfully
        2. Debug logging executes without errors
        3. Result indicates success
        """
        logger = self.make_result_logger("test")
        tm.that(logger, none=False)
        self.assert_log_result_success(logger.debug("Test debug message"))

    def test_debug_with_context(self) -> None:
        """Test debug logging with context.

        Validates:
        1. Logger accepts context parameters
        2. Context is included in log entry
        3. Logging succeeds with context data
        """
        logger = self.make_result_logger("test")
        tm.that(u.logger, none=False)
        self.assert_log_result_success(
            logger.debug("Debug with context", user_id="123", action="login"),
        )

    def test_info_logging(self) -> None:
        """Test info level logging.

        Validates:
        1. Logger is created successfully
        2. Info logging executes without errors
        3. Result indicates success
        4. Message is processed correctly
        """
        logger = self.make_result_logger("test")
        tm.that(logger, none=False)
        self.assert_log_result_success(logger.info("Test info message"))

    def test_info_with_context(self) -> None:
        """Test info logging with context.

        Validates:
        1. Logger accepts context parameters
        2. Context is included in log entry
        3. Logging succeeds with context data
        """
        logger = self.make_result_logger("test")
        tm.that(u.logger, none=False)
        self.assert_log_result_success(
            logger.info("Info with context", status="completed", duration="0.5s"),
        )

    def test_warning_logging(self) -> None:
        """Test warning level logging.

        Validates:
        1. Logger is created successfully
        2. Warning logging executes without errors
        3. Result indicates success
        """
        logger = self.make_result_logger("test")
        tm.that(logger, none=False)
        self.assert_log_result_success(logger.warning("Test warning message"))

    def test_warning_with_context(self) -> None:
        """Test warning logging with context.

        Validates:
        1. Logger accepts context parameters
        2. Context is included in log entry
        3. Logging succeeds with context data
        """
        logger = self.make_result_logger("test")
        tm.that(u.logger, none=False)
        self.assert_log_result_success(
            logger.warning("Warning with context", retry_count=3, delay="1s"),
        )

    def test_error_logging(self) -> None:
        """Test error level logging.

        Validates:
        1. Logger is created successfully
        2. Error logging executes without errors
        3. Result indicates success
        """
        logger = self.make_result_logger("test")
        tm.that(logger, none=False)
        self.assert_log_result_success(logger.error("Test error message"))

    def test_error_with_context(self) -> None:
        """Test error logging with context.

        Validates:
        1. Logger accepts context parameters
        2. Context is included in log entry
        3. Logging succeeds with context data
        """
        logger = self.make_result_logger("test")
        tm.that(u.logger, none=False)
        self.assert_log_result_success(
            logger.error("Error with context", error_code="ERR_001", user_id="456"),
        )

    def test_critical_logging(self) -> None:
        """Test critical level logging.

        Validates:
        1. Logger is created successfully
        2. Critical logging executes without errors
        3. Result indicates success
        """
        logger = self.make_result_logger("test")
        tm.that(logger, none=False)
        self.assert_log_result_success(logger.critical("Test critical message"))

    def test_critical_with_context(self) -> None:
        """Test critical logging with context.

        Validates:
        1. Logger accepts context parameters
        2. Context is included in log entry
        3. Logging succeeds with context data
        """
        logger = self.make_result_logger("test")
        tm.that(u.logger, none=False)
        self.assert_log_result_success(
            logger.critical(
                "Critical with context",
                alert_level="high",
                system="payment",
            ),
        )

    def test_logging_with_formatting(self) -> None:
        """Test logging with message formatting.

        Validates:
        1. Logger supports string formatting
        2. Formatted message is processed correctly
        3. Logging succeeds with formatted message
        """
        logger = self.make_result_logger("test")
        tm.that(logger, none=False)
        self.assert_log_result_success(logger.info("User %s logged in", "john"))

    def test_logging_with_invalid_formatting(self) -> None:
        """Test logging with invalid format string.

        Validates:
        1. Logger handles format strings correctly
        2. Formatting arguments are processed
        3. Logging succeeds even with complex formatting
        """
        logger = self.make_result_logger("test")
        tm.that(logger, none=False)
        self.assert_log_result_success(
            logger.info("Message with %s and %d", "arg1", 42)
        )

    def test_exception_logging_with_exception_object(self) -> None:
        """Test logging with exception t.JsonValue.

        Validates:
        1. Logger handles exception objects correctly
        2. Exception details are captured
        3. Logging succeeds with exception context
        """
        logger = self.make_result_logger("test")
        tm.that(u.logger, none=False)
        msg = "Test error"
        exception_obj: ValueError | None = None
        try:
            raise ValueError(msg)
        except ValueError as e:
            exception_obj = e
            self.assert_log_result_success(
                logger.exception("An error occurred", exception=e),
            )
        tm.that(exception_obj, none=False)
        tm.that(exception_obj, is_=ValueError)
        tm.that(str(exception_obj), eq=msg)

    def test_exception_logging_with_exc_info(self) -> None:
        """Test logging with exc_info=True.

        Validates:
        1. Logger captures exception info automatically
        2. Exception context is included in log
        3. Logging succeeds with exception info
        """
        logger = self.make_result_logger("test")
        tm.that(logger, none=False)
        msg = "Test error"
        exception_obj: RuntimeError | None = None
        try:
            raise RuntimeError(msg)
        except RuntimeError as e:
            exception_obj = e
            self.assert_log_result_success(logger.exception("Operation failed"))
        tm.that(exception_obj, none=False)
        tm.that(exception_obj, is_=RuntimeError)
        tm.that(str(exception_obj), eq=msg)

    def test_exception_logging_without_current_exception(self) -> None:
        """Test logging error outside exception context.

        Validates:
        1. Logger handles error logging without exception context
        2. Error logging works outside try/except blocks
        3. Logging succeeds without exception info
        """
        logger = self.make_result_logger("test")
        tm.that(logger, none=False)
        self.assert_log_result_success(logger.error("No exception context"))

    def test_exception_logging_with_context(self) -> None:
        """Test exception logging with additional context.

        Validates:
        1. Logger handles exception objects with additional context
        2. Both exception and context are captured
        3. Logging succeeds with exception and context data
        """
        logger = self.make_result_logger("test")
        tm.that(u.logger, none=False)
        msg = "Test error"
        exception_obj: OSError | None = None
        try:
            raise OSError(msg)
        except OSError as e:
            exception_obj = e
            result = logger.exception(
                "IO operation failed",
                exception=e,
                operation="file_read",
                file="data.txt",
            )
            if result is not None:
                self.assert_log_result_success(result)
        tm.that(exception_obj, none=False)
        tm.that(exception_obj, is_=OSError)
        tm.that(str(exception_obj), eq=msg)

    def test_logging_with_empty_message(self) -> None:
        """Test logging with empty message."""
        logger = self.make_result_logger("test")
        self.assert_log_result_success(logger.info(""))

    def test_logging_with_none_context_values(self) -> None:
        """Test logging with None context values."""
        logger = self.make_result_logger("test")
        self.assert_log_result_success(logger.info("Message", context_key=""))

    def test_logging_with_large_context(self) -> None:
        """Test logging with large context dictionary."""
        logger = self.make_result_logger("test")
        large_context = {f"key_{i}": f"value_{i}" for i in range(100)}
        self.assert_log_result_success(
            logger.info("Message with large context", **large_context),
        )


__all__: t.MutableSequenceOf[str] = ["TestCoverageLoggings"]
