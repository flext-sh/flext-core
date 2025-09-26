"""Comprehensive tests for FlextLogger - Logging System.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import os
import pathlib
import tempfile
import threading
import time

from flext_core import FlextLogger


class TestFlextLogger:
    """Test suite for FlextLogger logging functionality."""

    def test_logger_initialization(self) -> None:
        """Test logger initialization."""
        logger = FlextLogger("test_logger")
        assert logger is not None
        assert isinstance(logger, FlextLogger)

    def test_logger_with_custom_config(self) -> None:
        """Test logger initialization with custom configuration."""
        config = {"level": "INFO", "format": "%(message)s"}
        logger = FlextLogger("test_logger", config=config)
        assert logger is not None

    def test_logger_info_logging(self) -> None:
        """Test info level logging."""
        logger = FlextLogger("test_logger")

        result = logger.info("Test info message")
        assert result.is_success

    def test_logger_debug_logging(self) -> None:
        """Test debug level logging."""
        logger = FlextLogger("test_logger")

        result = logger.debug("Test debug message")
        assert result.is_success

    def test_logger_warning_logging(self) -> None:
        """Test warning level logging."""
        logger = FlextLogger("test_logger")

        result = logger.warning("Test warning message")
        assert result.is_success

    def test_logger_error_logging(self) -> None:
        """Test error level logging."""
        logger = FlextLogger("test_logger")

        result = logger.error("Test error message")
        assert result.is_success

    def test_logger_critical_logging(self) -> None:
        """Test critical level logging."""
        logger = FlextLogger("test_logger")

        result = logger.critical("Test critical message")
        assert result.is_success

    def test_logger_logging_with_context(self) -> None:
        """Test logging with context."""
        logger = FlextLogger("test_logger")

        context = {"user_id": "123", "action": "test"}
        result = logger.info("Test message", context=context)
        assert result.is_success

    def test_logger_logging_with_correlation_id(self) -> None:
        """Test logging with correlation ID."""
        logger = FlextLogger("test_logger")

        result = logger.info("Test message", correlation_id="test_corr_123")
        assert result.is_success

    def test_logger_logging_with_exception(self) -> None:
        """Test logging with exception."""
        logger = FlextLogger("test_logger")

        try:
            msg = "Test exception"
            raise ValueError(msg)
        except ValueError as e:
            result = logger.exception("Test error", exception=e)
            assert result.is_success

    def test_logger_logging_with_structured_data(self) -> None:
        """Test logging with structured data."""
        logger = FlextLogger("test_logger")

        structured_data = {
            "event": "user_action",
            "data": {"key": "value"},
            "metadata": {"timestamp": "2025-01-01"},
        }

        result = logger.info("Test message", **structured_data)
        assert result.is_success

    def test_logger_logging_with_performance_metrics(self) -> None:
        """Test logging with performance metrics."""
        logger = FlextLogger("test_logger")

        performance = {"duration_ms": 150.5, "memory_mb": 64.2, "cpu_percent": 25.8}

        result = logger.info("Test message", performance=performance)
        assert result.is_success

    def test_logger_logging_with_custom_fields(self) -> None:
        """Test logging with custom fields."""
        logger = FlextLogger("test_logger")

        custom_fields = {
            "custom_field_1": "value1",
            "custom_field_2": 42,
            "custom_field_3": True,
        }

        result = logger.info("Test message", **custom_fields)
        assert result.is_success

    def test_logger_logging_with_nested_context(self) -> None:
        """Test logging with nested context."""
        logger = FlextLogger("test_logger")

        nested_context = {
            "user": {
                "id": "123",
                "profile": {"name": "John Doe", "email": "john@example.com"},
            },
            "session": {"id": "abc", "started_at": "2025-01-01T00:00:00Z"},
        }

        result = logger.info("Test message", context=nested_context)
        assert result.is_success

    def test_logger_logging_with_array_data(self) -> None:
        """Test logging with array data."""
        logger = FlextLogger("test_logger")

        array_data = {
            "items": [1, 2, 3, 4, 5],
            "tags": ["tag1", "tag2", "tag3"],
            "scores": [85.5, 92.3, 78.9],
        }

        result = logger.info("Test message", **array_data)
        assert result.is_success

    def test_logger_logging_with_boolean_data(self) -> None:
        """Test logging with boolean data."""
        logger = FlextLogger("test_logger")

        boolean_data = {"is_active": True, "is_verified": False, "has_permission": True}

        result = logger.info("Test message", **boolean_data)
        assert result.is_success

    def test_logger_logging_with_numeric_data(self) -> None:
        """Test logging with numeric data."""
        logger = FlextLogger("test_logger")

        numeric_data = {"count": 42, "percentage": 85.5, "ratio": 0.75}

        result = logger.info("Test message", **numeric_data)
        assert result.is_success

    def test_logger_logging_with_null_data(self) -> None:
        """Test logging with null data."""
        logger = FlextLogger("test_logger")

        null_data = {"nullable_field": None, "optional_field": None}

        result = logger.info("Test message", **null_data)
        assert result.is_success

    def test_logger_logging_with_empty_data(self) -> None:
        """Test logging with empty data."""
        logger = FlextLogger("test_logger")

        empty_data = {"empty_string": "", "empty_list": [], "empty_dict": {}}

        result = logger.info("Test message", **empty_data)
        assert result.is_success

    def test_logger_logging_with_special_characters(self) -> None:
        """Test logging with special characters."""
        logger = FlextLogger("test_logger")

        special_data = {
            "unicode_text": "Hello 世界",
            "special_chars": "!@#$%^&*()",
            "newlines": "Line 1\nLine 2\nLine 3",
        }

        result = logger.info("Test message", **special_data)
        assert result.is_success

    def test_logger_logging_with_large_data(self) -> None:
        """Test logging with large data."""
        logger = FlextLogger("test_logger")

        large_data = {
            "large_string": "x" * 1000,
            "large_list": list(range(1000)),
            "large_dict": {f"key_{i}": f"value_{i}" for i in range(1000)},
        }

        result = logger.info("Test message", **large_data)
        assert result.is_success

    def test_logger_logging_with_complex_data(self) -> None:
        """Test logging with complex data structures."""
        logger = FlextLogger("test_logger")

        complex_data = {
            "nested": {
                "level1": {
                    "level2": {
                        "level3": {
                            "data": [1, 2, 3],
                            "metadata": {
                                "created": "2025-01-01",
                                "updated": "2025-01-02",
                            },
                        }
                    }
                }
            },
            "arrays": [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"},
                {"id": 3, "name": "Item 3"},
            ],
        }

        result = logger.info("Test message", **complex_data)
        assert result.is_success

    def test_logger_logging_with_validation(self) -> None:
        """Test logging with validation."""
        logger = FlextLogger("test_logger")

        # Valid data should pass
        result = logger.info("Valid message", valid_field="value")
        assert result.is_success

    def test_logger_logging_with_validation_failure(self) -> None:
        """Test logging with validation failure."""
        logger = FlextLogger("test_logger")

        # Invalid data should fail
        try:
            result = logger.info("Invalid message", invalid_field=object())
            assert result.is_failure
        except (TypeError, ValueError):
            pass  # Expected

    def test_logger_logging_with_middleware(self) -> None:
        """Test logging with middleware."""
        logger = FlextLogger("test_logger")

        middleware_called = False

        def middleware(message: str, **kwargs: object) -> tuple[str, dict[str, object]]:
            nonlocal middleware_called
            middleware_called = True
            return message, kwargs

        logger.add_middleware(middleware)
        result = logger.info("Test message")
        assert result.is_success
        assert middleware_called is True

    def test_logger_logging_with_filters(self) -> None:
        """Test logging with filters."""
        logger = FlextLogger("test_logger")

        def test_filter(record: object) -> bool:
            return "filtered" not in record.getMessage()

        logger.add_filter(test_filter)
        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_handlers(self) -> None:
        """Test logging with custom handlers."""
        logger = FlextLogger("test_logger")

        class TestHandler(logging.Handler):
            def __init__(self) -> None:
                super().__init__()
                self.messages: list[str] = []

            def emit(self, record: logging.LogRecord) -> None:
                self.messages.append(record.getMessage())

        handler = TestHandler()
        logger.add_handler(handler)

        result = logger.info("Test message")
        assert result.is_success
        assert "Test message" in handler.messages

    def test_logger_logging_with_formatters(self) -> None:
        """Test logging with custom formatters."""
        logger = FlextLogger("test_logger")

        formatter = logging.Formatter("%(name)s - %(message)s")
        logger.set_formatter(formatter)

        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_levels(self) -> None:
        """Test logging with different levels."""
        logger = FlextLogger("test_logger")

        # Test all levels
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in levels:
            result = logger.log(level, f"Test {level.lower()} message")
            assert result.is_success

    def test_logger_logging_with_level_filtering(self) -> None:
        """Test logging with level filtering."""
        logger = FlextLogger("test_logger", config={"level": "WARNING"})

        # Debug and info should be filtered out
        result = logger.debug("Debug message")
        assert result.is_success  # Logger should still return success

        result = logger.info("Info message")
        assert result.is_success  # Logger should still return success

        # Warning and above should be logged
        result = logger.warning("Warning message")
        assert result.is_success

        result = logger.error("Error message")
        assert result.is_success

    def test_logger_logging_with_thread_safety(self) -> None:
        """Test logging with thread safety."""
        logger = FlextLogger("test_logger")

        results = []

        def log_message(thread_id: int) -> None:
            result = logger.info(f"Thread {thread_id} message")
            results.append(result)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=log_message, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 10
        assert all(result.is_success for result in results)

    def test_logger_logging_with_performance(self) -> None:
        """Test logging performance."""
        logger = FlextLogger("test_logger")

        start_time = time.time()

        # Perform many logging operations
        for i in range(1000):
            logger.info(f"Test message {i}")

        end_time = time.time()

        # Should complete 1000 operations in reasonable time
        assert end_time - start_time < 1.0

    def test_logger_logging_with_error_handling(self) -> None:
        """Test logging with error handling."""
        logger = FlextLogger("test_logger")

        # Test logging with invalid data
        try:
            result = logger.info("Test message", invalid_field=object())
            assert result.is_failure
        except (TypeError, ValueError):
            pass  # Expected

    def test_logger_logging_with_cleanup(self) -> None:
        """Test logging with cleanup."""
        logger = FlextLogger("test_logger")

        result = logger.info("Test message")
        assert result.is_success

        # Cleanup
        logger.cleanup()

        # After cleanup, logging should still work
        result = logger.info("Test message after cleanup")
        assert result.is_success

    def test_logger_logging_with_statistics(self) -> None:
        """Test logging with statistics."""
        logger = FlextLogger("test_logger")

        logger.info("Test message 1")
        logger.warning("Test message 2")
        logger.error("Test message 3")

        stats = logger.get_statistics()
        assert stats["total_messages"] >= 3
        assert stats["info_messages"] >= 1
        assert stats["warning_messages"] >= 1
        assert stats["error_messages"] >= 1

    def test_logger_logging_with_audit(self) -> None:
        """Test logging with audit."""
        logger = FlextLogger("test_logger")

        result = logger.info("Test message")
        assert result.is_success

        # Check audit log
        audit_log = logger.get_audit_log()
        assert len(audit_log) >= 1
        assert audit_log[0]["message"] == "Test message"

    def test_logger_logging_with_context_management(self) -> None:
        """Test logging with context management."""
        logger = FlextLogger("test_logger")

        # Test basic logging
        result = logger.info("Test message")
        assert result.is_success

        # Test context management
        result = logger.set_correlation_id("test-correlation-123")
        assert result.is_success

        # Test logging with context
        result = logger.info("Test message with context", user_id="123", action="test")
        assert result.is_success

        # Test request context
        result = logger.set_request_context(service="test-service", version="1.0")
        assert result.is_success

        # Verify logging works with context
        result = logger.info("Test message with request context")
        assert result.is_success

    def test_logger_validation(self) -> None:
        """Test logger validation."""
        logger = FlextLogger("test_logger")

        result = logger.validate()
        assert result.is_success

    def test_logger_logging_with_singleton_pattern(self) -> None:
        """Test logging with singleton pattern."""
        logger1 = FlextLogger.get_instance("test_logger")
        logger2 = FlextLogger.get_instance("test_logger")

        assert logger1 is logger2

    def test_logger_logging_with_singleton_reset(self) -> None:
        """Test logging with singleton reset."""
        logger1 = FlextLogger.get_instance("test_logger")
        FlextLogger.reset_instance("test_logger")
        logger2 = FlextLogger.get_instance("test_logger")

        assert logger1 is not logger2

    def test_logger_logging_with_environment_variables(self) -> None:
        """Test logging with environment variables."""
        os.environ["FLEXT_LOG_LEVEL"] = "DEBUG"

        try:
            logger = FlextLogger("test_logger")
            result = logger.debug("Test debug message")
            assert result.is_success
        finally:
            if "FLEXT_LOG_LEVEL" in os.environ:
                del os.environ["FLEXT_LOG_LEVEL"]

    def test_logger_logging_with_file_output(self) -> None:
        """Test logging with file output."""
        with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w", delete=False) as f:
            temp_file = f.name

        try:
            config = {"output": "file", "file_path": temp_file}
            logger = FlextLogger("test_logger", config=config)

            result = logger.info("Test message to file")
            assert result.is_success

            # Verify message was written to file
            with pathlib.Path(temp_file).open(encoding="utf-8") as f:
                content = f.read()
                assert "Test message to file" in content
        finally:
            pathlib.Path(temp_file).unlink()

    def test_logger_logging_with_json_output(self) -> None:
        """Test logging with JSON output."""
        config = {"format": "json"}
        logger = FlextLogger("test_logger", config=config)

        result = logger.info("Test message", field1="value1", field2=42)
        assert result.is_success

    def test_logger_logging_with_structured_output(self) -> None:
        """Test logging with structured output."""
        config = {"format": "structured"}
        logger = FlextLogger("test_logger", config=config)

        result = logger.info("Test message", structured_field="value")
        assert result.is_success

    def test_logger_logging_with_custom_output(self) -> None:
        """Test logging with custom output."""
        config = {"format": "custom", "custom_format": "%(name)s: %(message)s"}
        logger = FlextLogger("test_logger", config=config)

        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_rotation(self) -> None:
        """Test logging with log rotation."""
        config = {"rotation": True, "max_size": "1MB", "backup_count": 5}
        logger = FlextLogger("test_logger", config=config)

        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_compression(self) -> None:
        """Test logging with compression."""
        config = {"compression": True, "compression_format": "gz"}
        logger = FlextLogger("test_logger", config=config)

        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_buffering(self) -> None:
        """Test logging with buffering."""
        config = {"buffering": True, "buffer_size": 1000}
        logger = FlextLogger("test_logger", config=config)

        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_async_output(self) -> None:
        """Test logging with async output."""
        config = {"async": True, "queue_size": 1000}
        logger = FlextLogger("test_logger", config=config)

        result = logger.info("Test message")
        assert result.is_success

    def test_logger_logging_with_metrics_collection(self) -> None:
        """Test logging with metrics collection."""
        logger = FlextLogger("test_logger")

        logger.info("Test message 1")
        logger.warning("Test message 2")
        logger.error("Test message 3")

        metrics = logger.get_metrics()
        assert metrics["total_messages"] >= 3
        assert metrics["info_messages"] >= 1
        assert metrics["warning_messages"] >= 1
        assert metrics["error_messages"] >= 1

    def test_logger_get_performance_metrics(self) -> None:
        """Test getting logger performance metrics."""
        logger = FlextLogger("test_logger")

        time.time()
        logger.info("Test message")
        time.time()

        performance = logger.get_performance_metrics()
        assert performance["avg_logging_time"] >= 0
        assert performance["total_logging_time"] >= 0
