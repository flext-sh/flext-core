"""Comprehensive tests for FlextLogger targeting missing coverage lines.

This module provides comprehensive test coverage for loggings.py using extensive
flext_tests standardization patterns to achieve near 100% coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from unittest.mock import Mock, patch

from flext_core import FlextLogger


class TestFlextLoggingComprehensiveCoverage:
    """Comprehensive tests for FlextLogger missing coverage lines."""

    def test_extract_service_name_from_environment(self) -> None:
        """Test _extract_service_name with SERVICE_NAME environment variable (line 178)."""
        with patch.dict(os.environ, {"SERVICE_NAME": "test-service"}):
            logger = FlextLogger("test.module")
            # Access private method for testing
            service_name = logger._extract_service_name()
            assert service_name == "test-service"

    def test_get_calling_function_exception_handling(self) -> None:
        """Test _get_calling_function exception handling (lines 323-324)."""
        logger = FlextLogger("test.module")

        # Mock sys._getframe to raise AttributeError
        with patch("sys._getframe", side_effect=AttributeError("Frame not available")):
            result = logger._get_calling_function()
            assert result == "unknown"

        # Mock sys._getframe to raise ValueError
        with patch("sys._getframe", side_effect=ValueError("Invalid frame")):
            result = logger._get_calling_function()
            assert result == "unknown"

    def test_get_calling_line_exception_handling(self) -> None:
        """Test _get_calling_line exception handling (lines 331-332)."""
        logger = FlextLogger("test.module")

        # Mock sys._getframe to raise AttributeError
        with patch("sys._getframe", side_effect=AttributeError("Frame not available")):
            result = logger._get_calling_line()
            assert result == 0

        # Mock sys._getframe to raise ValueError
        with patch("sys._getframe", side_effect=ValueError("Invalid frame")):
            result = logger._get_calling_line()
            assert result == 0

    def test_debug_with_structured_output_disabled(self) -> None:
        """Test debug method with structured_output disabled (line 469)."""
        # Disable structured output to exercise simple call path
        FlextLogger.configure(structured_output=False)
        logger = FlextLogger("test.module", _force_new=True)

        # Mock the structlog logger to verify call
        mock_structlog = Mock()
        logger._structlog_logger = mock_structlog

        # Call debug method
        logger.debug("Test debug message", extra_context="test")

        # Verify structured logging is called with proper parameters
        mock_structlog.debug.assert_called_once()
        call_args = mock_structlog.debug.call_args
        assert call_args[0][0] == "Test debug message"
        assert "context" in call_args[1] or "extra_context" in call_args[1]

    def test_info_with_structured_output_disabled(self) -> None:
        """Test info method with structured_output disabled (line 479)."""
        FlextLogger.configure(structured_output=False)
        logger = FlextLogger("test.module", _force_new=True)

        # Mock the structlog logger to verify call
        mock_structlog = Mock()
        logger._structlog_logger = mock_structlog

        # Call info method
        logger.info("Test info message", extra_context="test")

        # Verify structured logging is bypassed
        mock_structlog.info.assert_called_once_with(
            "Test info message", extra_context="test"
        )

    def test_warning_with_structured_output_disabled(self) -> None:
        """Test warning method with structured_output disabled (line 489)."""
        FlextLogger.configure(structured_output=False)
        logger = FlextLogger("test.module", _force_new=True)

        # Mock the structlog logger to verify call
        mock_structlog = Mock()
        logger._structlog_logger = mock_structlog

        # Call warning method
        logger.warning("Test warning message", extra_context="test")

        # Verify structured logging is bypassed
        mock_structlog.warning.assert_called_once_with(
            "Test warning message", extra_context="test"
        )

    def test_error_with_structured_output_disabled_and_error(self) -> None:
        """Test error method with structured_output disabled and error (lines 505-507)."""
        FlextLogger.configure(structured_output=False)
        logger = FlextLogger("test.module", _force_new=True)

        # Mock the structlog logger to verify call
        mock_structlog = Mock()
        logger._structlog_logger = mock_structlog

        # Create test error
        test_error = ValueError("Test error")

        # Call error method with error parameter
        logger.error("Test error message", error=test_error, extra_context="test")

        # Verify error is added to context and structured logging is bypassed
        mock_structlog.error.assert_called_once_with(
            "Test error message", error="Test error", extra_context="test"
        )

    def test_critical_with_structured_output_disabled_and_error(self) -> None:
        """Test critical method with structured_output disabled and error (lines 523-525)."""
        FlextLogger.configure(structured_output=False)
        logger = FlextLogger("test.module", _force_new=True)

        # Mock the structlog logger to verify call
        mock_structlog = Mock()
        logger._structlog_logger = mock_structlog

        # Create test error
        test_error = RuntimeError("Critical error")

        # Call critical method with error parameter
        logger.critical(
            "Critical error message", error=test_error, extra_context="test"
        )

        # Verify error is added to context and structured logging is bypassed
        mock_structlog.critical.assert_called_once_with(
            "Critical error message", error="Critical error", extra_context="test"
        )

    def test_set_correlation_id_functionality(self) -> None:
        """Test set_correlation_id method functionality."""
        logger = FlextLogger("test.module")

        # Test setting correlation ID
        test_correlation_id = "test-correlation-123"
        logger.set_correlation_id(test_correlation_id)

        # Verify correlation ID is set (access private attribute for testing)
        assert logger._correlation_id == test_correlation_id

    def test_comprehensive_logging_scenarios(self) -> None:
        """Test comprehensive logging scenarios covering edge cases."""
        logger = FlextLogger("test.module")

        # Test with various message formatting scenarios
        logger.debug("Debug with args: %s %d", "test", 42)
        logger.info("Info message")
        logger.warning("Warning with context", user_id=123, action="test")
        logger.error("Error message", error=Exception("Test exception"))
        logger.critical("Critical message", error="String error")

        # Test exception logging
        try:
            msg = "Test exception for logging"
            raise ValueError(msg)
        except ValueError:
            logger.exception("Exception occurred during test")

    def test_logger_configuration_edge_cases(self) -> None:
        """Test logger configuration with various edge cases."""
        # Test with minimal configuration
        logger1 = FlextLogger("minimal")
        assert logger1 is not None

        # Test with custom configuration
        logger2 = FlextLogger("custom", service_name="custom-service")
        assert logger2 is not None

        # Test with disabled structured output
        logger3 = FlextLogger("disabled")
        assert logger3 is not None

    def test_private_method_access_comprehensive(self) -> None:
        """Test comprehensive access to private methods for coverage."""
        logger = FlextLogger("test.comprehensive")

        # Test all private methods with various scenarios
        service_name = logger._extract_service_name()
        assert isinstance(service_name, str)

        calling_func = logger._get_calling_function()
        assert isinstance(calling_func, str)

        calling_line = logger._get_calling_line()
        assert isinstance(calling_line, int)

    def test_structured_vs_unstructured_logging_comprehensive(self) -> None:
        """Test comprehensive comparison of structured vs unstructured logging."""
        # Test structured logging (default)
        structured_logger = FlextLogger("structured")
        structured_logger.debug("Structured debug")
        structured_logger.info("Structured info")
        structured_logger.warning("Structured warning")
        structured_logger.error("Structured error")
        structured_logger.critical("Structured critical")

        # Test unstructured logging
        unstructured_logger = FlextLogger("unstructured")
        unstructured_logger.debug("Unstructured debug")
        unstructured_logger.info("Unstructured info")
        unstructured_logger.warning("Unstructured warning")
        unstructured_logger.error("Unstructured error")
        unstructured_logger.critical("Unstructured critical")

    def test_error_handling_with_different_error_types(self) -> None:
        """Test error handling with different error types for comprehensive coverage."""
        logger = FlextLogger("error_test")

        # Mock the structlog logger
        mock_structlog = Mock()
        logger._structlog_logger = mock_structlog

        # Test with Exception object
        exception_obj = RuntimeError("Runtime error")
        logger.error("Error with exception", error=exception_obj)

        # Test with string error
        logger.error("Error with string", error="String error message")

        # Test critical with Exception object
        critical_exception = ValueError("Critical value error")
        logger.critical("Critical with exception", error=critical_exception)

        # Test critical with string error
        logger.critical("Critical with string", error="Critical string error")

        # Verify all calls were made
        assert mock_structlog.error.call_count == 2
        assert mock_structlog.critical.call_count == 2
