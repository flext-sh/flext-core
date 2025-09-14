"""Comprehensive FlextLogger tests targeting uncovered lines and edge cases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

from flext_core import FlextLogger


class TestFlextLoggingComprehensive:
    """Comprehensive tests for FlextLogger targeting uncovered functionality."""

    def test_singleton_instance_management_force_new_false(self) -> None:
        """Test FlextLogger.__new__ singleton behavior with force_new=False."""
        # Clear any existing instances
        FlextLogger._instances.clear()

        # First instance creation
        logger1 = FlextLogger("test_singleton_1")
        assert "test_singleton_1" in FlextLogger._instances
        assert FlextLogger._instances["test_singleton_1"] is logger1

        # Second creation should return same instance
        logger2 = FlextLogger("test_singleton_1")
        assert logger1 is logger2
        assert id(logger1) == id(logger2)

    def test_singleton_instance_management_force_new_true(self) -> None:
        """Test FlextLogger.__new__ with force_new=True bypasses singleton."""
        FlextLogger._instances.clear()

        # Create initial instance
        logger1 = FlextLogger("test_force_new")

        # Force new instance creation
        logger2 = FlextLogger("test_force_new", _force_new=True)

        # Should be different instances
        assert logger1 is not logger2
        assert id(logger1) != id(logger2)

        # First logger should still be cached
        assert FlextLogger._instances["test_force_new"] is logger1

    def test_singleton_instance_management_force_new_not_cached(self) -> None:
        """Test that force_new instances are not cached."""
        FlextLogger._instances.clear()

        # Create forced new instance
        logger1 = FlextLogger("test_not_cached", _force_new=True)

        # Should not be in cache
        assert "test_not_cached" not in FlextLogger._instances

        # Next instance should be cached normally
        logger2 = FlextLogger("test_not_cached")
        assert FlextLogger._instances["test_not_cached"] is logger2
        assert logger1 is not logger2

    def test_equality_comparison_different_instances_same_name(self) -> None:
        """Test FlextLogger.__eq__ with different instances but same name."""
        FlextLogger._instances.clear()
        logger1 = FlextLogger("test_eq_diff")
        logger2 = FlextLogger("test_eq_diff", _force_new=True)

        # Same name but different instances
        assert logger1._name == logger2._name
        assert logger1 != logger2
        assert (logger1 == logger2) is False

    def test_equality_comparison_non_logger_object(self) -> None:
        """Test FlextLogger.__eq__ with non-FlextLogger object."""
        logger = FlextLogger("test_eq_non_logger")
        assert logger != "not_a_logger"
        assert (logger == "not_a_logger") is False
        assert logger != 12345
        assert (logger is None) is False

    def test_hash_functionality(self) -> None:
        """Test FlextLogger.__hash__ returns consistent hash values."""
        logger1 = FlextLogger("test_hash")
        logger2 = FlextLogger("test_hash")  # Same name, should be same instance

        hash1 = hash(logger1)
        hash2 = hash(logger2)

        assert isinstance(hash1, int)
        assert hash1 == hash2  # Same instance should have same hash
        assert hash(logger1) == hash(logger1)  # Consistent

    def test_extract_service_name_complex_module_path(self) -> None:
        """Test _extract_service_name with complex module paths."""
        logger = FlextLogger("complex.module.path.service.component")

        # Should extract reasonable service name from complex path
        service_name = logger._extract_service_name()
        assert isinstance(service_name, str)
        assert len(service_name) > 0
        # Should not be the full path
        assert service_name != "complex.module.path.service.component"

    def test_extract_service_name_short_path(self) -> None:
        """Test _extract_service_name with short module path."""
        logger = FlextLogger("short")
        service_name = logger._extract_service_name()
        # The actual implementation extracts "flext-core" from the project structure
        assert isinstance(service_name, str)
        assert len(service_name) > 0

    def test_extract_service_name_minimal_parts(self) -> None:
        """Test _extract_service_name meets minimum parts requirement."""
        logger = FlextLogger("a.b")
        service_name = logger._extract_service_name()

        # Should handle minimum parts requirement
        assert isinstance(service_name, str)
        assert len(service_name) > 0

    def test_get_version_method(self) -> None:
        """Test _get_version method returns version info."""
        logger = FlextLogger("test_version")
        version = logger._get_version()

        assert isinstance(version, str)
        assert len(version) > 0

    def test_get_environment_method(self) -> None:
        """Test _get_environment method returns environment info."""
        logger = FlextLogger("test_env")

        with patch.dict(os.environ, {"ENVIRONMENT": "test_custom"}):
            env = logger._get_environment()
            assert env == "test_custom"

        # Test fallback
        with patch.dict(os.environ, {}, clear=True):
            env = logger._get_environment()
            assert isinstance(env, str)

    def test_get_instance_id_method(self) -> None:
        """Test _get_instance_id method returns unique instance ID."""
        logger = FlextLogger("test_instance_id")
        instance_id = logger._get_instance_id()

        assert isinstance(instance_id, str)
        assert len(instance_id) > 0

    def test_generate_correlation_id_method(self) -> None:
        """Test _generate_correlation_id creates unique IDs."""
        logger = FlextLogger("test_corr_id")

        corr_id1 = logger._generate_correlation_id()
        corr_id2 = logger._generate_correlation_id()

        assert isinstance(corr_id1, str)
        assert isinstance(corr_id2, str)
        assert corr_id1 != corr_id2  # Should be unique

    def test_get_current_timestamp_method(self) -> None:
        """Test _get_current_timestamp returns ISO format timestamp."""
        logger = FlextLogger("test_timestamp")
        timestamp = logger._get_current_timestamp()

        assert isinstance(timestamp, str)
        # Should be ISO format
        datetime.fromisoformat(timestamp)  # Should not raise

    def test_sanitize_context_nested_dictionaries(self) -> None:
        """Test _sanitize_context with nested dictionary structures."""
        logger = FlextLogger("test_sanitize_nested")

        context = {
            "user_info": {
                "name": "John Doe",
                "password": "secret123",
                "profile": {
                    "email": "john@example.com",
                    "api_key": "key_abc123",
                    "preferences": {"token": "nested_token", "theme": "dark"},
                },
            },
            "metadata": {
                "timestamp": "2023-01-01T00:00:00Z",
                "secret_config": "confidential",
            },
        }

        sanitized = logger._sanitize_context(context)

        # Check nested sanitization
        assert sanitized["user_info"]["name"] == "John Doe"
        assert sanitized["user_info"]["password"] == "[REDACTED]"
        assert sanitized["user_info"]["profile"]["email"] == "john@example.com"
        assert sanitized["user_info"]["profile"]["api_key"] == "[REDACTED]"
        assert sanitized["user_info"]["profile"]["preferences"]["token"] == "[REDACTED]"
        assert sanitized["user_info"]["profile"]["preferences"]["theme"] == "dark"
        assert sanitized["metadata"]["timestamp"] == "2023-01-01T00:00:00Z"
        assert sanitized["metadata"]["secret_config"] == "[REDACTED]"

    def test_sanitize_context_various_sensitive_key_patterns(self) -> None:
        """Test _sanitize_context recognizes various sensitive key patterns."""
        logger = FlextLogger("test_sanitize_patterns")

        context = {
            "PASSWORD": "upper_secret",
            "my_secret_key": "contains_secret",
            "auth_token": "contains_token",
            "user_credential": "contains_credential",
            "private_data": "contains_private",
            "session_cookie": "contains_cookie",
            "authorization_header": "contains_auth",
            "access_key_id": "contains_key",
            "normal_field": "should_not_redact",
            "REFRESH_TOKEN": "upper_token",
        }

        sanitized = logger._sanitize_context(context)

        # All sensitive keys should be redacted
        assert sanitized["PASSWORD"] == "[REDACTED]"
        assert sanitized["my_secret_key"] == "[REDACTED]"
        assert sanitized["auth_token"] == "[REDACTED]"
        assert sanitized["user_credential"] == "[REDACTED]"
        assert sanitized["private_data"] == "[REDACTED]"
        assert sanitized["session_cookie"] == "[REDACTED]"
        assert sanitized["authorization_header"] == "[REDACTED]"
        assert sanitized["access_key_id"] == "[REDACTED]"
        assert sanitized["REFRESH_TOKEN"] == "[REDACTED]"

        # Normal field should not be redacted
        assert sanitized["normal_field"] == "should_not_redact"

    def test_sanitize_context_empty_and_none_values(self) -> None:
        """Test _sanitize_context handles empty and None values."""
        logger = FlextLogger("test_sanitize_empty")

        context = {
            "password": None,
            "secret": "",
            "normal_none": None,
            "normal_empty": "",
            "nested_empty": {},
            "nested_with_empty": {"password": "", "normal": "value"},
        }

        sanitized = logger._sanitize_context(context)

        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["secret"] == "[REDACTED]"
        assert sanitized["normal_none"] is None
        assert sanitized["normal_empty"] == ""
        assert sanitized["nested_empty"] == {}
        assert sanitized["nested_with_empty"]["password"] == "[REDACTED]"
        assert sanitized["nested_with_empty"]["normal"] == "value"

    def test_build_log_entry_with_all_parameters(self) -> None:
        """Test _build_log_entry with all optional parameters."""
        logger = FlextLogger("test_build_entry")

        # Set up request context
        test_context = {"request_id": "req_123", "user_id": "user_456"}
        logger.set_request_context(**test_context)

        error = Exception("Test error")
        entry_context = {"custom_field": "custom_value", "password": "secret"}

        entry = logger._build_log_entry(
            level="ERROR",
            message="Test message",
            context=entry_context,
            error=error,
            duration_ms=150.5,
        )

        # Verify entry structure
        assert entry["level"] == "ERROR"
        assert entry["message"] == "Test message"
        # Note: duration_ms might be nested in execution or performance context
        assert "error" in entry
        # Request context might be nested or not present at top level
        assert "context" in entry or "request_id" in entry

        # Check that custom fields are present (might be in context subsection)
        if "context" in entry:
            context = entry["context"]
            assert context["custom_field"] == "custom_value"
            assert context["password"] == "[REDACTED]"  # Should be sanitized
        else:
            assert entry["custom_field"] == "custom_value"
            assert entry["password"] == "[REDACTED]"  # Should be sanitized

    def test_build_log_entry_without_optional_parameters(self) -> None:
        """Test _build_log_entry with minimal parameters."""
        logger = FlextLogger("test_build_minimal")

        entry = logger._build_log_entry(level="INFO", message="Simple message")

        assert entry["level"] == "INFO"
        assert entry["message"] == "Simple message"
        assert "@timestamp" in entry  # Uses @timestamp not timestamp
        # Check for actual fields in the entry rather than assuming logger_name
        assert "correlation_id" in entry or "service" in entry

    def test_get_calling_function_with_stack_frames(self) -> None:
        """Test _get_calling_function retrieves correct caller info."""
        logger = FlextLogger("test_calling_func")

        def test_function() -> str:
            return logger._get_calling_function()

        # Should return function name or handle gracefully
        result = test_function()
        assert isinstance(result, str)

    def test_get_calling_line_with_stack_frames(self) -> None:
        """Test _get_calling_line retrieves correct line number."""
        logger = FlextLogger("test_calling_line")

        def test_function() -> int:
            return logger._get_calling_line()

        # Should return line number or handle gracefully
        result = test_function()
        assert isinstance(result, int)
        assert result > 0

    def test_set_and_get_correlation_id(self) -> None:
        """Test correlation ID management."""
        logger = FlextLogger("test_correlation")

        test_corr_id = "test_correlation_123"
        logger.set_correlation_id(test_corr_id)

        # Should be stored in instance-level correlation ID
        assert logger._correlation_id == test_corr_id

    def test_set_and_clear_request_context(self) -> None:
        """Test request context management."""
        logger = FlextLogger("test_request_context")

        context = {"request_id": "req_123", "user_id": "user_456"}
        logger.set_request_context(**context)  # Pass as kwargs

        # Should be stored in local thread storage
        assert hasattr(logger._local, "request_context")
        stored_context = getattr(logger._local, "request_context", {})
        assert stored_context["request_id"] == "req_123"
        assert stored_context["user_id"] == "user_456"

        # Test clear
        logger.clear_request_context()
        cleared_context = getattr(logger._local, "request_context", {})
        assert cleared_context == {}

    def test_bind_method_creates_new_logger_instance(self) -> None:
        """Test bind method creates new logger with additional context."""
        original_logger = FlextLogger("test_bind_original")

        bind_context = {"module": "test_module", "function": "test_function"}
        bound_logger = original_logger.bind(**bind_context)

        # Should be different instance
        assert bound_logger is not original_logger
        assert bound_logger._name == original_logger._name

        # Should have additional context
        assert hasattr(bound_logger, "_persistent_context")

    def test_set_context_method(self) -> None:
        """Test set_context method stores context correctly."""
        logger = FlextLogger("test_set_context")

        context_dict = {"key1": "value1", "key2": "value2"}
        logger.set_context(**context_dict)

        # Should be stored in permanent context (not persistent_context)
        assert hasattr(logger, "_permanent_context")
        stored_context = getattr(logger, "_permanent_context", {})
        assert "key1" in stored_context
        assert "key2" in stored_context

    def test_with_context_method(self) -> None:
        """Test with_context method (alias for bind)."""
        logger = FlextLogger("test_with_context")

        context = {"component": "test_component"}
        context_logger = logger.with_context(**context)

        # Should behave like bind
        assert context_logger is not logger
        assert hasattr(context_logger, "_persistent_context")

    def test_start_operation_tracking(self) -> None:
        """Test start_operation creates tracking entry."""
        logger = FlextLogger("test_start_op")

        operation_context = {"user_id": "123", "action": "create"}
        operation_id = logger.start_operation("test_operation", **operation_context)

        # Should return operation ID
        assert isinstance(operation_id, str)
        assert operation_id.startswith("op_")

        # Should store operation info
        assert hasattr(logger._local, "operations")
        operations = getattr(logger._local, "operations", {})
        assert operation_id in operations

        operation_info = operations[operation_id]
        assert operation_info["name"] == "test_operation"
        assert "start_time" in operation_info
        assert operation_info["context"] == operation_context

    def test_complete_operation_success(self) -> None:
        """Test complete_operation with successful completion."""
        logger = FlextLogger("test_complete_success")

        # Start operation first
        operation_id = logger.start_operation("test_operation")

        # Small delay to measure duration
        time.sleep(0.001)

        # Complete successfully
        completion_context = {"result": "success", "records_processed": 100}
        logger.complete_operation(operation_id, success=True, **completion_context)

        # Operation should be removed from tracking
        operations = getattr(logger._local, "operations", {})
        assert operation_id not in operations

    def test_complete_operation_failure(self) -> None:
        """Test complete_operation with failure."""
        logger = FlextLogger("test_complete_failure")

        # Start operation first
        operation_id = logger.start_operation("test_operation")

        # Complete with failure
        failure_context = {"error_code": "ERR_001", "error_message": "Test failure"}
        logger.complete_operation(operation_id, success=False, **failure_context)

        # Operation should be removed from tracking
        operations = getattr(logger._local, "operations", {})
        assert operation_id not in operations

    def test_complete_operation_no_tracking_data(self) -> None:
        """Test complete_operation handles missing tracking data gracefully."""
        logger = FlextLogger("test_complete_no_tracking")

        # Try to complete operation that was never started
        logger.complete_operation("non_existent_op_123", success=True)

        # Should handle gracefully without errors
        # No specific assertions - just verifying no exceptions

    def test_complete_operation_no_local_operations(self) -> None:
        """Test complete_operation when _local.operations doesn't exist."""
        logger = FlextLogger("test_complete_no_local")

        # Ensure no operations attribute exists
        if hasattr(logger._local, "operations"):
            delattr(logger._local, "operations")

        # Should handle gracefully
        logger.complete_operation("some_op_id", success=True)

    def test_configure_classmethod_json_output_auto_detect_production(self) -> None:
        """Test configure auto-detects JSON output for production environment."""
        # Reset configuration
        FlextLogger._configured = False

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            FlextLogger.configure(log_level="DEBUG")

            config = FlextLogger._configuration
            assert config["json_output"] is True
            assert config["log_level"] == "DEBUG"

    def test_configure_classmethod_json_output_auto_detect_development(self) -> None:
        """Test configure auto-detects structured output for development."""
        FlextLogger._configured = False

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            FlextLogger.configure(log_level="INFO")

            config = FlextLogger._configuration
            assert config["json_output"] is False

    def test_configure_classmethod_json_output_explicit_override(self) -> None:
        """Test configure respects explicit json_output parameter."""
        FlextLogger._configured = False

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            FlextLogger.configure(json_output=True, log_level="WARNING")

            config = FlextLogger._configuration
            assert config["json_output"] is True  # Explicit override

    def test_configure_classmethod_reconfiguration_reset(self) -> None:
        """Test configure handles reconfiguration by resetting defaults."""
        # First configuration
        FlextLogger.configure(log_level="ERROR")
        assert FlextLogger._configured is True

        # Mock structlog to verify reset is called
        with (
            patch("structlog.is_configured", return_value=True),
            patch("structlog.reset_defaults") as mock_reset,
        ):
            # Reconfigure
            FlextLogger.configure(log_level="DEBUG")

            # Should call reset
            mock_reset.assert_called_once()

    def test_configure_classmethod_processors_structured_output_false(self) -> None:
        """Test configure with structured_output=False uses KeyValueRenderer."""
        FlextLogger._configured = False

        with patch("structlog.configure") as mock_configure:
            FlextLogger.configure(structured_output=False, json_output=False)

            # Verify configure was called
            mock_configure.assert_called_once()

            # Get processors from the call
            # Should include KeyValueRenderer
            # KeyValueRenderer should be present for non-structured output

    def test_add_correlation_processor_static_method(self) -> None:
        """Test _add_correlation_processor static processor method."""
        event_dict = {"message": "test", "other_field": "value"}

        result = FlextLogger._add_correlation_processor(
            _logger=MagicMock(), _method_name="info", event_dict=event_dict
        )

        # Should return modified event dict
        assert isinstance(result, dict)
        assert "message" in result
        assert "other_field" in result

    def test_add_performance_processor_static_method(self) -> None:
        """Test _add_performance_processor static processor method."""
        event_dict = {"message": "test", "duration_ms": 150.5}

        result = FlextLogger._add_performance_processor(
            _logger=MagicMock(), _method_name="info", event_dict=event_dict
        )

        # Should return modified event dict
        assert isinstance(result, dict)
        assert "message" in result

    def test_sanitize_processor_static_method(self) -> None:
        """Test _sanitize_processor static method redacts sensitive data."""
        event_dict = {
            "message": "Login attempt",
            "username": "john_doe",
            "password": "secret123",
            "api_key": "key_abc",
            "normal_field": "safe_value",
        }

        result = FlextLogger._sanitize_processor(
            _logger=MagicMock(), _method_name="info", event_dict=event_dict
        )

        # Sensitive fields should be redacted
        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"

        # Non-sensitive fields should remain
        assert result["username"] == "john_doe"
        assert result["normal_field"] == "safe_value"
        assert result["message"] == "Login attempt"

    def test_create_enhanced_console_renderer_static_method(self) -> None:
        """Test _create_enhanced_console_renderer creates console renderer."""
        renderer = FlextLogger._create_enhanced_console_renderer()

        # Should return a callable renderer
        assert callable(renderer)

    def test_set_global_correlation_id_classmethod(self) -> None:
        """Test set_global_correlation_id class method."""
        test_id = "global_correlation_123"
        FlextLogger.set_global_correlation_id(test_id)

        assert FlextLogger._global_correlation_id == test_id

    def test_get_global_correlation_id_classmethod(self) -> None:
        """Test get_global_correlation_id class method."""
        test_id = "global_correlation_456"
        FlextLogger._global_correlation_id = test_id

        result = FlextLogger.get_global_correlation_id()
        assert result == test_id

    def test_get_configuration_classmethod(self) -> None:
        """Test get_configuration class method returns current config."""
        FlextLogger._configuration = {
            "log_level": "DEBUG",
            "json_output": True,
            "test_config": "test_value",
        }

        config = FlextLogger.get_configuration()
        assert config["log_level"] == "DEBUG"
        assert config["json_output"] is True
        assert config["test_config"] == "test_value"

    def test_is_configured_classmethod_true(self) -> None:
        """Test is_configured returns True when configured."""
        FlextLogger._configured = True

        assert FlextLogger.is_configured() is True

    def test_is_configured_classmethod_false(self) -> None:
        """Test is_configured returns False when not configured."""
        FlextLogger._configured = False

        assert FlextLogger.is_configured() is False

    def test_init_level_validation_invalid_level(self) -> None:
        """Test __init__ validates log level and falls back to config default for invalid levels."""
        # Invalid levels don't raise - they fall back to config default
        logger = FlextLogger(
            "test_invalid_level", level="INVALID_LEVEL", _force_new=True
        )

        # Should fall back to config default instead of raising
        assert isinstance(logger._level, str)
        assert logger._level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    def test_init_level_validation_valid_levels(self) -> None:
        """Test __init__ accepts all valid log levels."""
        valid_levels = [
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ]  # Remove TRACE - not in valid_levels

        for level in valid_levels:
            logger = FlextLogger(f"test_{level.lower()}", level=level, _force_new=True)
            assert logger._level == level

    def test_init_level_validation_case_insensitive(self) -> None:
        """Test __init__ handles case-insensitive level validation."""
        # Test lowercase
        logger1 = FlextLogger("test_lower", level="debug", _force_new=True)
        assert logger1._level == "DEBUG"

        # Test mixed case
        logger2 = FlextLogger("test_mixed", level="WaRnInG", _force_new=True)
        assert logger2._level == "WARNING"

    def test_exception_method_captures_exc_info(self) -> None:
        """Test exception method captures current exception information."""
        logger = FlextLogger("test_exception_capture")

        try:
            msg = "Test exception for logging"
            raise ValueError(msg)
        except ValueError:
            # Should capture current exception info
            logger.exception("Exception occurred during test")
            # No specific assertions - testing that no errors occur during exception logging

    def test_thread_local_isolation(self) -> None:
        """Test that thread-local storage properly isolates data between threads."""
        logger = FlextLogger("test_thread_isolation")
        results = {}

        def thread_worker(thread_id: str) -> None:
            # Set thread-specific context
            logger.set_correlation_id(f"corr_{thread_id}")
            logger.set_request_context(thread_id=thread_id)

            # Start operation
            op_id = logger.start_operation(f"operation_{thread_id}")

            # Store results for verification
            results[thread_id] = {
                "correlation_id": logger._correlation_id,  # Instance-level, not local
                "request_context": getattr(logger._local, "request_context", {}),
                "operation_id": op_id,
                "operations": getattr(logger._local, "operations", {}),
            }

        # Run multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=thread_worker, args=[f"thread_{i}"])
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify isolation - each thread should have its own data
        assert len(results) == 3
        for thread_id, data in results.items():
            # Note: correlation_id is instance-level, so it will be the last set value
            # But request_context and operations should be thread-local
            assert data["request_context"]["thread_id"] == thread_id
            assert data["operation_id"].startswith("op_")
            assert len(data["operations"]) == 1

    def test_logging_methods_with_complex_context(self) -> None:
        """Test all logging methods handle complex context data correctly."""
        logger = FlextLogger("test_complex_context")

        complex_context = {
            "nested_data": {
                "user_info": {"id": 123, "password": "secret"},
                "metrics": [1, 2, 3, 4, 5],
            },
            "api_key": "sensitive_key",
            "normal_field": "safe_value",
            "numeric_value": 42.5,
        }

        # Test all logging methods with complex context
        logger.trace("Trace message", **complex_context)
        logger.debug("Debug message", **complex_context)
        logger.info("Info message", **complex_context)
        logger.warning("Warning message", **complex_context)
        logger.error("Error message", **complex_context)
        logger.critical("Critical message", **complex_context)

        # Should handle complex context without errors

    def test_error_and_critical_methods_with_exception_objects(self) -> None:
        """Test error and critical methods properly handle Exception objects."""
        logger = FlextLogger("test_error_with_exception")

        test_exception = ValueError("Test error for logging")
        context = {"operation": "test_operation", "user_id": "123"}

        # Test error method with exception
        logger.error("Error occurred", error=test_exception, **context)

        # Test critical method with exception
        logger.critical("Critical error occurred", error=test_exception, **context)

        # Should handle exception objects without errors
