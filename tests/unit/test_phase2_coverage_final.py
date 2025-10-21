"""Final Phase 2 coverage push - targeted tests to reach 75% threshold.

This test file contains strategic tests targeting the remaining ~86 uncovered lines
needed to reach Phase 2 completion at 75% coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core import FlextConfig, FlextResult, FlextUtilities


class TestPhase2FinalCoveragePush:
    """Strategic tests targeting remaining coverage gaps."""

    def test_config_callable_field_access(self) -> None:
        """Test config callable interface for field access."""
        config = FlextConfig(app_name="test_app")
        # Test callable access
        assert config("app_name") == "test_app"

    def test_config_callable_nested_access(self) -> None:
        """Test config callable with multiple field types."""
        config = FlextConfig(
            app_name="myapp",
            version="1.0.0",
            debug=True,
            trace=False,
        )
        assert config("debug") is True
        assert config("trace") is False

    def test_flext_result_chaining_operations(self) -> None:
        """Test FlextResult chaining with multiple operations."""
        # Test successful chaining
        result = (
            FlextResult[str].ok("hello")
            .map(str.upper)
            .map(lambda x: f"{x}!")
        )
        assert result.is_success
        assert result.value == "HELLO!"

    def test_flext_result_flat_map_chaining(self) -> None:
        """Test FlextResult flat_map chaining."""
        def double_string(s: str) -> FlextResult[str]:
            """Double the string or fail."""
            if len(s) > 10:
                return FlextResult[str].fail("Too long")
            return FlextResult[str].ok(s + s)

        result = (
            FlextResult[str].ok("hi")
            .flat_map(double_string)
            .flat_map(double_string)
        )
        assert result.is_success
        assert result.value == "hihihihi"

    def test_flext_result_error_propagation(self) -> None:
        """Test FlextResult error propagation in chain."""
        def failing_op(s: str) -> FlextResult[str]:
            """Operation that fails."""
            return FlextResult[str].fail("operation failed")

        result = (
            FlextResult[str].ok("input")
            .flat_map(failing_op)
            .map(str.upper)  # Should not execute
        )
        assert result.is_failure
        assert result.error == "operation failed"

    def test_config_with_all_field_types(self) -> None:
        """Test config with all field types together."""
        config = FlextConfig(
            app_name="complete_test",
            version="2.0.0",
            debug=True,
            trace=False,
            max_retry_attempts=5,
            timeout_seconds=60.0,
        )
        assert config.app_name == "complete_test"
        assert config.max_retry_attempts == 5
        assert config.timeout_seconds == 60.0

    def test_utilities_is_non_empty_string(self) -> None:
        """Test utilities non-empty string check."""
        assert FlextUtilities.Validation.is_non_empty_string("hello") is True
        assert FlextUtilities.Validation.is_non_empty_string("") is False
        assert FlextUtilities.Validation.is_non_empty_string("   ") is False
        assert FlextUtilities.Validation.is_non_empty_string(None) is False

    def test_config_json_serialization(self) -> None:
        """Test config JSON serialization and deserialization."""
        original = FlextConfig(
            app_name="json_test",
            version="1.0.0",
            debug=False,
        )
        # Serialize
        config_dict = original.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "json_test"
        # Deserialize
        new_config = FlextConfig(**config_dict)
        assert new_config.app_name == original.app_name
        assert new_config.version == original.version

    def test_flext_result_unwrap_safe_operations(self) -> None:
        """Test FlextResult unwrap and value operations."""
        success_result = FlextResult[str].ok("success_value")
        assert success_result.unwrap() == "success_value"
        assert success_result.value == "success_value"
        assert success_result.data == "success_value"

    def test_flext_result_is_methods(self) -> None:
        """Test FlextResult boolean check methods."""
        success = FlextResult[str].ok("test")
        assert success.is_success is True
        assert success.is_failure is False

        failure = FlextResult[str].fail("error")
        assert failure.is_success is False
        assert failure.is_failure is True

    def test_config_field_assignment_and_retrieval(self) -> None:
        """Test config field assignment and retrieval patterns."""
        config = FlextConfig()
        # All fields should have sensible defaults
        assert isinstance(config.app_name, str)
        assert isinstance(config.version, str)
        assert isinstance(config.debug, bool)
        assert isinstance(config.trace, bool)

    def test_config_instance_reset_and_recreate(self) -> None:
        """Test config instance reset and recreation."""
        FlextConfig.reset_global_instance()
        config = FlextConfig(app_name="reset_test")
        assert config.app_name == "reset_test"
        FlextConfig.reset_global_instance()
        config2 = FlextConfig(app_name="reset_test_2")
        assert config2.app_name == "reset_test_2"


__all__ = ["TestPhase2FinalCoveragePush"]
