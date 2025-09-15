"""Targeted FlextCore exception coverage tests for lines 100-101.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Never
from unittest.mock import patch

from flext_core import FlextCore
from flext_tests import FlextTestsMatchers


class TestFlextCoreExceptionCoverage:
    """Targeted tests for FlextCore cleanup method exception handling (lines 100-101)."""

    def test_cleanup_exception_handling_path(self) -> None:
        """Test cleanup method exception handling - covers lines 100-101."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Mock _generate_session_id to raise an exception during cleanup
        with patch.object(
            instance,
            "_generate_session_id",
            side_effect=Exception("Forced cleanup error"),
        ):
            result = instance.cleanup()

            # Should handle exception and return failure result
            FlextTestsMatchers.assert_result_failure(result)
            assert result.error
            assert result.error is not None
            assert "Cleanup failed: Forced cleanup error" in result.error

    def test_cleanup_exception_handling_with_different_error_types(self) -> None:
        """Test cleanup method with various exception types."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Test with ValueError
        with patch.object(
            instance,
            "_generate_session_id",
            side_effect=ValueError("Invalid session generation"),
        ):
            result = instance.cleanup()
            FlextTestsMatchers.assert_result_failure(result)
            assert result.error
            assert result.error is not None
            assert "Cleanup failed: Invalid session generation" in result.error

        # Test with RuntimeError
        with patch.object(
            instance,
            "_generate_session_id",
            side_effect=RuntimeError("Runtime issue in cleanup"),
        ):
            result = instance.cleanup()
            FlextTestsMatchers.assert_result_failure(result)
            assert result.error
            assert result.error is not None
            assert "Cleanup failed: Runtime issue in cleanup" in result.error

        # Test with generic Exception
        with patch.object(
            instance, "_generate_session_id", side_effect=Exception("Generic error")
        ):
            result = instance.cleanup()
            FlextTestsMatchers.assert_result_failure(result)
            assert result.error
            assert result.error is not None
            assert "Cleanup failed: Generic error" in result.error

    def test_cleanup_exception_handling_with_complex_error_message(self) -> None:
        """Test cleanup exception handling with complex error messages."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Test with complex error containing special characters
        complex_error_msg = (
            "Session generation failed: ID='test_123', timestamp=None, error_code=500"
        )
        with patch.object(
            instance, "_generate_session_id", side_effect=Exception(complex_error_msg)
        ):
            result = instance.cleanup()
            FlextTestsMatchers.assert_result_failure(result)
            assert f"Cleanup failed: {complex_error_msg}" in result.error

    def test_cleanup_success_path_still_works(self) -> None:
        """Verify cleanup success path still works after exception testing."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Normal cleanup should still work
        result = instance.cleanup()
        FlextTestsMatchers.assert_result_success(result)

        # Cleanup should reset session_id
        old_session_id = instance.get_session_id()
        result = instance.cleanup()
        FlextTestsMatchers.assert_result_success(result)
        new_session_id = instance.get_session_id()

        # Session ID should be different after cleanup
        assert old_session_id != new_session_id

    def test_cleanup_exception_during_uuid_generation(self) -> None:
        """Test cleanup exception when UUID generation fails."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Mock uuid.uuid4 to fail during session ID generation
        with patch(
            "flext_core.core.uuid.uuid4",
            side_effect=Exception("UUID generation failed"),
        ):
            result = instance.cleanup()
            FlextTestsMatchers.assert_result_failure(result)
            assert result.error
            assert result.error is not None
            assert "Cleanup failed: UUID generation failed" in result.error

    def test_cleanup_exception_during_timestamp_generation(self) -> None:
        """Test cleanup exception when timestamp generation fails."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Mock the datetime module at the module level to avoid immutable type issues
        with patch("flext_core.core.datetime") as mock_datetime:
            mock_datetime.now.side_effect = Exception("Timestamp generation failed")
            mock_datetime.now.return_value = None  # Ensure it fails
            result = instance.cleanup()
            FlextTestsMatchers.assert_result_failure(result)
            assert result.error
            assert result.error is not None
            assert "Cleanup failed: Timestamp generation failed" in result.error

    def test_cleanup_exception_preserves_original_session_id(self) -> None:
        """Test that cleanup exception preserves original session ID."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Get original session ID
        original_session_id = instance.get_session_id()

        # Force exception during cleanup
        with patch.object(
            instance,
            "_generate_session_id",
            side_effect=Exception("Session generation failed"),
        ):
            result = instance.cleanup()
            FlextTestsMatchers.assert_result_failure(result)

            # Session ID should remain unchanged after failed cleanup
            current_session_id = instance.get_session_id()
            assert current_session_id == original_session_id

    def test_cleanup_multiple_exception_scenarios(self) -> None:
        """Test multiple consecutive cleanup exceptions."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # First exception
        with patch.object(
            instance, "_generate_session_id", side_effect=Exception("First error")
        ):
            result1 = instance.cleanup()
            FlextTestsMatchers.assert_result_failure(result1)
            assert result1.error is not None
            assert "Cleanup failed: First error" in result1.error

        # Second exception
        with patch.object(
            instance, "_generate_session_id", side_effect=Exception("Second error")
        ):
            result2 = instance.cleanup()
            FlextTestsMatchers.assert_result_failure(result2)
            assert result2.error is not None
            assert "Cleanup failed: Second error" in result2.error

        # Recovery - normal cleanup should work
        result3 = instance.cleanup()
        FlextTestsMatchers.assert_result_success(result3)

    def test_cleanup_exception_error_message_formatting(self) -> None:
        """Test that exception error messages are properly formatted in cleanup failures."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Test with empty error message
        with patch.object(instance, "_generate_session_id", side_effect=Exception("")):
            result = instance.cleanup()
            FlextTestsMatchers.assert_result_failure(result)
            assert result.error
            assert result.error is not None
            assert "Cleanup failed: " in result.error

        # Test with None-like error
        with patch.object(
            instance, "_generate_session_id", side_effect=Exception("None")
        ):
            result = instance.cleanup()
            FlextTestsMatchers.assert_result_failure(result)
            assert result.error
            assert result.error is not None
            assert "Cleanup failed: None" in result.error

    def test_cleanup_exception_with_nested_method_failure(self) -> None:
        """Test cleanup exception when nested method calls fail."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Create a more realistic failure scenario
        def failing_session_generator() -> Never:
            # Simulate failure in a nested operation
            error_message = "Database connection failed during session generation"
            raise RuntimeError(error_message)

        with patch.object(
            instance, "_generate_session_id", side_effect=failing_session_generator
        ):
            result = instance.cleanup()
            FlextTestsMatchers.assert_result_failure(result)
            assert (
                "Cleanup failed: Database connection failed during session generation"
                in result.error
            )
