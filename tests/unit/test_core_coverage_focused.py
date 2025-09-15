"""Focused tests for FlextCore to achieve 100% coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from unittest.mock import patch

from flext_core import FlextCore


class TestFlextCoreCoverageFocused:
    """Focused FlextCore tests targeting 100% coverage."""

    def test_core_get_instance_initialization_edge_case(self) -> None:
        """Test FlextCore get_instance when _instance is None."""
        # Reset instance to ensure we test the initialization path
        FlextCore.reset_instance()

        # Test that get_instance creates new instance when None
        instance1 = FlextCore.get_instance()
        assert instance1 is not None
        assert isinstance(instance1, FlextCore)

        # Test singleton behavior - should return same instance
        instance2 = FlextCore.get_instance()
        assert instance1 is instance2

    def test_core_config_initialization_with_exception_handling(self) -> None:
        """Test FlextCore config initialization with potential exception paths."""
        # Reset to test initialization
        FlextCore.reset_instance()

        # Test with potential FlextConfig initialization issues
        with patch("flext_core.core.FlextConfig"):
            # Mock config class to potentially raise exception
            instance = FlextCore.get_instance()
            config = instance.get_config()

            # Verify config is accessible
            assert config is not None

    def test_core_session_id_generation_edge_cases(self) -> None:
        """Test session ID generation edge cases."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Test session ID generation
        session_id = instance.get_session_id()
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0

        # Test that multiple calls return same session ID
        session_id2 = instance.get_session_id()
        assert session_id == session_id2

    def test_core_cleanup_operation_comprehensive(self) -> None:
        """Test cleanup operation with comprehensive coverage."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Ensure container and other resources are initialized
        container = instance.container
        assert container is not None

        # Test cleanup
        cleanup_result = instance.cleanup()
        assert cleanup_result.is_success

        # Cleanup should complete without error
        # Test multiple cleanups don't cause issues
        cleanup_result2 = instance.cleanup()
        assert cleanup_result2.is_success

    def test_core_string_representations(self) -> None:
        """Test string representation methods."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Test __str__ method
        str_repr = str(instance)
        assert "FlextCore" in str_repr
        assert "id=" in str_repr

        # Test __repr__ method
        repr_str = repr(instance)
        assert "FlextCore" in repr_str
        assert instance.entity_id in repr_str

    def test_core_container_access_edge_cases(self) -> None:
        """Test container property access edge cases."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Test direct container access
        container = instance.container
        assert container is not None

        # Test that container property is consistent
        container2 = instance.container
        assert container is container2

    def test_core_facade_access_patterns(self) -> None:
        """Test facade access patterns for comprehensive coverage."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Test accessing various facades through the core instance
        assert hasattr(instance, "Config")
        assert hasattr(instance, "Container")
        assert hasattr(instance, "Result")
        assert hasattr(instance, "Models")
        assert hasattr(instance, "Utilities")
        assert hasattr(instance, "Validations")

        # Verify these are not None
        assert instance.Config is not None
        assert instance.Container is not None
        assert instance.Result is not None

    def test_core_instance_lifecycle_management(self) -> None:
        """Test instance lifecycle: reset, create, reset, create new."""
        # Start with clean state
        FlextCore.reset_instance()
        assert FlextCore._instance is None

        # Create instance
        instance1 = FlextCore.get_instance()
        assert FlextCore._instance is instance1

        # Reset and verify
        FlextCore.reset_instance()
        assert FlextCore._instance is None

        # Create new instance
        instance2 = FlextCore.get_instance()
        assert FlextCore._instance is instance2
        assert instance1 is not instance2

    def test_core_configuration_access_patterns(self) -> None:
        """Test configuration access patterns."""
        FlextCore.reset_instance()
        instance = FlextCore.get_instance()

        # Test config access
        config = instance.get_config()
        assert config is not None

        # Test that config is consistent across calls
        config2 = instance.get_config()
        assert config is config2

    def test_core_exception_handling_during_init(self) -> None:
        """Test exception handling during initialization."""
        FlextCore.reset_instance()

        # Test that get_instance handles potential initialization issues gracefully
        try:
            instance = FlextCore.get_instance()
            assert instance is not None
        except Exception as e:
            # If any exception occurs during initialization, it should be handled
            raise AssertionError(
                f"get_instance should not raise exceptions: {e}"
            ) from e
