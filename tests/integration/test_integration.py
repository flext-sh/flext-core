"""Integration tests for FLEXT Core foundation library.

Enterprise-grade integration testing suite validating cross-component interactions,
service integration patterns, and end-to-end functionality of the FLEXT Core
foundation library.

Architecture:
    Integration Testing → Cross-Component Validation → Service Integration

    This module validates:
    - FlextCore.Result integration with FlextCore.Container dependency injection
    - Type system coherence across foundation patterns
    - Service registration and retrieval workflows
    - Mock-based external service integration patterns
    - Performance characteristics of integrated components

Integration Testing Strategy:
    - Component Interaction: Test how core components work together
    - Service Integration: Validate DI container with mocked services
    - Type Safety: Ensure type system works across component boundaries
    - Performance: Validate integrated workflows meet performance standards
    - Error Handling: Test error propagation across component boundaries


Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import cast

import pytest

from flext_core import FlextCore, __version__


class FunctionalExternalService:
    """Functional external service for integration testing - real implementation."""

    def __init__(self) -> None:
        """Initialize functional external service with processing state."""
        super().__init__()
        self.call_count = 0
        self.processed_items: FlextCore.Types.StringList = []
        self.should_fail = False
        self.failure_message = "Service unavailable"

    def process(self, data: str | None = None) -> FlextCore.Result[str]:
        """Functional processing method - validates real behavior.

        Returns:
            FlextCore.Result[str]: Success with processed data or failure with error message.

        """
        self.call_count += 1

        if self.should_fail:
            return FlextCore.Result[str].fail(self.failure_message)

        processed_data = data or "processed"
        self.processed_items.append(processed_data)

        return FlextCore.Result[str].ok(processed_data)

    def set_failure_mode(
        self,
        *,
        should_fail: bool,
        message: str = "Service unavailable",
    ) -> None:
        """Configure service to fail for testing error scenarios."""
        self.should_fail = should_fail
        self.failure_message = message

    def get_call_count(self) -> int:
        """Get number of times process was called.

        Returns:
            int: Number of times the process method was called.

        """
        return self.call_count

    def reset(self) -> None:
        """Reset service state."""
        self.call_count = 0
        self.processed_items.clear()
        self.should_fail = False


@pytest.fixture
def mock_external_service() -> FunctionalExternalService:
    """Functional external service for integration testing.

    Returns:
        FunctionalExternalService: A configured external service instance.

    """
    return FunctionalExternalService()


pytestmark = [pytest.mark.integration]


class TestLibraryIntegration:
    """Integration tests for FLEXT Core library components.

    Validates that core foundation components work together correctly
    and provide consistent behavior across the ecosystem.
    """

    @pytest.mark.integration
    @pytest.mark.core
    def test_all_exports_work(
        self,
        clean_container: FlextCore.Container,
        sample_data: FlextCore.Types.Dict,
    ) -> None:
        """Test comprehensive integration of core library exports.

        Validates that all primary exports work together seamlessly,
        including FlextCore.Result, FlextCore.Container, and type system integration.

        Args:
            clean_container: Isolated container fixture
            sample_data: Test data fixture

        """
        # Arrange
        test_value = str(sample_data["string"])

        # Act - Test FlextCore.Result creation
        result = FlextCore.Result[str].ok(test_value)

        # Assert - FlextCore.Result functionality
        assert result.is_success is True
        assert result.value == test_value

        # Act - Test entity ID type system using FlextCore.Utilities
        entity_id = (
            FlextCore.Utilities.Generators.generate_id()
        )  # Use actual method name

        # Assert - Type system coherence
        assert isinstance(entity_id, str)
        # ID format changed during simplification
        assert len(entity_id) > 0  # Just verify it's a non-empty string

        # Act - Test FlextCore.Container service registration
        register_result = clean_container.register("test_service", test_value)

        # Assert - Service registration success
        assert register_result.is_success is True

        # Act - Test service retrieval
        service_result = clean_container.get("test_service")

        # Assert - Service retrieval success
        assert service_result.is_success is True
        assert service_result.value == test_value

        # Act - Test global container access
        # API changed: use get_global() instead of ensure_global_manager()
        global_container = FlextCore.Container.get_global()

        # Assert - Global container availability
        assert isinstance(global_container, FlextCore.Container)

    @pytest.mark.integration
    @pytest.mark.core
    def test_flext_result_with_container(
        self,
        clean_container: FlextCore.Container,
        mock_external_service: FunctionalExternalService,
    ) -> None:
        """Test FlextCore.Result integration with DI container factory pattern.

        Validates that FlextCore.Result works seamlessly with dependency injection
        factory patterns for service creation and result handling.

        Args:
            clean_container: Isolated container fixture
            mock_external_service: Functional external service

        """
        # Arrange
        expected_result_data: str = "container_result"

        def create_result() -> FlextCore.Result[str]:
            # Use functional service processing - real behavior
            return mock_external_service.process(expected_result_data)

        # Act - Register factory in container
        register_result = clean_container.register_factory(
            "result_factory",
            create_result,
        )

        # Assert - Factory registration success
        assert register_result.is_success is True

        # Act - Get factory result from container
        factory_result = clean_container.get("result_factory")

        # Assert - Factory retrieval success
        assert factory_result.is_success is True

        # Act - Verify factory produced FlextCore.Result
        result: FlextCore.Result[str] = cast(
            "FlextCore.Result[str]", factory_result.value
        )

        # Assert - Result type and content validation
        assert isinstance(result, FlextCore.Result)
        assert result.is_success is True
        assert result.value == expected_result_data

        # Assert - Functional service was called (real validation)
        assert mock_external_service.get_call_count() == 1
        assert expected_result_data in mock_external_service.processed_items

    def test_entity_id_in_flext_result(self) -> None:
        """Test entity ID used in FlextCore.Result."""
        entity_id = (
            FlextCore.Utilities.Generators.generate_id()
        )  # Use actual method name
        result = FlextCore.Result[str].ok(entity_id)

        assert result.is_success
        # Entity ID is a valid UUID string (36 chars)
        assert isinstance(result.value, str)
        assert len(result.value) == 36  # UUIDs are 36 character strings
        # Verify it's a valid UUID format (contains hyphens at expected positions)
        assert (
            result.value.count("-") == 4
        )  # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

    def test_version_info_available(self) -> None:
        """Test that version info is available."""
        assert isinstance(__version__, str)
        assert len(__version__) > 0
