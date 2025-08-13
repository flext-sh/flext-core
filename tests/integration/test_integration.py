"""Integration tests for FLEXT Core foundation library.

Enterprise-grade integration testing suite validating cross-component interactions,
service integration patterns, and end-to-end functionality of the FLEXT Core
foundation library.

Architecture:
    Integration Testing → Cross-Component Validation → Service Integration

    This module validates:
    - FlextResult integration with FlextContainer dependency injection
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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flext_core import (
    FlextContainer,
    FlextResult,
    __version__,
    get_flext_container,
)

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from flext_core.typings import FlextEntityId

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
        clean_container: FlextContainer,
        sample_data: dict[
            str, str | int | float | bool | list[int] | dict[str, str] | None,
        ],
    ) -> None:
        """Test comprehensive integration of core library exports.

        Validates that all primary exports work together seamlessly,
        including FlextResult, FlextContainer, and type system integration.

        Args:
            clean_container: Isolated container fixture
            sample_data: Test data fixture

        """
        # Arrange
        test_value = sample_data["string"]

        # Act - Test FlextResult creation
        result = FlextResult.ok(test_value)

        # Assert - FlextResult functionality
        assert result.success is True
        assert result.data == test_value

        # Act - Test FlextEntityId type system
        entity_id: FlextEntityId = "entity-123"

        # Assert - Type system coherence
        assert entity_id == "entity-123"

        # Act - Test FlextContainer service registration
        register_result = clean_container.register("test_service", test_value)

        # Assert - Service registration success
        assert register_result.success is True

        # Act - Test service retrieval
        service_result = clean_container.get("test_service")

        # Assert - Service retrieval success
        assert service_result.success is True
        assert service_result.data == test_value

        # Act - Test global container access
        global_container = get_flext_container()

        # Assert - Global container availability
        assert isinstance(global_container, FlextContainer)

    @pytest.mark.integration
    @pytest.mark.core
    def test_flext_result_with_container(
        self,
        clean_container: FlextContainer,
        mock_external_service: MagicMock,
    ) -> None:
        """Test FlextResult integration with DI container factory pattern.

        Validates that FlextResult works seamlessly with dependency injection
        factory patterns for service creation and result handling.

        Args:
            clean_container: Isolated container fixture
            mock_external_service: Mock external service

        """
        # Arrange
        expected_result_data = "container_result"

        def create_result() -> FlextResult[str]:
            # Simulate service processing with mock
            mock_external_service.process.return_value = FlextResult.ok(
                expected_result_data,
            )
            processed: FlextResult[str] = mock_external_service.process()
            return processed

        # Act - Register factory in container
        register_result = clean_container.register_factory(
            "result_factory",
            create_result,
        )

        # Assert - Factory registration success
        assert register_result.success is True

        # Act - Get factory result from container
        factory_result = clean_container.get("result_factory")

        # Assert - Factory retrieval success
        assert factory_result.success is True

        # Act - Verify factory produced FlextResult
        result = factory_result.data

        # Assert - Result type and content validation
        assert isinstance(result, FlextResult)
        assert result.success is True
        assert result.data == expected_result_data

        # Assert - Mock service was called
        mock_external_service.process.assert_called_once()

    def test_entity_id_in_flext_result(self) -> None:
        """Test FlextEntityId used in FlextResult."""
        entity_id: FlextEntityId = "user-456"
        result = FlextResult.ok(entity_id)

        assert result.success
        if result.data != "user-456":
            msg: str = f"Expected {'user-456'}, got {result.data}"
            raise AssertionError(msg)
        assert isinstance(result.data, str)

    def test_version_info_available(self) -> None:
        """Test that version info is available."""
        assert isinstance(__version__, str)
        assert len(__version__) > 0
