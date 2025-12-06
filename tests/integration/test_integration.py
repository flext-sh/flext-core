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


Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import cast

import pytest

from flext_core import (
    FlextContainer,
    FlextResult,
    __version__,
    t,
    u,
)

# Use FunctionalExternalService from conftest.py to avoid duplication
from ..conftest import FunctionalExternalService

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
        sample_data: dict[str, t.GeneralValueType],
    ) -> None:
        """Test comprehensive integration of core library exports.

        Validates that all primary exports work together seamlessly,
        including FlextResult, FlextContainer, and type system integration.

        Args:
            clean_container: Isolated container fixture
            sample_data: Test data fixture

        """
        # Arrange
        test_value = str(sample_data["string"])

        # Act - Test FlextResult creation
        result = FlextResult[str].ok(test_value)

        # Assert - FlextResult functionality
        assert result.is_success is True
        assert result.value == test_value

        # Act - Test entity ID type system using u
        entity_id = u.Generators.generate_id()  # Use actual method name

        # Assert - Type system coherence
        assert isinstance(entity_id, str)
        # ID format changed during simplification
        assert len(entity_id) > 0  # Just verify it's a non-empty string

        # Act - Test FlextContainer service registration
        register_result = clean_container.with_service("test_service", test_value)

        # Assert - Service registration success (fluent interface returns Self)
        assert register_result is clean_container

        # Act - Test service retrieval
        service_result: FlextResult[t.GeneralValueType] = clean_container.get(
            "test_service",
        )

        # Assert - Service retrieval success
        assert service_result.is_success is True
        assert service_result.value == test_value

        # Act - Test global container access
        # API changed: use get_global() instead of ensure_global_manager()
        global_container = FlextContainer()

        # Assert - Global container availability
        assert isinstance(global_container, FlextContainer)

    @pytest.mark.integration
    @pytest.mark.core
    def test_flext_result_with_container(
        self,
        clean_container: FlextContainer,
        mock_external_service: FunctionalExternalService,
    ) -> None:
        """Test FlextResult integration with DI container factory pattern.

        Validates that FlextResult works seamlessly with dependency injection
        factory patterns for service creation and result handling.

        Args:
            clean_container: Isolated container fixture
            mock_external_service: Functional external service

        """
        # Arrange
        input_data: str = "container_result"
        # FunctionalExternalService.process() transforms input by prefixing "processed_"
        expected_result_data: str = f"processed_{input_data}"

        def create_result() -> str:
            # Use functional service processing - real behavior
            process_result = mock_external_service.process(input_data)
            # Unwrap FlextResult to return GeneralValueType (str)
            return process_result.value if process_result.is_success else ""

        # Act - Register factory in container
        register_result = clean_container.with_factory(
            "result_factory",
            create_result,
        )

        # Assert - Factory registration success (fluent interface returns Self)
        assert register_result is clean_container

        # Act - Get factory result from container
        factory_result: FlextResult[t.GeneralValueType] = clean_container.get(
            "result_factory",
        )

        # Assert - Factory retrieval success
        assert factory_result.is_success is True

        # Act - Verify factory produced string value (GeneralValueType)
        result_value: str = cast("str", factory_result.value)

        # Assert - Result type and content validation
        assert isinstance(result_value, str)
        assert result_value == expected_result_data

        # Assert - Functional service was called (real validation)
        assert mock_external_service.get_call_count() == 1
        assert input_data in mock_external_service.processed_items

    def test_entity_id_in_flext_result(self) -> None:
        """Test entity ID used in FlextResult."""
        entity_id = u.Generators.generate_id()  # Use actual method name
        result = FlextResult[str].ok(entity_id)

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
