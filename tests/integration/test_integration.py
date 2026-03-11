"""Integration tests for FLEXT Core foundation library.

Enterprise-grade integration testing suite validating cross-component interactions,
service integration patterns, and end-to-end functionality of the FLEXT Core
foundation library.

Architecture:
    Integration Testing → Cross-Component Validation → Service Integration

    This module validates:
    - r integration with FlextContainer dependency injection
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

import pytest

from flext_core import FlextContainer, __version__, r, t, u
from tests.test_utils import assertion_helpers

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
        sample_data: dict[str, t.ContainerValue],
    ) -> None:
        """Test comprehensive integration of core library exports.

        Validates that all primary exports work together seamlessly,
        including r, FlextContainer, and type system integration.

        Args:
            clean_container: Isolated container fixture
            sample_data: Test data fixture

        """
        test_value = str(sample_data["string"])
        result = r[str].ok(test_value)
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == test_value
        entity_id = u.generate()
        assert isinstance(entity_id, str)
        assert len(entity_id) > 0
        register_result = clean_container.register("test_service", test_value)
        assert register_result is clean_container
        service_result = clean_container.get("test_service")
        assert service_result.is_success is True
        assert service_result.value == test_value
        global_container = FlextContainer()
        assert isinstance(global_container, FlextContainer)

    @pytest.mark.integration
    @pytest.mark.core
    def test_flext_result_with_container(
        self,
        clean_container: FlextContainer,
        mock_external_service: FunctionalExternalService,
    ) -> None:
        """Test r integration with DI container factory pattern.

        Validates that r works seamlessly with dependency injection
        factory patterns for service creation and result handling.

        Args:
            clean_container: Isolated container fixture
            mock_external_service: Functional external service

        """
        input_data: str = "container_result"
        expected_result_data: str = f"processed_{input_data}"

        def create_result() -> str:
            process_result = mock_external_service.process(input_data)
            return process_result.unwrap_or("")

        register_result = clean_container.register(
            "result_factory",
            create_result,
            kind="factory",
        )
        assert register_result is clean_container
        factory_result = clean_container.get("result_factory")
        assert factory_result.is_success is True
        result_value = factory_result.value
        assert isinstance(result_value, str)
        assert result_value == expected_result_data
        assert mock_external_service.get_call_count() == 1
        assert expected_result_data in mock_external_service.processed_items

    def test_entity_id_in_flext_result(self) -> None:
        """Test entity ID used in r."""
        entity_id = u.generate()
        result = r[str].ok(entity_id)
        _ = assertion_helpers.assert_flext_result_success(result)
        assert isinstance(result.value, str)
        assert len(result.value) == 36
        assert result.value.count("-") == 4

    def test_version_info_available(self) -> None:
        """Test that version info is available."""
        assert isinstance(__version__, str)
        assert len(__version__) > 0
