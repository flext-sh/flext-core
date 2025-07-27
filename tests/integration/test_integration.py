"""Integration tests for pure library."""

from __future__ import annotations

import pytest

from flext_core import (
    FlextContainer,
    FlextEntityId,
    FlextResult,
    __version__,
    get_flext_container,
)

pytestmark = [pytest.mark.integration]


class TestLibraryIntegration:
    """Test library integration and main exports."""

    def test_all_exports_work(self) -> None:
        """Test that all main exports work together."""
        # Test FlextResult
        result = FlextResult.ok("test")
        assert result.is_success

        # Test FlextEntityId
        entity_id: FlextEntityId = "entity-123"
        assert entity_id == "entity-123"

        # Test FlextContainer
        container = FlextContainer()
        register_result = container.register("service", "value")
        assert register_result.is_success

        service_result = container.get("service")
        assert service_result.is_success
        assert service_result.data == "value"

        # Test global container
        global_container = get_flext_container()
        assert isinstance(global_container, FlextContainer)

    def test_flext_result_with_container(self) -> None:
        """Test FlextResult working with DI container."""
        container = FlextContainer()

        def create_result() -> FlextResult[str]:
            return FlextResult.ok("container_result")

        register_result = container.register_singleton(
            "result_factory",
            create_result,
        )
        assert register_result.is_success

        # The container calls the factory and returns the result
        factory_result = container.get("result_factory")
        assert factory_result.is_success

        result = factory_result.data
        assert isinstance(result, FlextResult)
        assert result.data == "container_result"

    def test_entity_id_in_flext_result(self) -> None:
        """Test FlextEntityId used in FlextResult."""
        entity_id: FlextEntityId = "user-456"
        result = FlextResult.ok(entity_id)

        assert result.is_success
        assert result.data == "user-456"
        assert isinstance(result.data, str)

    def test_version_info_available(self) -> None:
        """Test that version info is available."""
        assert isinstance(__version__, str)
        assert len(__version__) > 0
