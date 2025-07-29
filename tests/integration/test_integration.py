"""Integration tests for pure library."""

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
    from flext_core.types import FlextEntityId

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
        if entity_id != "entity-123":
            msg = f"Expected {"entity-123"}, got {entity_id}"
            raise AssertionError(msg)

        # Test FlextContainer
        container = FlextContainer()
        register_result = container.register("service", "value")
        assert register_result.is_success

        service_result = container.get("service")
        assert service_result.is_success
        if service_result.data != "value":
            msg = f"Expected {"value"}, got {service_result.data}"
            raise AssertionError(msg)

        # Test global container
        global_container = get_flext_container()
        assert isinstance(global_container, FlextContainer)

    def test_flext_result_with_container(self) -> None:
        """Test FlextResult working with DI container."""
        container = FlextContainer()

        def create_result() -> FlextResult[str]:
            return FlextResult.ok("container_result")

        register_result = container.register_factory(
            "result_factory",
            create_result,
        )
        assert register_result.is_success

        # The container calls the factory and returns the result
        factory_result = container.get("result_factory")
        assert factory_result.is_success

        result = factory_result.data
        assert isinstance(result, FlextResult)
        if result.data != "container_result":
            msg = f"Expected {"container_result"}, got {result.data}"
            raise AssertionError(msg)

    def test_entity_id_in_flext_result(self) -> None:
        """Test FlextEntityId used in FlextResult."""
        entity_id: FlextEntityId = "user-456"
        result = FlextResult.ok(entity_id)

        assert result.is_success
        if result.data != "user-456":
            msg = f"Expected {"user-456"}, got {result.data}"
            raise AssertionError(msg)
        assert isinstance(result.data, str)

    def test_version_info_available(self) -> None:
        """Test that version info is available."""
        assert isinstance(__version__, str)
        assert len(__version__) > 0
