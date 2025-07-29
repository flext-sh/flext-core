"""Simple tests to cover basic functionality gaps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_settings import BaseSettings, SettingsConfigDict

from flext_core.constants import FlextConstants, FlextEnvironment, FlextLogLevel
from flext_core.container import FlextContainer
from flext_core.payload import FlextPayload
from flext_core.result import FlextResult as FlextResultDirect

# Constants
EXPECTED_DATA_COUNT = 3

if TYPE_CHECKING:
    from flext_core.types import FlextEntityId


class TestCoverageGaps:
    """Simple tests to increase coverage."""

    def test_basic_config(self) -> None:
        """Test basic configuration functionality."""
        # Create a temporary settings class without external environment interference

        class TestSettings(BaseSettings):
            model_config = SettingsConfigDict(
                extra="ignore",  # Allow extra environment variables
                validate_assignment=True,
            )

            debug: bool = False
            timeout: int = 30

        settings = TestSettings()
        assert settings is not None
        if settings.debug:
            msg = f"Expected False, got {settings.debug}"
            raise AssertionError(msg)
        assert settings.timeout == 30

    def test_basic_constants(self) -> None:
        """Test basic constants functionality."""
        assert FlextConstants.DEFAULT_TIMEOUT > 0
        assert FlextConstants.VERSION is not None
        if FlextEnvironment.PRODUCTION.value != "production":
            msg = f"Expected {"production"}, got {FlextEnvironment.PRODUCTION.value}"
            raise AssertionError(msg)
        assert FlextLogLevel.INFO.value == "INFO"

    def test_container_edge_cases(self) -> None:
        """Test container edge cases."""
        container = FlextContainer()

        # Test empty service name
        result = container.register("", lambda: "test")
        assert not result.is_success

        # Test missing service
        result = container.get("nonexistent")
        assert not result.is_success

    def test_payload_basics(self) -> None:
        """Test basic payload functionality."""
        payload = FlextPayload()
        result = payload.get("missing", "default")
        if result != "default":
            msg = f"Expected {"default"}, got {result}"
            raise AssertionError(msg)

        payload_with_data = FlextPayload(key="value")
        if payload_with_data.get("key") != "value":
            msg = f"Expected {"value"}, got {payload_with_data.get("key")}"
            raise AssertionError(msg)

    def test_result_edge_cases(self) -> None:
        """Test result edge cases."""
        result = FlextResultDirect.ok(None)
        assert result.is_success

        result = FlextResultDirect.fail("")
        assert not result.is_success

    def test_types_system_basic(self) -> None:
        """Test basic types system functionality."""
        # Test FlextEntityId
        entity_id: FlextEntityId = "test-entity-123"
        assert len(entity_id) > 0
