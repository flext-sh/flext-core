"""Simple tests to cover basic functionality gaps."""

from __future__ import annotations

import pytest

from flext_core.config import FlextCoreSettings
from flext_core.constants import FlextConstants
from flext_core.constants import FlextEnvironment
from flext_core.constants import FlextLogLevel
from flext_core.container import FlextContainer
from flext_core.payload import FlextPayload
from flext_core.result import FlextResult as FlextResultDirect
from flext_core.types_system import FlextEntityId
from flext_core.types_system import FlextIdentifier


class TestCoverageGaps:
    """Simple tests to increase coverage."""

    def test_basic_config(self) -> None:
        """Test basic configuration functionality."""
        settings = FlextCoreSettings()
        assert settings is not None

    def test_basic_constants(self) -> None:
        """Test basic constants functionality."""
        assert FlextConstants.DEFAULT_TIMEOUT > 0
        assert FlextConstants.VERSION is not None
        assert FlextEnvironment.PRODUCTION == "production"
        assert FlextLogLevel.INFO == "INFO"

    def test_container_edge_cases(self) -> None:
        """Test container edge cases."""
        container = FlextContainer()

        # Test empty service name
        result = container.register("", lambda: "test")
        assert not result.success

        # Test missing service
        result = container.get("nonexistent")
        assert not result.success

    def test_payload_basics(self) -> None:
        """Test basic payload functionality."""
        payload = FlextPayload()
        result = payload.get("missing", "default")
        assert result == "default"

        payload_with_data = FlextPayload(key="value")
        assert payload_with_data.get("key") == "value"

    def test_result_edge_cases(self) -> None:
        """Test result edge cases."""
        result = FlextResultDirect.ok(None)
        assert result.success

        result = FlextResultDirect.fail("")
        assert not result.success

    def test_types_system_basic(self) -> None:
        """Test basic types system functionality."""
        identifier = FlextIdentifier(value="test-id")
        assert str(identifier) == "test-id"

        # Test FlextEntityId
        entity_id: FlextEntityId = "test-entity-123"
        assert len(entity_id) > 0

    def test_identifier_validation(self) -> None:
        """Test identifier validation."""
        with pytest.raises(ValueError, match=".*"):
            FlextIdentifier(value="")
