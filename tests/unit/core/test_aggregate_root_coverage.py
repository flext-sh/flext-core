"""Coverage tests for aggregate_root.py module."""

from __future__ import annotations

from unittest.mock import patch

from flext_core.aggregate_root import FlextAggregateRoot
from flext_core.result import FlextResult


class TestAggregateRoot(FlextAggregateRoot):
    """Test aggregate root for coverage testing."""

    name: str

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate domain rules."""
        return FlextResult.ok(None)


class TestAggregateRootCoverage:
    """Test cases specifically for improving coverage of aggregate_root.py module."""

    def test_add_domain_event_exception_handling(self) -> None:
        """Test add_domain_event exception handling (lines 261-262)."""
        aggregate = TestAggregateRoot(id="test-123", name="Test")

        # Mock FlextEvent.create_event to raise TypeError
        with patch("flext_core.aggregate_root.FlextEvent.create_event") as mock_create:
            mock_create.side_effect = TypeError("Invalid event type")

            result = aggregate.add_domain_event("invalid_event", {"data": "test"})

            assert result.is_failure
            assert "Failed to add domain event: Invalid event type" in result.error

    def test_add_domain_event_value_error_handling(self) -> None:
        """Test add_domain_event ValueError handling (lines 261-262)."""
        aggregate = TestAggregateRoot(id="test-123", name="Test")

        # Mock FlextEvent.create_event to raise ValueError
        with patch("flext_core.aggregate_root.FlextEvent.create_event") as mock_create:
            mock_create.side_effect = ValueError("Invalid event data")

            result = aggregate.add_domain_event("test_event", {"invalid": None})

            assert result.is_failure
            assert "Failed to add domain event: Invalid event data" in result.error

    def test_add_domain_event_attribute_error_handling(self) -> None:
        """Test add_domain_event AttributeError handling (lines 261-262)."""
        aggregate = TestAggregateRoot(id="test-123", name="Test")

        # Mock FlextEvent.create_event to raise AttributeError
        with patch("flext_core.aggregate_root.FlextEvent.create_event") as mock_create:
            mock_create.side_effect = AttributeError("Missing attribute")

            result = aggregate.add_domain_event("test_event", {"data": "test"})

            assert result.is_failure
            assert "Failed to add domain event: Missing attribute" in result.error
