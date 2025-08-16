"""Coverage tests for aggregate_root.py module."""

from __future__ import annotations

from unittest.mock import patch

from flext_core import FlextAggregateRoot, FlextResult


class TestAggregateRoot(FlextAggregateRoot):
    """Test aggregate root for coverage testing."""

    name: str

    def validate_business_rules(self) -> FlextResult[None]:
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
            error_message: str = "Failed to add domain event: Invalid event type"

            assert error_message in (result.error or "")

    def test_add_domain_event_value_error_handling(self) -> None:
        """Test add_domain_event ValueError handling (lines 261-262)."""
        aggregate = TestAggregateRoot(id="test-123", name="Test")

        # Mock FlextEvent.create_event to raise ValueError
        with patch("flext_core.aggregate_root.FlextEvent.create_event") as mock_create:
            mock_create.side_effect = ValueError("Invalid event data")

            result = aggregate.add_domain_event("test_event", {"invalid": None})

            assert result.is_failure
            error_message: str = "Failed to add domain event: Invalid event data"
            assert error_message in (result.error or "")

    def test_add_domain_event_attribute_error_handling(self) -> None:
        """Test add_domain_event AttributeError handling (lines 261-262)."""
        aggregate = TestAggregateRoot(id="test-123", name="Test")

        # Mock FlextEvent.create_event to raise AttributeError
        with patch("flext_core.aggregate_root.FlextEvent.create_event") as mock_create:
            mock_create.side_effect = AttributeError("Missing attribute")

            result = aggregate.add_domain_event("test_event", {"data": "test"})

            assert result.is_failure
            error_message: str = "Failed to add domain event: Missing attribute"
            assert error_message in (result.error or "")
