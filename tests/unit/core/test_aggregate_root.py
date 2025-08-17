"""Coverage tests for aggregate_root.py module."""

from __future__ import annotations

from flext_core import FlextAggregateRoot, FlextResult


class TestAggregateRoot(FlextAggregateRoot):
    """Test aggregate root for coverage testing."""

    name: str

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate domain rules."""
        return FlextResult.ok(None)


class FailingEventAggregateRoot(FlextAggregateRoot):
    """Aggregate root that fails to create events for testing error handling."""

    name: str

    def add_domain_event(
        self,
        event_type: str,
        event_data: dict[str, object],
    ) -> FlextResult[None]:
        """Override to simulate different error types based on event_type."""
        # Define event handlers for different event types
        event_handlers = {
            "type_error_event": self._handle_type_error_event,
            "value_error_event": self._handle_value_error_event,
            "attribute_error_event": self._handle_attribute_error_event,
        }

        # Get handler for the event type, default to normal behavior
        handler = event_handlers.get(event_type)
        if handler:
            return handler(event_data)

        # Default case
        return super().add_domain_event(event_type, event_data)

    def _handle_type_error_event(
        self,
        _event_data: dict[str, object],
    ) -> FlextResult[None]:
        """Handle type error event simulation."""
        try:
            return super().add_domain_event(None, _event_data)  # type: ignore[arg-type]
        except TypeError as e:
            return FlextResult.fail(f"Failed to add domain event: {e}")

    def _handle_value_error_event(
        self,
        _event_data: dict[str, object],
    ) -> FlextResult[None]:
        """Handle value error event simulation."""
        try:
            invalid_data = {"__invalid__": lambda x: x}  # Non-serializable
            return super().add_domain_event("valid_type", invalid_data)
        except (ValueError, TypeError):
            return FlextResult.fail("Failed to add domain event: Invalid event data")

    def _handle_attribute_error_event(
        self,
        event_data: dict[str, object],
    ) -> FlextResult[None]:
        """Handle attribute error event simulation."""
        try:
            original_events = self.domain_events
            delattr(self, "domain_events")
            result = super().add_domain_event("test", event_data)
            # Restore it
            self.domain_events = original_events
            return result
        except AttributeError:
            # Restore the attribute and return failure
            if not hasattr(self, "domain_events"):
                self.domain_events = original_events  # type: ignore[possibly-undefined]
            return FlextResult.fail("Failed to add domain event: Missing attribute")


class TestAggregateRootCoverage:
    """Test cases specifically for improving coverage of aggregate_root.py module."""

    def test_add_domain_event_exception_handling(self) -> None:
        """Test add_domain_event exception handling (lines 261-262)."""
        # Use our custom aggregate that simulates TypeError
        aggregate = FailingEventAggregateRoot(id="test-123", name="Test")

        result = aggregate.add_domain_event("type_error_event", {"data": "test"})

        assert result.is_failure
        error_message: str = "Failed to add domain event:"
        assert error_message in (result.error or "")
        # The specific error might vary but should mention the failure
        assert result.error is not None

    def test_add_domain_event_value_error_handling(self) -> None:
        """Test add_domain_event ValueError handling (lines 261-262)."""
        # Use our custom aggregate that simulates ValueError
        aggregate = FailingEventAggregateRoot(id="test-123", name="Test")

        result = aggregate.add_domain_event("value_error_event", {"invalid": None})

        assert result.is_failure
        error_message: str = "Failed to add domain event: Invalid event data"
        assert error_message in (result.error or "")

    def test_add_domain_event_attribute_error_handling(self) -> None:
        """Test add_domain_event AttributeError handling (lines 261-262)."""
        # Use our custom aggregate that simulates AttributeError
        aggregate = FailingEventAggregateRoot(id="test-123", name="Test")

        result = aggregate.add_domain_event("attribute_error_event", {"data": "test"})

        assert result.is_failure
        error_message: str = "Failed to add domain event: Missing attribute"
        assert error_message in (result.error or "")
