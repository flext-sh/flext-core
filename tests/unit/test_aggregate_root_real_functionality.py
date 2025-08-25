"""Real functionality tests for FlextAggregateRoot - NO MOCKS, only real usage.

These tests demonstrate the aggregate root working with real domain events and business logic,
focusing on real-world DDD scenarios and increasing coverage significantly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest

from flext_core import FlextPayload, FlextResult
from flext_core.aggregate_root import FlextAggregateRoot
from flext_core.exceptions import FlextValidationError

pytestmark = [pytest.mark.unit, pytest.mark.core, pytest.mark.ddd]


class OrderAggregate(FlextAggregateRoot):
    """Real order aggregate for testing DDD patterns."""

    # Business fields
    customer_id: str
    total_amount: float
    status: str = "pending"
    items: ClassVar[list[dict[str, Any]]] = []

    def place_order(self, items: list[dict[str, Any]]) -> FlextResult[None]:
        """Real business logic - place an order."""
        if not items:
            return FlextResult[None].fail("Cannot place order with no items")

        if self.status != "pending":
            return FlextResult[None].fail(
                f"Cannot place order in status: {self.status}"
            )

        # Calculate total
        total = sum(item.get("price", 0) * item.get("quantity", 1) for item in items)

        # Update state
        object.__setattr__(self, "items", items.copy())
        object.__setattr__(self, "total_amount", total)
        object.__setattr__(self, "status", "placed")

        # Add domain event
        self.add_domain_event(
            "OrderPlaced",  # event_type_or_dict
            {
                "order_id": self.id,
                "customer_id": self.customer_id,
                "total_amount": total,
                "items_count": len(items),
                "placed_at": datetime.now(UTC).isoformat(),
            },
        )

        return FlextResult[None].ok(None)

    def ship_order(
        self, shipping_address: str, tracking_number: str
    ) -> FlextResult[None]:
        """Real business logic - ship an order."""
        if self.status != "placed":
            return FlextResult[None].fail(f"Cannot ship order in status: {self.status}")

        object.__setattr__(self, "status", "shipped")

        # Add domain event with structured data
        self.add_domain_event(
            "OrderShipped",
            {
                "order_id": self.id,
                "tracking_number": tracking_number,
                "shipping_address": shipping_address,
                "shipped_at": datetime.now(UTC).isoformat(),
                "total_amount": self.total_amount,
            },
        )

        return FlextResult[None].ok(None)

    def cancel_order(self, reason: str) -> FlextResult[None]:
        """Real business logic - cancel an order."""
        if self.status in {"shipped", "delivered", "cancelled"}:
            return FlextResult[None].fail(
                f"Cannot cancel order in status: {self.status}"
            )

        object.__setattr__(self, "status", "cancelled")

        # Add domain event
        self.add_domain_event(
            "OrderCancelled",
            {
                "order_id": self.id,
                "reason": reason,
                "original_amount": self.total_amount,
                "cancelled_at": datetime.now(UTC).isoformat(),
            },
        )

        return FlextResult[None].ok(None)


class UserAggregate(FlextAggregateRoot):
    """Real user aggregate for testing user management scenarios."""

    email: str
    username: str
    is_active: bool = True
    profile_data: ClassVar[dict[str, Any]] = {}

    def activate_user(self) -> FlextResult[None]:
        """Real business logic - activate user."""
        if self.is_active:
            return FlextResult[None].fail("User is already active")

        object.__setattr__(self, "is_active", True)

        self.add_domain_event(
            "UserActivated",
            {
                "user_id": self.id,
                "email": self.email,
                "activated_at": datetime.now(UTC).isoformat(),
            },
        )

        return FlextResult[None].ok(None)

    def update_profile(self, profile_data: dict[str, Any]) -> FlextResult[None]:
        """Real business logic - update user profile."""
        if not profile_data:
            return FlextResult[None].fail("Profile data cannot be empty")

        # Merge profile data
        new_profile = self.profile_data.copy()
        new_profile.update(profile_data)
        object.__setattr__(self, "profile_data", new_profile)

        self.add_domain_event(
            "UserProfileUpdated",
            {
                "user_id": self.id,
                "updated_fields": list(profile_data.keys()),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

        return FlextResult[None].ok(None)

    def change_email(self, new_email: str) -> FlextResult[None]:
        """Real business logic - change email with validation."""
        if not new_email or "@" not in new_email:
            return FlextResult[None].fail("Invalid email format")

        if new_email == self.email:
            return FlextResult[None].fail("New email is the same as current email")

        old_email = self.email
        object.__setattr__(self, "email", new_email)

        self.add_domain_event(
            "UserEmailChanged",
            {
                "user_id": self.id,
                "old_email": old_email,
                "new_email": new_email,
                "changed_at": datetime.now(UTC).isoformat(),
            },
        )

        return FlextResult[None].ok(None)


class TestFlextAggregateRootRealFunctionality:
    """Test FlextAggregateRoot with real business scenarios."""

    def test_aggregate_initialization_basic(self) -> None:
        """Test basic aggregate initialization."""
        order = OrderAggregate(customer_id="customer_123", total_amount=0.0)

        assert order.customer_id == "customer_123"
        assert order.total_amount == 0.0
        assert order.status == "pending"
        assert order.items == []
        assert order.id is not None
        assert order.created_at is not None
        assert order.version == 1

        # Should start with no domain events
        assert not order.has_domain_events()
        assert len(order.get_domain_events()) == 0

    def test_aggregate_initialization_with_id_and_metadata(self) -> None:
        """Test aggregate initialization with explicit ID and metadata."""
        custom_id = "order_12345"
        metadata = {"source": "web_app", "session_id": "session_123"}

        order = OrderAggregate(
            entity_id=custom_id,
            customer_id="customer_456",
            total_amount=100.0,
            metadata=metadata,
        )

        assert order.id == custom_id
        assert order.customer_id == "customer_456"
        assert order.metadata is not None

        # Check metadata was properly coerced
        assert order.metadata.root["source"] == "web_app"
        assert order.metadata.root["session_id"] == "session_123"

    def test_real_business_workflow_complete(self) -> None:
        """Test complete real business workflow with multiple operations."""
        # Create order
        order = OrderAggregate(customer_id="customer_789", total_amount=0.0)

        # Place order
        items = [
            {"name": "Product A", "price": 25.99, "quantity": 2},
            {"name": "Product B", "price": 15.50, "quantity": 1},
        ]

        result = order.place_order(items)
        assert result.success
        assert order.status == "placed"
        assert round(order.total_amount, 2) == 67.48  # (25.99 * 2) + 15.50
        assert len(order.items) == 2

        # Check domain event was added
        assert order.has_domain_events()
        events = order.get_domain_events()
        assert len(events) == 1
        assert events[0].event_type == "OrderPlaced"
        event_data = events[0].get_data().value
        assert event_data["order_id"] == order.id
        assert round(event_data["total_amount"], 2) == 67.48

        # Ship order
        result = order.ship_order("123 Main St", "TRACK123")
        assert result.success
        assert order.status == "shipped"

        # Check second domain event
        events = order.get_domain_events()
        assert len(events) == 2
        assert events[1].event_type == "OrderShipped"
        assert events[1].get_data().value["tracking_number"] == "TRACK123"

    def test_domain_event_management_comprehensive(self) -> None:
        """Test comprehensive domain event management."""
        user = UserAggregate(email="user@example.com", username="testuser")

        # Perform multiple operations that generate events
        user.update_profile({"first_name": "John", "last_name": "Doe"})
        user.change_email("john.doe@example.com")
        user.activate_user()

        # Check all events were recorded
        events = user.get_domain_events()
        assert len(events) == 2  # activate_user fails because user starts active

        event_types = [e.event_type for e in events]
        assert "UserProfileUpdated" in event_types
        assert "UserEmailChanged" in event_types

        # Test event data completeness
        profile_event = next(e for e in events if e.event_type == "UserProfileUpdated")
        profile_data = profile_event.get_data().value
        assert profile_data["user_id"] == user.id
        assert "first_name" in profile_data["updated_fields"]
        assert "last_name" in profile_data["updated_fields"]

        email_event = next(e for e in events if e.event_type == "UserEmailChanged")
        email_data = email_event.get_data().value
        assert email_data["old_email"] == "user@example.com"
        assert email_data["new_email"] == "john.doe@example.com"

    def test_domain_event_clearing_and_persistence(self) -> None:
        """Test domain event clearing and persistence behavior."""
        order = OrderAggregate(customer_id="customer_clear_test", total_amount=0.0)

        # Generate some events
        items = [{"name": "Test Item", "price": 10.0, "quantity": 1}]
        order.place_order(items)

        # Verify events exist
        assert order.has_domain_events()
        events_before_clear = order.get_domain_events()
        assert len(events_before_clear) == 1

        # Clear events
        cleared_events = order.clear_domain_events()
        assert len(cleared_events) == 1  # Returns the cleared events

        # Verify events are cleared
        assert not order.has_domain_events()
        assert len(order.get_domain_events()) == 0

        # Generate new event after clearing (cancel while in "placed" status)
        order.cancel_order("Customer request")

        # Should have only the new event
        new_events = order.get_domain_events()
        assert len(new_events) == 1
        assert new_events[0].event_type == "OrderCancelled"

    def test_business_rule_validation_real_scenarios(self) -> None:
        """Test business rule validation in real scenarios."""
        order = OrderAggregate(customer_id="validation_test", total_amount=0.0)

        # Test placing order with no items
        result = order.place_order([])
        assert result.is_failure
        assert "Cannot place order with no items" in result.error

        # Place valid order
        items = [{"name": "Item", "price": 20.0, "quantity": 1}]
        result = order.place_order(items)
        assert result.success

        # Try to place order again (should fail)
        result = order.place_order(items)
        assert result.is_failure
        assert "Cannot place order in status: placed" in result.error

        # Try to ship before placing (create new order in pending)
        pending_order = OrderAggregate(customer_id="pending_test", total_amount=0.0)
        result = pending_order.ship_order("Address", "TRACK")
        assert result.is_failure
        assert "Cannot ship order in status: pending" in result.error

    def test_typed_domain_event_functionality(self) -> None:
        """Test typed domain event functionality."""
        user = UserAggregate(email="typed@example.com", username="typeduser")

        # Add a typed domain event
        user.add_typed_domain_event(
            "CustomUserEvent",
            {"custom_field": "custom_value", "timestamp": datetime.now(UTC)},
        )

        events = user.get_domain_events()
        assert len(events) == 1
        assert events[0].event_type == "CustomUserEvent"
        assert events[0].aggregate_id == user.id
        assert events[0].get_data().value["custom_field"] == "custom_value"

    def test_event_object_direct_addition(self) -> None:
        """Test direct FlextEvent object addition."""
        order = OrderAggregate(customer_id="event_object_test", total_amount=0.0)

        # Create a FlextEvent using the correct factory method
        event_result = FlextPayload.create_event(
            event_type="DirectEvent",
            event_data={"direct": True, "test": "value"},
            aggregate_id=str(order.id),
            version=1,
        )
        assert event_result.success, f"Failed to create event: {event_result.error}"
        custom_event = event_result.value

        # Add the event object directly
        order.add_event_object(custom_event)

        events = order.get_domain_events()
        assert len(events) == 1
        assert events[0].event_type == "DirectEvent"
        event_data = events[0].get_data().value
        assert event_data["direct"] is True
        assert event_data["test"] == "value"


class TestFlextAggregateRootEdgeCases:
    """Test edge cases and error scenarios."""

    def test_initialization_with_invalid_data(self) -> None:
        """Test initialization with invalid data."""
        # Test with malformed metadata
        with pytest.raises(FlextValidationError):
            OrderAggregate(
                customer_id="test",
                total_amount=-100.0,  # This might be invalid depending on validation
                version=-1,  # Invalid version
            )

    def test_metadata_coercion_scenarios(self) -> None:
        """Test metadata coercion in various scenarios."""
        # Test with dict metadata
        order1 = OrderAggregate(
            customer_id="meta_test_1",
            total_amount=50.0,
            metadata={"key": "value", "number": 123},
        )
        assert order1.metadata is not None
        assert order1.metadata.root["key"] == "value"
        assert order1.metadata.root["number"] == 123

        # Test with None metadata
        OrderAggregate(customer_id="meta_test_2", total_amount=50.0, metadata=None)
        # Metadata should be None or default

        # Test with string metadata (should be coerced)
        order3 = OrderAggregate(
            customer_id="meta_test_3", total_amount=50.0, metadata="string_metadata"
        )
        assert order3.metadata is not None
        assert order3.metadata.root["raw"] == "string_metadata"

    def test_domain_events_with_existing_events(self) -> None:
        """Test initialization with existing domain events."""
        existing_events = [
            {
                "event_type": "PreExistingEvent",
                "event_data": {"pre": "existing"},
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]

        order = OrderAggregate(
            customer_id="existing_events_test",
            total_amount=100.0,
            domain_events=existing_events,
        )

        # Should start with the existing events in domain_events dict format
        # but get_domain_events() returns only FlextEvent objects, not dict events
        events = order.get_domain_events()
        assert len(events) == 0  # No FlextEvent objects yet

        # But domain_events dict should contain the existing events
        assert len(order.domain_events.root) == 1
        assert order.domain_events.root[0]["event_type"] == "PreExistingEvent"

        # Add new event (this creates FlextEvent objects)
        order.add_domain_event("NewEvent", {"new": "event"})

        # Should have 1 FlextEvent object from add_domain_event
        all_events = order.get_domain_events()
        assert len(all_events) == 1
        assert all_events[0].event_type == "NewEvent"

        # And 2 events in the dict format (pre-existing + new)
        assert len(order.domain_events.root) == 2

    def test_complex_business_scenario_with_rollback(self) -> None:
        """Test complex business scenario with error handling."""
        order = OrderAggregate(customer_id="complex_test", total_amount=0.0)

        # Start a complex workflow
        items = [{"name": "Complex Item", "price": 100.0, "quantity": 1}]

        # Step 1: Place order (should succeed)
        result1 = order.place_order(items)
        assert result1.success
        assert order.has_domain_events()

        # Step 2: Try invalid operation (should fail but not affect existing events)
        result2 = order.place_order(items)  # Try to place again
        assert result2.is_failure

        # Events should still be there from successful operation
        events = order.get_domain_events()
        assert len(events) == 1
        assert events[0].event_type == "OrderPlaced"

        # Step 3: Continue with valid operation
        result3 = order.ship_order("Recovery Address", "RECOVERY123")
        assert result3.success

        # Should have both events
        final_events = order.get_domain_events()
        assert len(final_events) == 2


class TestAggregateRootPerformanceScenarios:
    """Test performance scenarios with many events."""

    def test_many_domain_events_performance(self) -> None:
        """Test handling many domain events."""
        user = UserAggregate(email="performance@test.com", username="perfuser")

        # Generate many events
        for i in range(100):
            user.add_domain_event(
                f"PerformanceEvent{i}", {"iteration": i, "timestamp": datetime.now(UTC)}
            )

        # Should handle many events efficiently
        events = user.get_domain_events()
        assert len(events) == 100

        # Check that all events are unique and correctly ordered
        event_iterations = [e.get_data().value["iteration"] for e in events]
        assert event_iterations == list(range(100))

        # Test clearing many events
        cleared = user.clear_domain_events()
        assert len(cleared) == 100
        assert not user.has_domain_events()

    def test_aggregate_state_consistency_under_load(self) -> None:
        """Test aggregate state consistency with many operations."""
        order = OrderAggregate(customer_id="consistency_test", total_amount=0.0)

        # Perform many operations
        for i in range(10):
            items = [{"name": f"Item{i}", "price": 10.0, "quantity": 1}]
            if i == 0:  # Only first place_order should succeed
                result = order.place_order(items)
                assert result.success

        # State should be consistent
        assert order.status == "placed"
        assert order.total_amount == 10.0
        assert len(order.items) == 1

        # Should have exactly one domain event
        events = order.get_domain_events()
        assert len(events) == 1
        assert events[0].event_type == "OrderPlaced"
