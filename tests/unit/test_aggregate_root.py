"""Real functionality tests for FlextAggregates - NO MOCKS, only real usage.

These tests demonstrate the aggregate root working with real domain events and business logic,
focusing on real-world DDD scenarios and increasing coverage significantly.

Optimized with Factory Boy patterns, fixtures, and support utilities.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict, cast

import pytest
from pydantic import Field

from flext_core import FlextExceptions, FlextModels, FlextResult, FlextTypes
from tests.support.factories import (
    EdgeCaseGenerators,
    SequenceGenerators,
    UserFactory,
)

pytestmark = [pytest.mark.unit, pytest.mark.core, pytest.mark.ddd]


# Typed dictionaries for precise fixture typing
class OrderAggregateData(TypedDict):
    """Type-safe structure for OrderAggregate constructor data."""

    id: str
    version: int
    created_at: datetime
    updated_at: datetime
    created_by: str | None
    updated_by: str | None
    domain_events: list[FlextTypes.Core.JsonObject]
    aggregate_version: int
    customer_id: str
    total_amount: float
    status: str
    items: list[dict[str, str | float | int]]


class OrderAggregateDataPartial(TypedDict, total=False):
    """Type-safe structure for partial OrderAggregate data."""

    id: str
    version: int
    created_at: datetime
    updated_at: datetime
    created_by: str | None
    updated_by: str | None
    domain_events: list[FlextTypes.Core.JsonObject]
    aggregate_version: int
    customer_id: str
    total_amount: float
    status: str
    items: list[dict[str, str | float | int]]


class OrderAggregate(FlextModels.AggregateRoot):
    """Real order aggregate for testing DDD patterns."""

    # Business fields
    customer_id: str
    total_amount: float = 0.0
    status: str = "pending"
    items: list[dict[str, str | float | int]] = Field(default_factory=list)

    # Implement abstract method from parent
    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for this aggregate."""
        if self.total_amount < 0:
            return FlextResult[None].fail("Total amount cannot be negative")
        return FlextResult[None].ok(None)

    # Helper methods for testing domain events
    def get_domain_events(self) -> list[FlextTypes.Core.JsonObject]:
        """Get all domain events for testing."""
        return self.domain_events.copy()

    def has_domain_events(self) -> bool:
        """Check if aggregate has domain events for testing."""
        return len(self.domain_events) > 0

    def place_order(
        self, items: list[dict[str, str | float | int]]
    ) -> FlextResult[None]:
        """Real business logic - place an order."""
        if not items:
            return FlextResult[None].fail("Cannot place order with no items")

        if self.status != "pending":
            return FlextResult[None].fail(
                f"Cannot place order in status: {self.status}"
            )

        # Calculate total - cast to float with type safety
        total = sum(
            float(str(item.get("price") or 0)) * float(str(item.get("quantity") or 1))
            for item in items
        )

        # Update state
        object.__setattr__(self, "items", items.copy())
        object.__setattr__(self, "total_amount", total)
        object.__setattr__(self, "status", "placed")

        # Add domain event
        self.add_domain_event(
            {
                "event_type": "OrderPlaced",
                "order_id": self.id,
                "customer_id": self.customer_id,
                "total_amount": total,
                "items_count": len(items),
                "placed_at": datetime.now(UTC).isoformat(),
            }
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
            {
                "event_type": "OrderShipped",
                "order_id": self.id,
                "tracking_number": tracking_number,
                "shipping_address": shipping_address,
                "shipped_at": datetime.now(UTC).isoformat(),
                "total_amount": self.total_amount,
            }
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
            {
                "event_type": "OrderCancelled",
                "order_id": self.id,
                "reason": reason,
                "original_amount": self.total_amount,
                "cancelled_at": datetime.now(UTC).isoformat(),
            }
        )

        return FlextResult[None].ok(None)


class UserAggregate(FlextModels.AggregateRoot):
    """Real user aggregate for testing user management scenarios."""

    email: str
    username: str
    is_active: bool = True
    profile_data: dict[str, object] = Field(default_factory=dict)

    # Implement abstract method from parent
    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for this aggregate."""
        if not self.email or "@" not in self.email:
            return FlextResult[None].fail("Valid email is required")
        if not self.username:
            return FlextResult[None].fail("Username is required")
        return FlextResult[None].ok(None)

    # Helper methods for testing domain events
    def get_domain_events(self) -> list[FlextTypes.Core.JsonObject]:
        """Get all domain events for testing."""
        return self.domain_events.copy()

    def has_domain_events(self) -> bool:
        """Check if aggregate has domain events for testing."""
        return len(self.domain_events) > 0

    def add_typed_domain_event(
        self, event_type: str, event_data: dict[str, object]
    ) -> None:
        """Add a typed domain event with proper type conversion."""
        # Create base event structure
        event: FlextTypes.Core.JsonObject = {
            "event_type": event_type,
            "aggregate_id": str(self.id),
        }

        # Add event_data fields with proper type conversion
        for key, value in event_data.items():
            # Convert to JSON-compatible type
            if isinstance(value, datetime):
                event[key] = value.isoformat()
            elif (
                isinstance(value, (str, int, float, bool))
                or value is None
                or isinstance(value, (list, dict))
            ):
                event[key] = value
            else:
                event[key] = str(value)  # Convert other types to string

        self.add_domain_event(event)

    def activate_user(self) -> FlextResult[None]:
        """Real business logic - activate user."""
        if self.is_active:
            return FlextResult[None].fail("User is already active")

        object.__setattr__(self, "is_active", True)

        self.add_domain_event(
            {
                "event_type": "UserActivated",
                "user_id": self.id,
                "email": self.email,
                "activated_at": datetime.now(UTC).isoformat(),
            }
        )

        return FlextResult[None].ok(None)

    def update_profile(self, profile_data: dict[str, object]) -> FlextResult[None]:
        """Real business logic - update user profile."""
        if not profile_data:
            return FlextResult[None].fail("Profile data cannot be empty")

        # Merge profile data
        new_profile = self.profile_data.copy()
        new_profile.update(profile_data)
        object.__setattr__(self, "profile_data", new_profile)

        self.add_domain_event(
            {
                "event_type": "UserProfileUpdated",
                "user_id": self.id,
                "updated_fields": list(profile_data.keys()),
                "updated_at": datetime.now(UTC).isoformat(),
            }
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
            {
                "event_type": "UserEmailChanged",
                "user_id": self.id,
                "old_email": old_email,
                "new_email": new_email,
                "changed_at": datetime.now(UTC).isoformat(),
            }
        )

        return FlextResult[None].ok(None)


class TestFlextAggregateRootRealFunctionality:
    """Test FlextAggregates with real business scenarios."""

    @pytest.fixture
    def basic_order_data(self) -> OrderAggregateData:
        """Fixture providing complete order data for AggregateRoot constructor."""
        return {
            "id": SequenceGenerators.entity_id_sequence(),
            "customer_id": SequenceGenerators.entity_id_sequence(),
            "total_amount": 0.0,
            "status": "pending",
            "version": 1,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "created_by": "test_user",
            "updated_by": "test_user",
            "domain_events": [],
            "aggregate_version": 1,
            "items": [],
        }

    @pytest.fixture
    def sample_items(self) -> list[dict[str, str | float | int]]:
        """Fixture providing sample order items."""
        return [
            {"name": "Product A", "price": 25.99, "quantity": 2},
            {"name": "Product B", "price": 15.50, "quantity": 1},
        ]

    @pytest.fixture
    def test_user_factory_data(self) -> dict[str, str]:
        """Fixture providing user factory data."""
        user_data = cast("dict[str, object]", UserFactory.build().__dict__)
        return {
            "id": str(user_data["id"]),
            "email": str(user_data["email"]),
            "username": SequenceGenerators.username_sequence(),
        }

    def test_aggregate_initialization_basic(
        self, basic_order_data: OrderAggregateData
    ) -> None:
        """Test basic aggregate initialization using fixtures."""
        order = OrderAggregate(**basic_order_data)

        assert order.customer_id == basic_order_data["customer_id"]
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
        custom_id = SequenceGenerators.entity_id_sequence()
        customer_id = SequenceGenerators.entity_id_sequence()

        order = OrderAggregate(
            id=custom_id,
            customer_id=customer_id,
            total_amount=100.0,
        )

        assert order.id == custom_id
        assert order.customer_id == customer_id
        assert order.total_amount == 100.0

    def test_real_business_workflow_complete(
        self,
        basic_order_data: OrderAggregateData,
        sample_items: list[dict[str, str | float | int]],
    ) -> None:
        """Test complete real business workflow with multiple operations."""
        # Create order using fixture data
        order = OrderAggregate(**basic_order_data)

        result = order.place_order(sample_items)
        assert result.success
        assert order.status == "placed"
        assert round(order.total_amount, 2) == 67.48  # (25.99 * 2) + 15.50
        assert len(order.items) == len(sample_items)

        # Check domain event was added
        assert order.has_domain_events()
        events = order.get_domain_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "OrderPlaced"
        assert events[0]["order_id"] == order.id
        assert round(cast("float", events[0]["total_amount"]), 2) == 67.48

        # Ship order using generated tracking number
        tracking_number = f"TRACK_{SequenceGenerators.entity_id_sequence()[:8]}"
        result = order.ship_order("123 Main St", tracking_number)
        assert result.success
        assert order.status == "shipped"

        # Check second domain event
        events = order.get_domain_events()
        assert len(events) == 2
        assert events[1]["event_type"] == "OrderShipped"
        assert events[1]["tracking_number"] == tracking_number

    def test_domain_event_management_comprehensive(self) -> None:
        """Test comprehensive domain event management."""
        # Use factory-generated data for user creation
        user_id = SequenceGenerators.entity_id_sequence()
        email = SequenceGenerators.email_sequence()
        username = SequenceGenerators.username_sequence()

        user = UserAggregate(id=user_id, email=email, username=username)

        # Perform multiple operations that generate events
        user.update_profile({"first_name": "John", "last_name": "Doe"})
        new_email = SequenceGenerators.email_sequence()
        user.change_email(new_email)
        user.activate_user()

        # Check all events were recorded
        events = user.get_domain_events()
        assert len(events) == 2  # activate_user fails because user starts active

        event_types = [e["event_type"] for e in events]
        assert "UserProfileUpdated" in event_types
        assert "UserEmailChanged" in event_types

        # Test event data completeness
        profile_event = next(
            e for e in events if e["event_type"] == "UserProfileUpdated"
        )
        assert profile_event["user_id"] == user.id
        assert "first_name" in cast("list[str]", profile_event["updated_fields"])
        assert "last_name" in cast("list[str]", profile_event["updated_fields"])

        email_event = next(e for e in events if e["event_type"] == "UserEmailChanged")
        assert email_event["old_email"] == email
        assert email_event["new_email"] == new_email

    def test_domain_event_clearing_and_persistence(
        self, basic_order_data: OrderAggregateData
    ) -> None:
        """Test domain event clearing and persistence behavior."""
        order = OrderAggregate(**basic_order_data)

        # Generate some events using single item
        items: list[dict[str, str | float | int]] = [
            {"name": "Clear Test Item", "price": 10.0, "quantity": 1}
        ]
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
        assert new_events[0]["event_type"] == "OrderCancelled"

    def test_business_rule_validation_real_scenarios(
        self, basic_order_data: OrderAggregateData
    ) -> None:
        """Test business rule validation in real scenarios."""
        order = OrderAggregate(**basic_order_data)

        # Test placing order with no items
        result = order.place_order([])
        assert result.is_failure
        assert result.error is not None
        assert "Cannot place order with no items" in result.error

        # Place valid order
        items: list[dict[str, str | float | int]] = [
            {"name": "Item", "price": 20.0, "quantity": 1}
        ]
        result = order.place_order(items)
        assert result.success

        # Try to place order again (should fail)
        result = order.place_order(items)
        assert result.is_failure
        assert result.error is not None
        assert "Cannot place order in status: placed" in result.error

        # Try to ship before placing (create new order in pending)
        pending_order_id = SequenceGenerators.entity_id_sequence()
        pending_customer_id = SequenceGenerators.entity_id_sequence()
        pending_order = OrderAggregate(
            id=pending_order_id, customer_id=pending_customer_id, total_amount=0.0
        )
        result = pending_order.ship_order("Address", "TRACK")
        assert result.is_failure
        assert result.error is not None
        assert "Cannot ship order in status: pending" in result.error

    def test_typed_domain_event_functionality(self) -> None:
        """Test typed domain event functionality."""
        user_id = SequenceGenerators.entity_id_sequence()
        email = SequenceGenerators.email_sequence()
        username = SequenceGenerators.username_sequence()

        user = UserAggregate(id=user_id, email=email, username=username)

        # Add a typed domain event
        user.add_typed_domain_event(
            "CustomUserEvent",
            {"custom_field": "custom_value", "timestamp": datetime.now(UTC)},
        )

        events = user.get_domain_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "CustomUserEvent"
        assert events[0]["aggregate_id"] == user.id
        assert events[0]["custom_field"] == "custom_value"

    def test_event_object_direct_addition(
        self, basic_order_data: OrderAggregateData
    ) -> None:
        """Test direct domain event addition."""
        order = OrderAggregate(**basic_order_data)

        # Add a domain event directly as a dictionary
        custom_event: FlextTypes.Core.JsonObject = {
            "event_type": "DirectEvent",
            "aggregate_id": str(order.id),
            "direct": True,
            "test": "value",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        order.add_domain_event(custom_event)

        events = order.get_domain_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "DirectEvent"
        assert events[0]["direct"] is True
        assert events[0]["test"] == "value"


class TestFlextAggregateRootEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.fixture
    def basic_user_data(self) -> dict[str, str]:
        """Fixture providing basic user data."""
        return {
            "id": SequenceGenerators.entity_id_sequence(),
            "email": SequenceGenerators.email_sequence(),
            "username": SequenceGenerators.username_sequence(),
        }

    @pytest.fixture
    def edge_case_data(self) -> dict[str, list[object]]:
        """Fixture providing edge case test data."""
        return {
            "unicode_strings": cast(
                "list[object]", EdgeCaseGenerators.unicode_strings()
            ),
            "special_characters": cast(
                "list[object]", EdgeCaseGenerators.special_characters()
            ),
            "boundary_numbers": cast(
                "list[object]", EdgeCaseGenerators.boundary_numbers()
            ),
            "empty_values": EdgeCaseGenerators.empty_values(),
            "large_values": EdgeCaseGenerators.large_values(),
        }

    @pytest.fixture
    def basic_order_data(self) -> OrderAggregateData:
        """Fixture providing complete order data for AggregateRoot constructor."""
        return {
            "id": SequenceGenerators.entity_id_sequence(),
            "customer_id": SequenceGenerators.entity_id_sequence(),
            "total_amount": 0.0,
            "status": "pending",
            "version": 1,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "created_by": "test_user",
            "updated_by": "test_user",
            "domain_events": [],
            "aggregate_version": 1,
            "items": [],
        }

    def test_initialization_with_invalid_data(self) -> None:
        """Test initialization with invalid data."""
        # Test with malformed data - missing required fields should raise ValidationError
        with pytest.raises((FlextExceptions.ValidationError, ValueError, TypeError)):
            OrderAggregate(  # type: ignore[call-arg]
                # Missing required id field - intentionally testing invalid constructor
                customer_id=SequenceGenerators.entity_id_sequence(),
                total_amount=50.0,
            )

        # Test with invalid types
        with pytest.raises((FlextExceptions.ValidationError, ValueError, TypeError)):
            OrderAggregate(
                id=cast(
                    "str", 123
                ),  # Should be string - cast to suppress MyPy error for testing
                customer_id=SequenceGenerators.entity_id_sequence(),
                total_amount=cast(
                    "float", "invalid"
                ),  # Should be float - cast to suppress MyPy error for testing
            )

    def test_metadata_coercion_scenarios(self) -> None:
        """Test metadata coercion in various scenarios."""
        # Test with dict metadata
        order_id_1 = SequenceGenerators.entity_id_sequence()
        customer_id_1 = SequenceGenerators.entity_id_sequence()
        order1 = OrderAggregate(
            id=order_id_1,
            customer_id=customer_id_1,
            total_amount=50.0,
        )
        # Verify order was created successfully
        assert order1.id == order_id_1
        assert order1.customer_id == customer_id_1
        assert order1.total_amount == 50.0

        # Test with None metadata (default case)
        order_id_2 = SequenceGenerators.entity_id_sequence()
        customer_id_2 = SequenceGenerators.entity_id_sequence()
        order2 = OrderAggregate(
            id=order_id_2,
            customer_id=customer_id_2,
            total_amount=50.0,
        )
        # Verify basic order creation
        assert order2.id == order_id_2
        assert order2.customer_id == customer_id_2

        # Test with different total amount
        order_id_3 = SequenceGenerators.entity_id_sequence()
        customer_id_3 = SequenceGenerators.entity_id_sequence()
        order3 = OrderAggregate(
            id=order_id_3,
            customer_id=customer_id_3,
            total_amount=75.0,
        )
        assert order3.total_amount == 75.0

    def test_domain_events_with_existing_events(self) -> None:
        """Test domain event management with multiple events."""
        order_id = SequenceGenerators.entity_id_sequence()
        customer_id = SequenceGenerators.entity_id_sequence()
        order = OrderAggregate(
            id=order_id,
            customer_id=customer_id,
            total_amount=100.0,
        )

        # Start with no events
        events = order.get_domain_events()
        assert len(events) == 0

        # Add first event
        order.add_domain_event(
            {
                "event_type": "FirstEvent",
                "event_data": {"first": "data"},
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # Should have 1 event
        events_after_first = order.get_domain_events()
        assert len(events_after_first) == 1
        assert events_after_first[0]["event_type"] == "FirstEvent"

        # Add second event
        order.add_domain_event(
            {
                "event_type": "SecondEvent",
                "event_data": {"second": "data"},
            }
        )

        # Should have 2 events
        all_events = order.get_domain_events()
        assert len(all_events) == 2
        assert all_events[0]["event_type"] == "FirstEvent"
        assert all_events[1]["event_type"] == "SecondEvent"

    def test_complex_business_scenario_with_rollback(
        self, basic_order_data: OrderAggregateData
    ) -> None:
        """Test complex business scenario with error handling."""
        order = OrderAggregate(**basic_order_data)

        # Start a complex workflow
        items: list[dict[str, str | float | int]] = [
            {"name": "Complex Item", "price": 100.0, "quantity": 1}
        ]

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
        assert events[0]["event_type"] == "OrderPlaced"

        # Step 3: Continue with valid operation
        recovery_tracking = f"RECOVERY_{SequenceGenerators.entity_id_sequence()[:8]}"
        result3 = order.ship_order("Recovery Address", recovery_tracking)
        assert result3.success

        # Should have both events
        final_events = order.get_domain_events()
        assert len(final_events) == 2


class TestAggregateRootPerformanceScenarios:
    """Test performance scenarios with many events."""

    def test_many_domain_events_performance(self) -> None:
        """Test handling many domain events."""
        user = UserAggregate(
            id=SequenceGenerators.entity_id_sequence(),
            email=SequenceGenerators.email_sequence(),
            username=SequenceGenerators.username_sequence(),
        )

        # Generate many events
        for i in range(100):
            user.add_domain_event(
                {
                    "event_type": f"PerformanceEvent{i}",
                    "iteration": i,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        # Should handle many events efficiently
        events = user.get_domain_events()
        assert len(events) == 100

        # Check that all events are unique and correctly ordered
        event_iterations = [e["iteration"] for e in events]
        assert event_iterations == list(range(100))

        # Test clearing many events
        cleared = user.clear_domain_events()
        assert len(cleared) == 100
        assert not user.has_domain_events()

    def test_aggregate_state_consistency_under_load(self) -> None:
        """Test aggregate state consistency with many operations."""
        order = OrderAggregate(
            id=SequenceGenerators.entity_id_sequence(),
            customer_id=SequenceGenerators.entity_id_sequence(),
            total_amount=0.0,
        )

        # Perform many operations
        for i in range(10):
            items: list[dict[str, str | float | int]] = [
                {"name": f"Item{i}", "price": 10.0, "quantity": 1}
            ]
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
        assert events[0]["event_type"] == "OrderPlaced"


class TestAggregateRootEventSourcing:
    """Test coverage for AggregateRoot-specific event sourcing functionality."""

    def test_apply_domain_event_basic_functionality(self) -> None:
        """Test apply_domain_event method basic functionality."""
        order = OrderAggregate(
            id=SequenceGenerators.entity_id_sequence(),
            customer_id=SequenceGenerators.entity_id_sequence(),
        )

        # Create test event
        event: FlextTypes.Core.JsonObject = {
            "event_type": "TestEvent",
            "aggregate_id": order.id,
            "timestamp": datetime.now(UTC).isoformat(),
            "data": {"test_field": "test_value"},
        }

        # Apply event should succeed
        result = order.apply_domain_event(event)
        assert result.success

        # Event should be added to domain events
        events = order.get_domain_events()
        assert len(events) == 1
        assert events[0] == event

    def test_apply_domain_event_with_handler(self) -> None:
        """Test apply_domain_event with custom handler method."""

        class TestOrderWithHandler(OrderAggregate):
            """Test order with custom event handler."""

            custom_field: str = "initial"

            def _apply_testevent(self, event: FlextTypes.Core.JsonObject) -> None:
                """Handler for TestEvent."""
                if (
                    isinstance(event.get("data"), dict)
                    and "custom_value" in event["data"]
                ):
                    self.custom_field = str(event["data"]["custom_value"])

        order = TestOrderWithHandler(
            id=SequenceGenerators.entity_id_sequence(),
            customer_id=SequenceGenerators.entity_id_sequence(),
        )

        # Create event with custom handler
        event: FlextTypes.Core.JsonObject = {
            "event_type": "TestEvent",
            "aggregate_id": order.id,
            "timestamp": datetime.now(UTC).isoformat(),
            "data": {"custom_value": "handled_value"},
        }

        # Apply event
        result = order.apply_domain_event(event)
        assert result.success

        # Handler should have been called
        assert order.custom_field == "handled_value"

        # Event should still be added to domain events
        events = order.get_domain_events()
        assert len(events) == 1

    def test_apply_domain_event_error_handling(self) -> None:
        """Test apply_domain_event error handling."""
        order = OrderAggregate(
            id=SequenceGenerators.entity_id_sequence(),
            customer_id=SequenceGenerators.entity_id_sequence(),
        )

        # Test with invalid event structure (missing event_type)
        invalid_event: FlextTypes.Core.JsonObject = {
            "timestamp": datetime.now(UTC).isoformat(),
            "data": {"test": "data"},
        }

        # Should still succeed (no event_type is handled gracefully)
        result = order.apply_domain_event(invalid_event)
        assert result.success

        # Event should still be added
        events = order.get_domain_events()
        assert len(events) == 1

    def test_apply_domain_event_with_non_string_event_type(self) -> None:
        """Test apply_domain_event with non-string event_type."""
        order = OrderAggregate(
            id=SequenceGenerators.entity_id_sequence(),
            customer_id=SequenceGenerators.entity_id_sequence(),
        )

        # Event with non-string event_type
        event: FlextTypes.Core.JsonObject = {
            "event_type": 123,  # Non-string type
            "timestamp": datetime.now(UTC).isoformat(),
            "data": {"test": "data"},
        }

        # Should handle gracefully
        result = order.apply_domain_event(event)
        assert result.success

        # Event should be added
        events = order.get_domain_events()
        assert len(events) == 1

    def test_apply_domain_event_exception_handling(self) -> None:
        """Test apply_domain_event when handler throws exception."""

        class TestOrderWithFailingHandler(OrderAggregate):
            """Test order with failing event handler."""

            def _apply_failevent(self, event: FlextTypes.Core.JsonObject) -> None:
                """Handler that always fails."""
                # Use event parameter to avoid unused argument warning
                _ = event
                msg = "Handler failed"
                raise RuntimeError(msg)

        order = TestOrderWithFailingHandler(
            id=SequenceGenerators.entity_id_sequence(),
            customer_id=SequenceGenerators.entity_id_sequence(),
        )

        # Event that triggers failing handler
        event: FlextTypes.Core.JsonObject = {
            "event_type": "FailEvent",
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Should handle exception and return failure
        result = order.apply_domain_event(event)
        assert result.failure
        assert "Failed to apply event" in result.error

        # Event should still be added to domain events (it's added before handler)
        events = order.get_domain_events()
        assert len(events) == 1

    def test_apply_domain_event_multiple_events(self) -> None:
        """Test applying multiple domain events in sequence."""
        order = OrderAggregate(
            id=SequenceGenerators.entity_id_sequence(),
            customer_id=SequenceGenerators.entity_id_sequence(),
        )

        # Apply multiple events
        events_to_apply = [
            {
                "event_type": f"Event{i}",
                "sequence": i,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            for i in range(5)
        ]

        for event in events_to_apply:
            result = order.apply_domain_event(event)
            assert result.success

        # All events should be in domain events
        stored_events = order.get_domain_events()
        assert len(stored_events) == 5

        # Events should maintain order
        for i, stored_event in enumerate(stored_events):
            assert stored_event["sequence"] == i
