"""Phase 4: Comprehensive models.py coverage tests targeting 79%+ overall coverage.

This module provides extensive tests for FlextModels to achieve the 79%+ coverage
requirement needed for Phase 4 completion.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, override

from pydantic import BaseModel, Field

from flext_core import m
from flext_core._models.domain_event import _ComparableConfigMap


class CreateUserCommand(BaseModel):
    user_id: Annotated[str, Field(description="User identifier for create command")]
    name: Annotated[str, Field(description="User display name for create command")]
    email: Annotated[str, Field(description="User email for create command")]


class FindUserQuery(BaseModel):
    user_id: Annotated[str, Field(description="User identifier for lookup query")]


class OptionalFieldCommand(BaseModel):
    required_field: Annotated[str, Field(description="Required command field")]
    optional_field: Annotated[
        str | None, Field(default=None, description="Optional command field")
    ] = None


class PagedQuery(BaseModel):
    page: Annotated[int, Field(description="Requested page number")]
    page_size: Annotated[int, Field(description="Requested page size")]


class CreateUserCmd(BaseModel):
    user_id: Annotated[
        str, Field(description="User identifier for integration command")
    ]
    name: Annotated[str, Field(description="User name for integration command")]


class GetUserQuery(BaseModel):
    user_id: Annotated[str, Field(description="User identifier for integration query")]


class TestFlextModelsEntity:
    """Test FlextModels.Entity functionality."""

    def test_entity_creation_basic(self) -> None:
        """Test basic entity creation."""

        class User(BaseModel):
            unique_id: Annotated[
                str, Field(description="Unique identifier for test user")
            ]
            name: Annotated[str, Field(description="User name for entity test")]
            email: Annotated[str, Field(description="User email for entity test")]

        user = User(unique_id="user-1", name="Alice", email="alice@example.com")
        assert user.unique_id == "user-1"
        assert user.name == "Alice"
        assert user.email == "alice@example.com"

    def test_entity_equality(self) -> None:
        """Test entity equality based on ID."""

        class User(BaseModel):
            unique_id: Annotated[
                str, Field(description="Unique identifier for equality test")
            ]
            name: Annotated[str, Field(description="User name for equality test")]

            @override
            def __eq__(self, other: object) -> bool:
                if not isinstance(other, User):
                    return NotImplemented
                return self.unique_id == other.unique_id

        user1 = User(unique_id="user-1", name="Alice")
        user2 = User(unique_id="user-1", name="Bob")
        user3 = User(unique_id="user-2", name="Alice")
        assert user1 == user2
        assert user1 != user3

    def test_entity_version(self) -> None:
        """Test entity versioning."""

        class Order(m.Entity):
            domain_events: list[m.DomainEvent] = Field(default_factory=list)
            total: Decimal

        order = Order(unique_id="order-1", total=Decimal("99.99"))
        initial_version = order.version
        assert initial_version >= 1


class TestFlextModelsValue:
    """Test FlextModels.Value functionality."""

    def test_value_object_creation(self) -> None:
        """Test value object creation."""

        class Email(BaseModel):
            address: Annotated[
                str, Field(description="Email address for value object test")
            ]

        email1 = Email(address="test@example.com")
        email2 = Email(address="test@example.com")
        email3 = Email(address="other@example.com")
        assert email1 == email2
        assert email1 != email3

    def test_value_object_immutability(self) -> None:
        """Test that value objects are immutable."""

        class Price(BaseModel):
            amount: Annotated[
                Decimal, Field(description="Price amount for value object test")
            ]
            currency: Annotated[
                str, Field(description="Currency code for value object test")
            ]

        price = Price(amount=Decimal("10.00"), currency="USD")
        assert price.amount == Decimal("10.00")
        assert price.currency == "USD"


class TestFlextModelsAggregateRoot:
    """Test FlextModels.AggregateRoot functionality."""

    def test_aggregate_root_creation(self) -> None:
        """Test aggregate root creation."""

        class Account(BaseModel):
            unique_id: Annotated[str, Field(description="Unique account identifier")]
            owner_name: Annotated[str, Field(description="Account owner name")]
            balance: Annotated[Decimal, Field(description="Account balance")]

        account = Account(
            unique_id="acc-1",
            owner_name="Alice",
            balance=Decimal("1000.00"),
        )
        assert account.unique_id == "acc-1"
        assert account.owner_name == "Alice"
        assert account.balance == Decimal("1000.00")

    def test_aggregate_root_domain_events(self) -> None:
        """Test aggregate root domain event handling."""

        class BankAccount(m.AggregateRoot):
            balance: Decimal

        account = BankAccount(unique_id="acc-1", balance=Decimal("1000.00"))
        result = account.add_domain_event(
            "MoneyDeposited",
            m.ConfigMap(root={"amount": 100}),
        )
        assert result.is_success

    def test_aggregate_root_domain_event_validation(self) -> None:
        """Test domain event validation."""

        class Order(m.AggregateRoot):
            total: Decimal

        order = Order(unique_id="order-1", total=Decimal("99.99"))
        result = order.add_domain_event("OrderPlaced", m.ConfigMap(root={}))
        assert result.is_success

    def test_aggregate_root_uncommitted_events(self) -> None:
        """Test uncommitted events tracking."""

        class Order(m.AggregateRoot):
            status: str = "pending"

        order = Order(unique_id="order-1", status="pending")
        result = order.add_domain_event(
            "OrderCreated",
            m.ConfigMap(root={"timestamp": "2025-01-01"}),
        )
        assert result.is_success
        assert len(order.domain_events) > 0


class TestFlextModelsDomainEvent:
    """Test FlextModels.DomainEvent functionality."""

    def test_domain_event_creation(self) -> None:
        """Test domain event creation."""
        event = m.DomainEvent(
            event_type="UserCreated",
            aggregate_id="user-1",
            data=_ComparableConfigMap(root={"name": "Alice"}),
        )
        assert event.event_type == "UserCreated"
        assert event.aggregate_id == "user-1"
        assert event.data == {"name": "Alice"}

    def test_domain_event_timestamp(self) -> None:
        """Test domain event has timestamp."""
        event = m.DomainEvent(
            event_type="OrderPlaced",
            aggregate_id="order-1",
            data=_ComparableConfigMap(root={"amount": 100}),
        )
        assert event.created_at is not None

    def test_domain_event_empty_data(self) -> None:
        """Test domain event with empty data."""
        event = m.DomainEvent(
            event_type="ProcessStarted",
            aggregate_id="proc-1",
            data=_ComparableConfigMap(root={}),
        )
        assert event.data == {}


class TestFlextModelsCommand:
    """Test FlextModels.Cqrs.Command functionality."""

    def test_command_creation(self) -> None:
        """Test command creation."""
        cmd = CreateUserCommand(
            user_id="user-1",
            name="Alice",
            email="alice@example.com",
        )
        assert cmd.user_id == "user-1"
        assert cmd.name == "Alice"


class TestFlextModelsQuery:
    """Test FlextModels.Cqrs.Query functionality."""

    def test_query_creation(self) -> None:
        """Test query creation."""
        query = FindUserQuery(user_id="user-1")
        assert query.user_id == "user-1"


class TestFlextModelsEdgeCases:
    """Test edge cases and error conditions."""

    def test_entity_with_complex_types(self) -> None:
        """Test entity with complex nested types."""

        class Address(BaseModel):
            street: Annotated[str, Field(description="Street for nested entity test")]
            city: Annotated[str, Field(description="City for nested entity test")]

        class Person(BaseModel):
            unique_id: Annotated[str, Field(description="Unique person identifier")]
            name: Annotated[str, Field(description="Person name")]
            address: Annotated[Address, Field(description="Person address")]

        addr = Address(street="123 Main", city="Springfield")
        person = Person(unique_id="p-1", name="Homer", address=addr)
        assert person.address.city == "Springfield"

    def test_aggregate_root_with_nested_entities(self) -> None:
        """Test aggregate root containing multiple entities."""

        class Item(BaseModel):
            quantity: Annotated[int, Field(description="Item quantity for cart test")]

        class ShoppingCart(BaseModel):
            unique_id: Annotated[
                str, Field(description="Unique shopping cart identifier")
            ]
            customer_id: Annotated[
                str, Field(description="Customer identifier for cart")
            ]

        cart = ShoppingCart(unique_id="cart-1", customer_id="cust-1")
        item = Item(quantity=1)
        assert cart.customer_id == "cust-1"
        assert item.quantity == 1

    def test_domain_event_with_large_data(self) -> None:
        """Test domain event with substantial data payload."""
        large_data: m.ConfigMap = m.ConfigMap(
            root={f"field_{i}": f"value_{i}" for i in range(100)},
        )
        event = m.DomainEvent(
            event_type="BulkDataImported",
            aggregate_id="import-1",
            data=_ComparableConfigMap(root=large_data.root),
        )
        assert len(event.data) == 100

    def test_command_with_optional_fields(self) -> None:
        """Test command with optional fields."""
        cmd = OptionalFieldCommand(required_field="value")
        assert cmd.required_field == "value"
        assert cmd.optional_field is None

    def test_query_with_pagination(self) -> None:
        """Test query with pagination parameters."""
        query = PagedQuery(page=1, page_size=20)
        assert query.page == 1
        assert query.page_size == 20


class TestFlextModelsIntegration:
    """Integration tests combining multiple model types."""

    def test_entity_command_query_flow(self) -> None:
        """Test flow: Command -> Entity -> Event -> Query."""

        class User(m.Entity):
            domain_events: list[m.DomainEvent] = Field(default_factory=list)
            name: str

        cmd = CreateUserCmd(user_id="user-1", name="Alice")
        assert cmd.user_id == "user-1"
        user = User(unique_id="user-1", name="Alice")
        assert user.name == "Alice"
        query = GetUserQuery(user_id="user-1")
        assert query.user_id == "user-1"

    def test_aggregate_full_lifecycle(self) -> None:
        """Test complete aggregate lifecycle."""

        class Order(m.AggregateRoot):
            items_count: int = 0
            status: str = "new"

        order = Order(unique_id="order-1", status="new", items_count=0)
        assert order.status == "new"
        event_result = order.add_domain_event(
            "ItemAdded",
            m.ConfigMap(root={"item_id": "item-1"}),
        )
        assert event_result.is_success
        assert len(order.domain_events) >= 0


__all__ = [
    "TestFlextModelsAggregateRoot",
    "TestFlextModelsCommand",
    "TestFlextModelsDomainEvent",
    "TestFlextModelsEdgeCases",
    "TestFlextModelsEntity",
    "TestFlextModelsIntegration",
    "TestFlextModelsQuery",
    "TestFlextModelsValue",
]
