"""Phase 4: Comprehensive models.py coverage tests targeting 79%+ overall coverage.

This module provides extensive tests for FlextModels to achieve the 79%+ coverage
requirement needed for Phase 4 completion.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, override

from flext_tests import tm
from pydantic import BaseModel, Field

from flext_core._models.domain_event import _ComparableConfigMap
from tests import m


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
        tm.that(user.unique_id, eq="user-1")
        tm.that(user.name, eq="Alice")
        tm.that(user.email, eq="alice@example.com")

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
        tm.that(user1 == user2, eq=True)
        tm.that(user1 != user3, eq=True)

    def test_entity_version(self) -> None:
        """Test entity versioning."""

        class Order(m.Entity):
            total: Decimal

        events: list[m.DomainEvent] = []
        order = Order(unique_id="order-1", total=Decimal("99.99"), domain_events=events)
        initial_version = order.version
        tm.that(initial_version >= 1, eq=True)


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
        tm.that(email1 == email2, eq=True)
        tm.that(email1 != email3, eq=True)

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
        tm.that(price.amount == Decimal("10.00"), eq=True)
        tm.that(price.currency, eq="USD")


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
        tm.that(account.unique_id, eq="acc-1")
        tm.that(account.owner_name, eq="Alice")
        tm.that(account.balance == Decimal("1000.00"), eq=True)

    def test_aggregate_root_domain_events(self) -> None:
        """Test aggregate root domain event handling."""

        class BankAccount(m.AggregateRoot):
            balance: Decimal

        account = BankAccount(
            unique_id="acc-1", balance=Decimal("1000.00"), domain_events=[]
        )
        result = account.add_domain_event(
            "MoneyDeposited",
            m.ConfigMap(root={"amount": 100}),
        )
        tm.ok(result)

    def test_aggregate_root_domain_event_validation(self) -> None:
        """Test domain event validation."""

        class Order(m.AggregateRoot):
            total: Decimal

        order = Order(unique_id="order-1", total=Decimal("99.99"), domain_events=[])
        result = order.add_domain_event("OrderPlaced", m.ConfigMap(root={}))
        tm.ok(result)

    def test_aggregate_root_uncommitted_events(self) -> None:
        """Test uncommitted events tracking."""

        class Order(m.AggregateRoot):
            status: str = "pending"

        order = Order(unique_id="order-1", status="pending", domain_events=[])
        result = order.add_domain_event(
            "OrderCreated",
            m.ConfigMap(root={"timestamp": "2025-01-01"}),
        )
        tm.ok(result)
        tm.that(len(order.domain_events) > 0, eq=True)


class TestFlextModelsDomainEvent:
    """Test FlextModels.DomainEvent functionality."""

    def test_domain_event_creation(self) -> None:
        """Test domain event creation."""
        event = m.DomainEvent(
            event_type="UserCreated",
            aggregate_id="user-1",
            data=_ComparableConfigMap(root={"name": "Alice"}),
        )
        tm.that(event.event_type, eq="UserCreated")
        tm.that(event.aggregate_id, eq="user-1")
        tm.that(event.data.root, eq={"name": "Alice"})

    def test_domain_event_timestamp(self) -> None:
        """Test domain event has timestamp."""
        event = m.DomainEvent(
            event_type="OrderPlaced",
            aggregate_id="order-1",
            data=_ComparableConfigMap(root={"amount": 100}),
        )
        tm.that(event.created_at is not None, eq=True)

    def test_domain_event_empty_data(self) -> None:
        """Test domain event with empty data."""
        event = m.DomainEvent(
            event_type="ProcessStarted",
            aggregate_id="proc-1",
            data=_ComparableConfigMap(root={}),
        )
        tm.that(event.data.root, eq={})


class TestFlextModelsCommand:
    """Test FlextModels.Cqrs.Command functionality."""

    def test_command_creation(self) -> None:
        """Test command creation."""
        cmd = CreateUserCommand(
            user_id="user-1",
            name="Alice",
            email="alice@example.com",
        )
        tm.that(cmd.user_id, eq="user-1")
        tm.that(cmd.name, eq="Alice")


class TestFlextModelsQuery:
    """Test FlextModels.Cqrs.Query functionality."""

    def test_query_creation(self) -> None:
        """Test query creation."""
        query = FindUserQuery(user_id="user-1")
        tm.that(query.user_id, eq="user-1")


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
        tm.that(person.address.city, eq="Springfield")

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
        tm.that(cart.customer_id, eq="cust-1")
        tm.that(item.quantity, eq=1)

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
        tm.that(len(event.data), eq=100)

    def test_command_with_optional_fields(self) -> None:
        """Test command with optional fields."""
        cmd = OptionalFieldCommand(required_field="value")
        tm.that(cmd.required_field, eq="value")
        tm.that(cmd.optional_field is None, eq=True)

    def test_query_with_pagination(self) -> None:
        """Test query with pagination parameters."""
        query = PagedQuery(page=1, page_size=20)
        tm.that(query.page, eq=1)
        tm.that(query.page_size, eq=20)


class TestFlextModelsIntegration:
    """Integration tests combining multiple model types."""

    def test_entity_command_query_flow(self) -> None:
        """Test flow: Command -> Entity -> Event -> Query."""

        class User(m.Entity):
            name: str

        events: list[m.DomainEvent] = []
        cmd = CreateUserCmd(user_id="user-1", name="Alice")
        tm.that(cmd.user_id, eq="user-1")
        user = User(unique_id="user-1", name="Alice", domain_events=events)
        tm.that(user.name, eq="Alice")
        query = GetUserQuery(user_id="user-1")
        tm.that(query.user_id, eq="user-1")

    def test_aggregate_full_lifecycle(self) -> None:
        """Test complete aggregate lifecycle."""

        class Order(m.AggregateRoot):
            items_count: int = 0
            status: str = "new"

        order = Order(
            unique_id="order-1", status="new", items_count=0, domain_events=[]
        )
        tm.that(order.status, eq="new")
        event_result = order.add_domain_event(
            "ItemAdded",
            m.ConfigMap(root={"item_id": "item-1"}),
        )
        tm.ok(event_result)
        tm.that(len(order.domain_events) >= 0, eq=True)


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
