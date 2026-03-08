"""Comprehensive coverage tests for FlextModels DDD patterns.

Module: flext_core.models
Scope: FlextModels - Value Objects, Entities, Aggregate Roots, Commands, Queries, Domain Events, Metadata

This module provides extensive tests for Domain-Driven Design patterns:
- Value Objects (immutable, value-based equality)
- Entities (identity-based, lifecycle tracking)
- Aggregate Roots (consistency boundaries)
- Commands/Queries (CQRS patterns)
- Domain Events (event sourcing)
- Metadata (flexible attribute tracking)

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest
from pydantic import ValidationError

from flext_core import m
from flext_core._models.domain_event import _ComparableConfigMap


class ModelScenarios:
    """Centralized model test scenarios using FlextConstants."""


class TestValues:
    """Test immutable value objects using FlextTestsUtilities."""

    def test_value_object_creation(self) -> None:
        """Test creating a value object."""
        money = Money(amount=100.0, currency="USD")
        assert money.amount == pytest.approx(100.0)
        assert money.currency == "USD"

    def test_value_object_immutability(self) -> None:
        """Test value object is immutable."""
        point = Point(x=1.0, y=2.0)
        with pytest.raises(ValidationError):
            setattr(point, "x", 3.0)

    def test_value_object_equality_by_value(self) -> None:
        """Test value objects compared by value."""
        color1 = Color(red=255, green=0, blue=0)
        color2 = Color(red=255, green=0, blue=0)
        assert color1 == color2

    def test_value_object_validation(self) -> None:
        """Test value object validation."""
        email = Email(address="USER@EXAMPLE.COM")
        assert email.address == "user@example.com"
        with pytest.raises(ValidationError):
            Email(address="notanemail")

    def test_value_object_hashable(self) -> None:
        """Test value objects are hashable."""
        isbn1 = ISBN(code="978-0-262-03384-8")
        isbn2 = ISBN(code="978-0-262-03384-8")
        isbn_set: set[ISBN] = {isbn1, isbn2}
        assert len(isbn_set) == 1


class TestEntities:
    """Test domain entities with identity using FlextTestsUtilities."""

    def test_entity_creation(self) -> None:
        """Test creating an entity."""
        person = Person(name="Alice", age=30)
        assert person.name == "Alice"
        assert person.unique_id is not None
        assert person.created_at is not None
        assert person.updated_at is not None

    def test_entity_identity_tracking(self) -> None:
        """Test entities are compared by identity."""
        account1 = Account(name="Checking", balance=100.0)
        account2 = Account(name="Checking", balance=100.0)
        assert account1.unique_id != account2.unique_id
        assert account1 != account2

    def test_entity_lifecycle_tracking(self) -> None:
        """Test entity creation and update timestamps."""
        doc = Document(title="Test Doc")
        assert doc.created_at is not None
        assert doc.updated_at is not None
        assert doc.created_at <= doc.updated_at

    def test_entity_validation(self) -> None:
        """Test entity field validation."""
        user = User(email="user@example.com", username="alice")
        assert user.username == "alice"
        with pytest.raises(ValidationError):
            User(email="user@example.com", username="ab")

    def test_entity_model_dump_serialization(self) -> None:
        """Test entity serialization using Pydantic model_dump."""
        product = Product(name="Widget", price=19.99)
        product_dict = product.model_dump()
        assert isinstance(product_dict, dict)
        assert product_dict["name"] == "Widget"
        assert product_dict["price"] == pytest.approx(19.99)
        assert all(
            key in product_dict for key in ["unique_id", "created_at", "updated_at"]
        )


class TestAggregateRoots:
    """Test aggregate roots for consistency boundaries using FlextTestsUtilities."""

    def test_aggregate_root_creation(self) -> None:
        """Test creating an aggregate root."""
        order = Order(order_number="ORD-001", status="pending")
        assert order.order_number == "ORD-001"
        assert order.unique_id is not None

    def test_aggregate_root_invariants(self) -> None:
        """Test aggregate root enforces invariants."""
        account = Account(balance=1000.0, currency="USD")
        assert account.balance >= 0.0

    def test_aggregate_root_lifecycle(self) -> None:
        """Test aggregate root lifecycle."""
        project = Project(name="New Project", status="planning")
        assert project.status == "planning"
        assert project.created_at is not None


class TestCommands:
    """Test CQRS command pattern using FlextTestsUtilities."""

    def test_command_creation(self) -> None:
        """Test creating a command."""
        cmd = CreateUserCommand(email="user@example.com", username="alice")
        assert cmd.email == "user@example.com"
        assert cmd.command_id is not None

    def test_command_mutation_behavior(self) -> None:
        """Test command mutation behavior with validate_assignment."""
        cmd = UpdateProfileCommand(name="Alice", bio="Developer")
        original_name = cmd.name
        cmd.name = "Bob"
        assert cmd.name == "Bob"
        assert cmd.name != original_name

    def test_command_validation(self) -> None:
        """Test command validation."""
        cmd = DepositCommand(account_id="ACC-001", amount=100.0)
        assert cmd.amount == pytest.approx(100.0)
        with pytest.raises(ValidationError):
            DepositCommand(account_id="ACC-001", amount=-50.0)


# Define Query classes at module level using direct import


class TestQueries:
    """Test CQRS query pattern using FlextTestsUtilities."""

    def test_query_creation(self) -> None:
        """Test creating a query."""
        query = GetUserQuery(
            filters=m.Dict(root={"user_id": "USER-001"}),
            query_type="get_user",
        )
        assert query.filters["user_id"] == "USER-001"
        assert query.query_id is not None
        assert query.query_type == "get_user"

    def test_query_mutation_behavior(self) -> None:
        """Test query mutation behavior with validate_assignment."""
        query = ListAccountsQuery(page=1, limit=10)
        original_page = query.page
        query.page = 2
        assert query.page == 2
        assert query.page != original_page

    def test_query_with_filters(self) -> None:
        """Test query with filtering."""
        query = SearchProductsQuery(
            keyword="laptop",
            category="electronics",
            min_price=500.0,
        )
        assert query.keyword == "laptop"
        assert query.category == "electronics"
        assert query.min_price == pytest.approx(500.0)


class TestDomainEvents:
    """Test domain events for event sourcing using FlextTestsUtilities."""

    def test_domain_event_creation(self) -> None:
        """Test creating a domain event with data payload."""
        event = m.DomainEvent(
            event_type="UserCreated",
            aggregate_id="USER-001",
            data=_ComparableConfigMap(
                root={"user_id": "USER-001", "email": "user@example.com"},
            ),
        )
        assert event.event_type == "UserCreated"
        assert event.aggregate_id == "USER-001"
        assert event.data["user_id"] == "USER-001"
        assert event.data["email"] == "user@example.com"
        assert event.unique_id is not None
        assert event.created_at is not None

    def test_domain_event_equality(self) -> None:
        """Test domain events can be compared and tracked."""
        event1 = m.DomainEvent(
            event_type="OrderShipped",
            aggregate_id="ORD-001",
            data=_ComparableConfigMap(
                root={"tracking_number": "TRACK-123"},
            ),
        )
        event2 = m.DomainEvent(
            event_type="OrderShipped",
            aggregate_id="ORD-001",
            data=_ComparableConfigMap(
                root={"tracking_number": "TRACK-123"},
            ),
        )
        assert event1.unique_id != event2.unique_id
        assert event1.event_type == event2.event_type
        assert event1.aggregate_id == event2.aggregate_id

    def test_domain_event_timestamp(self) -> None:
        """Test domain events have timestamps."""
        event = AccountUpdatedEvent(
            event_type="AccountUpdated",
            aggregate_id="ACC-001",
            data=_ComparableConfigMap(root={"field": "balance"}),
        )
        assert event.created_at is not None
        assert isinstance(event.created_at, datetime)

    def test_domain_event_causality(self) -> None:
        """Test domain events track causality via id and timestamps."""
        event = PaymentProcessedEvent(
            event_type="PaymentProcessed",
            aggregate_id="PAY-001",
            data=_ComparableConfigMap(root={"amount": 99.99}),
        )
        assert event.unique_id is not None
        assert event.created_at is not None


class TestMetadata:
    """Test flexible metadata model using FlextTestsUtilities."""

    def test_metadata_creation(self) -> None:
        """Test creating metadata."""
        metadata = m.Metadata(
            attributes={"user_id": "123", "operation": "create"},
        )
        assert metadata.attributes["user_id"] == "123"

    def test_metadata_with_various_types(self) -> None:
        """Test metadata with different attribute types."""
        metadata = m.Metadata(
            attributes={
                "string": "value",
                "number": 42,
                "float": math.pi,
                "bool": True,
            },
        )
        assert metadata.attributes["string"] == "value"
        assert metadata.attributes["number"] == 42


class TestModelValidation:
    """Test model validation patterns using FlextTestsUtilities."""

    def test_model_validation_error_handling(self) -> None:
        """Test model validation error handling."""
        entity = ValidatedEntity(email="user@example.com", age=30)
        assert entity.age == 30
        with pytest.raises(ValidationError):
            ValidatedEntity(email="user@example.com", age=200)

    def test_multiple_field_validation(self) -> None:
        """Test multiple field validators."""
        profile = Profile(username="alice", email="alice@example.com", bio="Developer")
        assert profile.username == "alice"


class TestModelSerialization:
    """Test model serialization patterns using FlextTestsUtilities."""

    def test_entity_model_dump(self) -> None:
        """Test model_dump serialization."""
        task = Task(title="Complete tests", completed=False)
        dumped = task.model_dump()
        assert dumped["title"] == "Complete tests"
        assert dumped["completed"] is False
        assert "unique_id" in dumped

    def test_command_serialization(self) -> None:
        """Test command serialization."""
        cmd = SendEmailCommand(
            recipient="user@example.com",
            subject="Test",
            body="Message body",
        )
        dumped = cmd.model_dump()
        assert dumped["recipient"] == "user@example.com"
        assert dumped["subject"] == "Test"

    def test_aggregate_root_serialization(self) -> None:
        """Test aggregate root serialization."""
        cart = ShoppingCart(
            items=[
                {"product_id": "P1", "quantity": 2},
                {"product_id": "P2", "quantity": 1},
            ],
            total=99.99,
        )
        dumped = cart.model_dump()
        assert len(dumped["items"]) == 2
        assert dumped["total"] == pytest.approx(99.99)


class TestModelIntegration:
    """Test model integration with FlextResult using FlextTestsUtilities."""

    def test_entity_model_validation(self) -> None:
        """Test entity model validation via model_validate."""
        customer = Customer(name="John", email="john@example.com")
        # Create dict with only actual model fields (exclude computed fields)
        # In Pydantic v2, iterate model_fields and only include existing keys
        dumped = customer.model_dump()
        customer_dict = {
            k: dumped[k] for k in type(customer).model_fields if k in dumped
        }
        validated = Customer.model_validate(customer_dict)
        assert validated is not None
        assert validated.name == "John"
        assert validated.email == "john@example.com"

    def test_command_factory_pattern(self) -> None:
        """Test command creation as factories."""
        cmd = RegisterUserCommand(email="user@example.com", password="secure123")
        assert cmd.email == "user@example.com"


__all__ = [
    "TestAggregateRoots",
    "TestCommands",
    "TestDomainEvents",
    "TestEntities",
    "TestMetadata",
    "TestModelIntegration",
    "TestModelSerialization",
    "TestModelValidation",
    "TestQueries",
    "TestValues",
]
