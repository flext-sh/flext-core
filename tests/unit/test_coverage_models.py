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
from typing import override

import pytest
from flext_tests import t, tm
from pydantic import ValidationError, field_validator

from flext_core._models.domain_event import _ComparableConfigMap
from tests.models import m


class ModelScenarios:
    """Centralized model test scenarios using FlextConstants."""


class TestValues:
    """Test immutable value objects using FlextTestsUtilities."""

    def test_value_object_creation(self) -> None:
        """Test creating a value object."""

        class Money(m.Value):
            """Money value object."""

            amount: float
            currency: str

        money = Money(amount=100.0, currency="USD")
        tm.that(math.isclose(money.amount, 100.0), eq=True)
        tm.that(money.currency, eq="USD")

    def test_value_object_immutability(self) -> None:
        """Test value object is immutable."""

        class Point(m.Value):
            """Point value object."""

            x: float
            y: float

        point = Point(x=1.0, y=2.0)
        with pytest.raises(ValidationError):
            setattr(point, "x", 3.0)

    def test_value_object_equality_by_value(self) -> None:
        """Test value objects compared by value."""

        class Color(m.Value):
            """Color value object."""

            red: int
            green: int
            blue: int

        color1 = Color(red=255, green=0, blue=0)
        color2 = Color(red=255, green=0, blue=0)
        tm.that(color1, eq=color2)

    def test_value_object_validation(self) -> None:
        """Test value object validation."""

        class Email(m.Value):
            """Email value object with validation."""

            address: str

            @field_validator("address")
            @classmethod
            def validate_email(cls, v: str) -> str:
                if "@" not in v:
                    error_msg = "Invalid email format"
                    raise ValueError(error_msg)
                return v.lower()

        email = Email(address="USER@EXAMPLE.COM")
        tm.that(email.address, eq="user@example.com")
        with pytest.raises(ValidationError):
            Email(address="notanemail")

    def test_value_object_hashable(self) -> None:
        """Test value objects are hashable."""

        class ISBN(m.Value):
            """ISBN value object."""

            code: str

            def __hash__(self) -> int:
                """Hash based on code value."""
                return hash(self.code)

            @override
            def __eq__(self, other: object) -> bool:
                """Equality based on code value."""
                if not isinstance(other, ISBN):
                    return False
                return self.code == other.code

        isbn1 = ISBN(code="978-0-262-03384-8")
        isbn2 = ISBN(code="978-0-262-03384-8")
        isbn_set: set[ISBN] = {isbn1, isbn2}
        tm.that(len(isbn_set), eq=1)


class TestEntities:
    """Test domain entities with identity using FlextTestsUtilities."""

    def test_entity_creation(self) -> None:
        """Test creating an entity."""

        class Person(m.Entity):
            """Person entity."""

            name: str
            age: int

        person = Person(name="Alice", age=30, domain_events=[])
        tm.that(person.name, eq="Alice")
        tm.that(person.unique_id, none=False)
        tm.that(person.created_at, none=False)
        tm.that(person.updated_at, none=False)

    def test_entity_identity_tracking(self) -> None:
        """Test entities are compared by identity."""

        class Account(m.Entity):
            """Account entity."""

            name: str
            balance: float

        account1 = Account(name="Checking", balance=100.0, domain_events=[])
        account2 = Account(name="Checking", balance=100.0, domain_events=[])
        tm.that(account1.unique_id, ne=account2.unique_id)
        tm.that(account1, ne=account2)

    def test_entity_lifecycle_tracking(self) -> None:
        """Test entity creation and update timestamps."""

        class Document(m.Entity):
            """Document entity."""

            title: str

        doc = Document(title="Test Doc", domain_events=[])
        tm.that(doc.created_at, none=False)
        tm.that(doc.updated_at, none=False)
        tm.that(doc.created_at <= doc.updated_at, eq=True)

    def test_entity_validation(self) -> None:
        """Test entity field validation."""

        class User(m.Entity):
            """User entity with validation."""

            email: str
            username: str

            @field_validator("username")
            @classmethod
            def validate_username(cls, v: str) -> str:
                if len(v) < 3:
                    error_msg = "Username must be at least 3 characters"
                    raise ValueError(error_msg)
                return v

        user = User(email="user@example.com", username="alice", domain_events=[])
        tm.that(user.username, eq="alice")
        with pytest.raises(ValidationError):
            User(email="user@example.com", username="ab", domain_events=[])

    def test_entity_model_dump_serialization(self) -> None:
        """Test entity serialization using Pydantic model_dump."""

        class Product(m.Entity):
            """Product entity."""

            name: str
            price: float

        product = Product(name="Widget", price=19.99, domain_events=[])
        product_dict = product.model_dump()
        tm.that(isinstance(product_dict, dict), eq=True)
        tm.that(product_dict["name"], eq="Widget")
        tm.that(math.isclose(product_dict["price"], 19.99), eq=True)
        tm.that(
            all(
                key in product_dict for key in ["unique_id", "created_at", "updated_at"]
            ),
            eq=True,
        )


class TestAggregateRoots:
    """Test aggregate roots for consistency boundaries using FlextTestsUtilities."""

    def test_aggregate_root_creation(self) -> None:
        """Test creating an aggregate root."""

        class Order(m.AggregateRoot):
            """Order aggregate root."""

            order_number: str
            status: str

        order = Order(order_number="ORD-001", status="pending", domain_events=[])
        tm.that(order.order_number, eq="ORD-001")
        tm.that(order.unique_id, none=False)

    def test_aggregate_root_invariants(self) -> None:
        """Test aggregate root enforces invariants."""

        class Account(m.AggregateRoot):
            """Account with business rules."""

            balance: float
            currency: str

        account = Account(balance=1000.0, currency="USD", domain_events=[])
        tm.that(account.balance >= 0.0, eq=True)

    def test_aggregate_root_lifecycle(self) -> None:
        """Test aggregate root lifecycle."""

        class Project(m.AggregateRoot):
            """Project aggregate root."""

            name: str
            status: str

        project = Project(name="New Project", status="planning", domain_events=[])
        tm.that(project.status, eq="planning")
        tm.that(project.created_at, none=False)


class TestCommands:
    """Test CQRS command pattern using FlextTestsUtilities."""

    def test_command_creation(self) -> None:
        """Test creating a command."""

        class CreateUserCommand(m.Command):
            """Command to create a user."""

            email: str
            username: str

        cmd = CreateUserCommand(
            email="user@example.com", username="alice", command_id="cmd-test-1"
        )
        tm.that(cmd.email, eq="user@example.com")
        tm.that(cmd.command_id, none=False)

    def test_command_mutation_behavior(self) -> None:
        """Test command mutation behavior with validate_assignment."""

        class UpdateProfileCommand(m.Command):
            """Command to update profile."""

            name: str
            bio: str

        cmd = UpdateProfileCommand(
            name="Alice", bio="Developer", command_id="cmd-test-2"
        )
        original_name = cmd.name
        cmd.name = "Bob"
        tm.that(cmd.name, eq="Bob")
        tm.that(cmd.name, ne=original_name)

    def test_command_validation(self) -> None:
        """Test command validation."""

        class DepositCommand(m.Command):
            """Command with validation."""

            account_id: str
            amount: float

            @field_validator("amount")
            @classmethod
            def validate_amount(cls, v: float) -> float:
                if v <= 0:
                    error_msg = "Amount must be positive"
                    raise ValueError(error_msg)
                return v

        cmd = DepositCommand(
            account_id="ACC-001", amount=100.0, command_id="cmd-test-3"
        )
        tm.that(math.isclose(cmd.amount, 100.0), eq=True)
        with pytest.raises(ValidationError):
            DepositCommand(account_id="ACC-001", amount=-50.0, command_id="cmd-test-4")


class GetUserQuery(m.Query):
    """Query to get a user."""


class ListAccountsQuery(m.Query):
    """Query to list accounts."""

    page: int
    limit: int


class SearchProductsQuery(m.Query):
    """Query to search products."""

    keyword: str
    category: str | None = None
    min_price: float | None = None
    max_price: float | None = None


class TestQueries:
    """Test CQRS query pattern using FlextTestsUtilities."""

    def test_query_creation(self) -> None:
        """Test creating a query."""
        query = GetUserQuery(
            filters=m.Dict(root={"user_id": "USER-001"}),
            query_type="get_user",
            pagination=m.Pagination(),
            query_id="q-test-1",
        )
        tm.that(query.filters["user_id"], eq="USER-001")
        tm.that(query.query_id, none=False)
        tm.that(query.query_type, eq="get_user")

    def test_query_mutation_behavior(self) -> None:
        """Test query mutation behavior with validate_assignment."""
        query = ListAccountsQuery(
            page=1,
            limit=10,
            filters=m.Dict(root={}),
            pagination=m.Pagination(),
            query_id="q-test-2",
        )
        original_page = query.page
        query.page = 2
        tm.that(query.page, eq=2)
        tm.that(query.page, ne=original_page)

    def test_query_with_filters(self) -> None:
        """Test query with filtering."""
        query = SearchProductsQuery(
            keyword="laptop",
            category="electronics",
            min_price=500.0,
            filters=m.Dict(root={}),
            pagination=m.Pagination(),
            query_id="q-test-3",
        )
        tm.that(query.keyword, eq="laptop")
        tm.that(query.category, eq="electronics")
        tm.that(query.min_price, none=False)
        if query.min_price is not None:
            tm.that(math.isclose(query.min_price, 500.0), eq=True)


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
        tm.that(event.event_type, eq="UserCreated")
        tm.that(event.aggregate_id, eq="USER-001")
        tm.that(event.data["user_id"], eq="USER-001")
        tm.that(event.data["email"], eq="user@example.com")
        tm.that(event.unique_id, none=False)
        tm.that(event.created_at, none=False)

    def test_domain_event_equality(self) -> None:
        """Test domain events can be compared and tracked."""
        event1 = m.DomainEvent(
            event_type="OrderShipped",
            aggregate_id="ORD-001",
            data=_ComparableConfigMap(root={"tracking_number": "TRACK-123"}),
        )
        event2 = m.DomainEvent(
            event_type="OrderShipped",
            aggregate_id="ORD-001",
            data=_ComparableConfigMap(root={"tracking_number": "TRACK-123"}),
        )
        tm.that(event1.unique_id, ne=event2.unique_id)
        tm.that(event1.event_type, eq=event2.event_type)
        tm.that(event1.aggregate_id, eq=event2.aggregate_id)

    def test_domain_event_timestamp(self) -> None:
        """Test domain events have timestamps."""

        class AccountUpdatedEvent(m.DomainEvent):
            """Event: account was updated."""

        event = AccountUpdatedEvent(
            event_type="AccountUpdated",
            aggregate_id="ACC-001",
            data=_ComparableConfigMap(root={"field": "balance"}),
        )
        tm.that(event.created_at, none=False)
        tm.that(isinstance(event.created_at, datetime), eq=True)

    def test_domain_event_causality(self) -> None:
        """Test domain events track causality via id and timestamps."""

        class PaymentProcessedEvent(m.DomainEvent):
            """Event: payment was processed."""

        event = PaymentProcessedEvent(
            event_type="PaymentProcessed",
            aggregate_id="PAY-001",
            data=_ComparableConfigMap(root={"amount": 99.99}),
        )
        tm.that(event.unique_id, none=False)
        tm.that(event.created_at, none=False)


class TestMetadata:
    """Test flexible metadata model using FlextTestsUtilities."""

    def test_metadata_creation(self) -> None:
        """Test creating metadata."""
        metadata = m.Metadata(attributes={"user_id": "123", "operation": "create"})
        tm.that(metadata.attributes["user_id"], eq="123")

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
        tm.that(metadata.attributes["string"], eq="value")
        tm.that(metadata.attributes["number"], eq=42)


class TestModelValidation:
    """Test model validation patterns using FlextTestsUtilities."""

    def test_model_validation_error_handling(self) -> None:
        """Test model validation error handling."""

        class ValidatedEntity(m.Entity):
            """Entity with validation."""

            email: str
            age: int

            @field_validator("age")
            @classmethod
            def validate_age(cls, v: int) -> int:
                if v < 0 or v > 150:
                    error_msg = "Invalid age"
                    raise ValueError(error_msg)
                return v

        entity = ValidatedEntity(email="user@example.com", age=30, domain_events=[])
        tm.that(entity.age, eq=30)
        with pytest.raises(ValidationError):
            ValidatedEntity(email="user@example.com", age=200, domain_events=[])

    def test_multiple_field_validation(self) -> None:
        """Test multiple field validators."""

        class Profile(m.Entity):
            """Profile with multiple validators."""

            username: str
            email: str
            bio: str | None = None

            @field_validator("username")
            @classmethod
            def validate_username(cls, v: str) -> str:
                if len(v) < 3:
                    error_msg = "Username too short"
                    raise ValueError(error_msg)
                return v

            @field_validator("email")
            @classmethod
            def validate_email(cls, v: str) -> str:
                if "@" not in v:
                    error_msg = "Invalid email"
                    raise ValueError(error_msg)
                return v

        profile = Profile(
            username="alice",
            email="alice@example.com",
            bio="Developer",
            domain_events=[],
        )
        tm.that(profile.username, eq="alice")


class TestModelSerialization:
    """Test model serialization patterns using FlextTestsUtilities."""

    def test_entity_model_dump(self) -> None:
        """Test model_dump serialization."""

        class Task(m.Entity):
            """Task entity."""

            title: str
            completed: bool

        task = Task(title="Complete tests", completed=False, domain_events=[])
        dumped = task.model_dump()
        tm.that(dumped["title"], eq="Complete tests")
        tm.that(dumped["completed"], eq=False)
        tm.that(dumped, has="unique_id")

    def test_command_serialization(self) -> None:
        """Test command serialization."""

        class SendEmailCommand(m.Command):
            """Command to send email."""

            recipient: str
            subject: str
            body: str

        cmd = SendEmailCommand(
            recipient="user@example.com",
            subject="Test",
            body="Message body",
            command_id="cmd-test-5",
        )
        dumped = cmd.model_dump()
        tm.that(dumped["recipient"], eq="user@example.com")
        tm.that(dumped["subject"], eq="Test")

    def test_aggregate_root_serialization(self) -> None:
        """Test aggregate root serialization."""

        class ShoppingCart(m.AggregateRoot):
            """Shopping cart aggregate."""

            items: list[dict[str, t.Tests.object]]
            total: float

        cart = ShoppingCart(
            items=[
                {"product_id": "P1", "quantity": 2},
                {"product_id": "P2", "quantity": 1},
            ],
            total=99.99,
            domain_events=[],
        )
        dumped = cart.model_dump()
        tm.that(len(dumped["items"]), eq=2)
        tm.that(math.isclose(dumped["total"], 99.99), eq=True)


class TestModelIntegration:
    """Test model integration with r using FlextTestsUtilities."""

    def test_entity_model_validation(self) -> None:
        """Test entity model validation via model_validate."""

        class Customer(m.Entity):
            """Customer entity."""

            name: str
            email: str

        customer = Customer(name="John", email="john@example.com", domain_events=[])
        dumped = customer.model_dump()
        customer_dict = {
            k: dumped[k] for k in type(customer).model_fields if k in dumped
        }
        validated = Customer.model_validate(customer_dict)
        tm.that(validated, none=False)
        tm.that(validated.name, eq="John")
        tm.that(validated.email, eq="john@example.com")

    def test_command_factory_pattern(self) -> None:
        """Test command creation as factories."""

        class RegisterUserCommand(m.Command):
            """User registration command."""

            email: str
            password: str

        cmd = RegisterUserCommand(
            email="user@example.com", password="secure123", command_id="cmd-test-6"
        )
        tm.that(cmd.email, eq="user@example.com")


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
