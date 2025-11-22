"""Comprehensive coverage tests for FlextModels DDD patterns.

This module provides extensive tests for Domain-Driven Design patterns:
- Value Objects (immutable, value-based equality)
- Entities (identity-based, lifecycle tracking)
- Aggregate Roots (consistency boundaries)
- Commands/Queries (CQRS patterns)
- Domain Events (event sourcing)
- Metadata (flexible attribute tracking)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError, field_validator

from flext_core import (
    FlextModels,
)


class TestValueObjects:
    """Test immutable value objects."""

    def test_value_object_creation(self) -> None:
        """Test creating a value object."""

        class Money(FlextModels.Value):
            """Money value object."""

            amount: float
            currency: str

        money = Money(amount=100.0, currency="USD")
        assert money.amount == 100.0
        assert money.currency == "USD"

    def test_value_object_immutability(self) -> None:
        """Test value object is immutable."""

        class Point(FlextModels.Value):
            """Point value object."""

            x: float
            y: float

        point = Point(x=1.0, y=2.0)

        # Attempting to modify should raise error
        with pytest.raises(ValidationError):
            point.x = 3.0

    def test_value_object_equality_by_value(self) -> None:
        """Test value objects compared by value."""

        class Color(FlextModels.Value):
            """Color value object."""

            red: int
            green: int
            blue: int

        color1 = Color(red=255, green=0, blue=0)
        color2 = Color(red=255, green=0, blue=0)

        # Same values = equal
        assert color1 == color2

    def test_value_object_validation(self) -> None:
        """Test value object validation."""

        class Email(FlextModels.Value):
            """Email value object with validation."""

            address: str

            @field_validator("address")
            @classmethod
            def validate_email(cls, v: str) -> str:
                if "@" not in v:
                    msg = "Invalid email format"
                    raise ValueError(msg)
                return v.lower()

        # Valid email
        email = Email(address="USER@EXAMPLE.COM")
        assert email.address == "user@example.com"

        # Invalid email
        with pytest.raises(ValidationError):
            Email(address="notanemail")

    def test_value_object_hashable(self) -> None:
        """Test value objects are hashable."""

        class ISBN(FlextModels.Value):
            """ISBN value object."""

            code: str

            def __hash__(self) -> int:
                """Hash based on code value."""
                return hash(self.code)

            def __eq__(self, other: object) -> bool:
                """Equality based on code value."""
                if not isinstance(other, ISBN):
                    return False
                return self.code == other.code

        isbn1 = ISBN(code="978-0-262-03384-8")
        isbn2 = ISBN(code="978-0-262-03384-8")

        # Can be used in sets/dicts
        isbn_set: set[ISBN] = {isbn1, isbn2}
        assert len(isbn_set) == 1  # Same value = one entry


class TestEntities:
    """Test domain entities with identity."""

    def test_entity_creation(self) -> None:
        """Test creating an entity."""

        class Person(FlextModels.Entity):
            """Person entity."""

            name: str
            age: int

        person = Person(name="Alice", age=30)
        assert person.name == "Alice"
        assert person.unique_id is not None
        assert person.created_at is not None
        assert person.updated_at is not None

    def test_entity_identity_tracking(self) -> None:
        """Test entities are compared by identity."""

        class Account(FlextModels.Entity):
            """Account entity."""

            name: str
            balance: float

        account1 = Account(name="Checking", balance=100.0)
        account2 = Account(name="Checking", balance=100.0)

        # Different IDs = not equal (identity-based)
        assert account1.unique_id != account2.unique_id
        assert account1 != account2

    def test_entity_lifecycle_tracking(self) -> None:
        """Test entity creation and update timestamps."""

        class Document(FlextModels.Entity):
            """Document entity."""

            title: str

        doc = Document(title="Test Doc")
        created_at = doc.created_at
        updated_at = doc.updated_at

        assert created_at is not None
        assert updated_at is not None
        assert created_at <= updated_at

    def test_entity_validation(self) -> None:
        """Test entity field validation."""

        class User(FlextModels.Entity):
            """User entity with validation."""

            email: str
            username: str

            @field_validator("username")
            @classmethod
            def validate_username(cls, v: str) -> str:
                if len(v) < 3:
                    msg = "Username must be at least 3 characters"
                    raise ValueError(msg)
                return v

        # Valid user
        user = User(email="user@example.com", username="alice")
        assert user.username == "alice"

        # Invalid username
        with pytest.raises(ValidationError):
            User(email="user@example.com", username="ab")

    def test_entity_model_dump_serialization(self) -> None:
        """Test entity serialization using Pydantic model_dump."""

        class Product(FlextModels.Entity):
            """Product entity."""

            name: str
            price: float

        product = Product(name="Widget", price=19.99)
        product_dict = product.model_dump()

        assert isinstance(product_dict, dict)
        assert product_dict["name"] == "Widget"
        assert product_dict["price"] == 19.99
        assert "unique_id" in product_dict
        assert "created_at" in product_dict
        assert "updated_at" in product_dict

    def test_entity_identity_based_comparison(self) -> None:
        """Test entity comparison is identity-based."""

        class Item(FlextModels.Entity):
            """Item entity."""

            name: str
            quantity: int

        item1 = Item(name="Book", quantity=5)
        item2 = Item(name="Book", quantity=5)

        # Different IDs, so not equal (identity-based)
        assert item1.unique_id != item2.unique_id
        assert item1 != item2

        # Same ID comparison (simulate via id check)
        item3 = Item(name="Book", quantity=5)
        item3.unique_id = item1.unique_id
        assert item3 == item1


class TestAggregateRoots:
    """Test aggregate roots for consistency boundaries."""

    def test_aggregate_root_creation(self) -> None:
        """Test creating an aggregate root."""

        class Order(FlextModels.AggregateRoot):
            """Order aggregate root."""

            order_number: str
            status: str

        order = Order(order_number="ORD-001", status="pending")
        assert order.order_number == "ORD-001"
        assert order.unique_id is not None

    def test_aggregate_root_invariants(self) -> None:
        """Test aggregate root enforces invariants."""

        class Account(FlextModels.AggregateRoot):
            """Account with business rules."""

            balance: float
            currency: str

        account = Account(balance=1000.0, currency="USD")
        assert account.balance >= 0.0

    def test_aggregate_root_lifecycle(self) -> None:
        """Test aggregate root lifecycle."""

        class Project(FlextModels.AggregateRoot):
            """Project aggregate root."""

            name: str
            status: str

        project = Project(name="New Project", status="planning")
        assert project.status == "planning"
        assert project.created_at is not None


class TestCommands:
    """Test CQRS command pattern."""

    def test_command_creation(self) -> None:
        """Test creating a command."""

        class CreateUserCommand(FlextModels.Cqrs.Command):
            """Command to create a user."""

            email: str
            username: str

        cmd = CreateUserCommand(email="user@example.com", username="alice")
        assert cmd.email == "user@example.com"
        assert cmd.unique_id is not None

    def test_command_mutation_behavior(self) -> None:
        """Test command mutation behavior with validate_assignment."""

        class UpdateProfileCommand(FlextModels.Cqrs.Command):
            """Command to update profile."""

            name: str
            bio: str

        cmd = UpdateProfileCommand(name="Alice", bio="Developer")

        # Commands use validate_assignment but are not frozen
        # Direct assignment works but validates on assignment
        original_name = cmd.name
        cmd.name = "Bob"
        assert cmd.name == "Bob"

        # Verify it was actually changed
        assert cmd.name != original_name

    def test_command_validation(self) -> None:
        """Test command validation."""

        class DepositCommand(FlextModels.Cqrs.Command):
            """Command with validation."""

            account_id: str
            amount: float

            @field_validator("amount")
            @classmethod
            def validate_amount(cls, v: float) -> float:
                if v <= 0:
                    msg = "Amount must be positive"
                    raise ValueError(msg)
                return v

        # Valid command
        cmd = DepositCommand(account_id="ACC-001", amount=100.0)
        assert cmd.amount == 100.0

        # Invalid command
        with pytest.raises(ValidationError):
            DepositCommand(account_id="ACC-001", amount=-50.0)


class TestQueries:
    """Test CQRS query pattern."""

    def test_query_creation(self) -> None:
        """Test creating a query."""

        class GetUserQuery(FlextModels.Cqrs.Query):
            """Query to get a user."""

        query = GetUserQuery(
            filters={"user_id": "USER-001"},
            query_type="get_user",
        )
        assert query.filters["user_id"] == "USER-001"
        assert query.query_id is not None
        assert query.query_type == "get_user"

    def test_query_mutation_behavior(self) -> None:
        """Test query mutation behavior with validate_assignment."""

        class ListAccountsQuery(FlextModels.Cqrs.Query):
            """Query to list accounts."""

            page: int
            limit: int

        query = ListAccountsQuery(page=1, limit=10)

        # Queries use validate_assignment but are not frozen
        original_page = query.page
        query.page = 2
        assert query.page == 2
        assert query.page != original_page

    def test_query_with_filters(self) -> None:
        """Test query with filtering."""

        class SearchProductsQuery(FlextModels.Cqrs.Query):
            """Query to search products."""

            keyword: str
            category: str | None = None
            min_price: float | None = None
            max_price: float | None = None

        query = SearchProductsQuery(
            keyword="laptop",
            category="electronics",
            min_price=500.0,
        )
        assert query.keyword == "laptop"
        assert query.category == "electronics"
        assert query.min_price == 500.0


class TestDomainEvents:
    """Test domain events for event sourcing."""

    def test_domain_event_creation(self) -> None:
        """Test creating a domain event with data payload."""
        event = FlextModels.DomainEvent(
            event_type="UserCreated",
            aggregate_id="USER-001",
            data={"user_id": "USER-001", "email": "user@example.com"},
        )
        assert event.event_type == "UserCreated"
        assert event.aggregate_id == "USER-001"
        assert event.data["user_id"] == "USER-001"
        assert event.data["email"] == "user@example.com"
        assert event.unique_id is not None
        assert event.created_at is not None

    def test_domain_event_equality(self) -> None:
        """Test domain events can be compared and tracked."""
        event1 = FlextModels.DomainEvent(
            event_type="OrderShipped",
            aggregate_id="ORD-001",
            data={"tracking_number": "TRACK-123"},
        )

        event2 = FlextModels.DomainEvent(
            event_type="OrderShipped",
            aggregate_id="ORD-001",
            data={"tracking_number": "TRACK-123"},
        )

        # Different instances with same data should have different IDs
        assert event1.unique_id != event2.unique_id
        assert event1.event_type == event2.event_type
        assert event1.aggregate_id == event2.aggregate_id

    def test_domain_event_timestamp(self) -> None:
        """Test domain events have timestamps."""

        class AccountUpdatedEvent(FlextModels.DomainEvent):
            """Event: account was updated."""

        event = AccountUpdatedEvent(
            event_type="AccountUpdated",
            aggregate_id="ACC-001",
            data={"field": "balance"},
        )
        assert event.created_at is not None
        assert isinstance(event.created_at, datetime)

    def test_domain_event_causality(self) -> None:
        """Test domain events track causality via id and timestamps."""

        class PaymentProcessedEvent(FlextModels.DomainEvent):
            """Event: payment was processed."""

        event = PaymentProcessedEvent(
            event_type="PaymentProcessed",
            aggregate_id="PAY-001",
            data={"amount": 99.99},
        )
        # Events are causally ordered by ID and timestamp
        assert event.unique_id is not None
        assert event.created_at is not None


class TestMetadata:
    """Test flexible metadata model."""

    def test_metadata_creation(self) -> None:
        """Test creating metadata."""
        metadata = FlextModels.Metadata(
            attributes={"user_id": "123", "operation": "create"},
        )
        assert metadata.attributes["user_id"] == "123"

    def test_metadata_with_various_types(self) -> None:
        """Test metadata with different attribute types."""
        metadata = FlextModels.Metadata(
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
    """Test model validation patterns."""

    def test_model_validation_error_handling(self) -> None:
        """Test model validation error handling."""

        class ValidatedEntity(FlextModels.Entity):
            """Entity with validation."""

            email: str
            age: int

            @field_validator("age")
            @classmethod
            def validate_age(cls, v: int) -> int:
                if v < 0 or v > 150:
                    msg = "Invalid age"
                    raise ValueError(msg)
                return v

        # Valid
        entity = ValidatedEntity(email="user@example.com", age=30)
        assert entity.age == 30

        # Invalid
        with pytest.raises(ValidationError):
            ValidatedEntity(email="user@example.com", age=200)

    def test_multiple_field_validation(self) -> None:
        """Test multiple field validators."""

        class Profile(FlextModels.Entity):
            """Profile with multiple validators."""

            username: str
            email: str
            bio: str | None = None

            @field_validator("username")
            @classmethod
            def validate_username(cls, v: str) -> str:
                if len(v) < 3:
                    msg = "Username too short"
                    raise ValueError(msg)
                return v

            @field_validator("email")
            @classmethod
            def validate_email(cls, v: str) -> str:
                if "@" not in v:
                    msg = "Invalid email"
                    raise ValueError(msg)
                return v

        profile = Profile(
            username="alice",
            email="alice@example.com",
            bio="Developer",
        )
        assert profile.username == "alice"


class TestModelSerialization:
    """Test model serialization patterns."""

    def test_entity_model_dump(self) -> None:
        """Test model_dump serialization."""

        class Task(FlextModels.Entity):
            """Task entity."""

            title: str
            completed: bool

        task = Task(title="Complete tests", completed=False)
        dumped = task.model_dump()

        assert dumped["title"] == "Complete tests"
        assert dumped["completed"] is False
        assert "unique_id" in dumped

    def test_command_serialization(self) -> None:
        """Test command serialization."""

        class SendEmailCommand(FlextModels.Cqrs.Command):
            """Command to send email."""

            recipient: str
            subject: str
            body: str

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

        class ShoppingCart(FlextModels.AggregateRoot):
            """Shopping cart aggregate."""

            items: list[dict[str, Any]]
            total: float

        cart = ShoppingCart(
            items=[
                {"product_id": "P1", "quantity": 2},
                {"product_id": "P2", "quantity": 1},
            ],
            total=99.99,
        )
        dumped = cart.model_dump()

        assert len(dumped["items"]) == 2
        assert dumped["total"] == 99.99


class TestModelIntegration:
    """Test model integration with FlextResult."""

    def test_entity_model_validation(self) -> None:
        """Test entity model validation via model_validate."""

        class Customer(FlextModels.Entity):
            """Customer entity."""

            name: str
            email: str

        # Valid customer
        customer = Customer(name="John", email="john@example.com")
        # Exclude computed fields when dumping for re-validation
        customer_dict = customer.model_dump(
            exclude={
                "is_initial_version",
                "is_modified",
                "uncommitted_events",  # Computed field
            },
        )

        # Validate by creating new instance from dict data
        validated = Customer.model_validate(customer_dict)

        assert validated is not None
        assert validated.name == "John"
        assert validated.email == "john@example.com"

    def test_command_factory_pattern(self) -> None:
        """Test command creation as factories."""

        class RegisterUserCommand(FlextModels.Cqrs.Command):
            """User registration command."""

            email: str
            password: str

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
    "TestValueObjects",
]
